//! Reusable native device CSR conversion, planned once by column counts.
use crate::{
    array::{Array, Dtype, Layout},
    rank_support::*,
    sparse_ovr::{CscArrays, CscBlock},
    staging::BatchStreams,
};
use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*};

struct Buffers<'py> {
    data: Bound<'py, PyAny>,
    indices: Bound<'py, PyAny>,
    capacity: usize,
}

pub struct CsrWindows<'py> {
    data: Bound<'py, PyAny>,
    indices: Bound<'py, PyAny>,
    indptr: Bound<'py, PyAny>,
    d: Array,
    i: Array,
    p: Array,
    rows: usize,
    offsets: Vec<u64>,
    buffers: Vec<Option<Buffers<'py>>>,
}

impl<'py> CsrWindows<'py> {
    pub fn new(
        cp: &Bound<'py, PyModule>,
        source: &Bound<'py, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        let data = source.getattr("data")?;
        let indices = source.getattr("indices")?;
        let indptr = source.getattr("indptr")?;
        let d = floating(&data, cp, "CSR data", Layout::C)?;
        let i = vector(&indices, cp, "CSR indices", None)?;
        let p = vector(&indptr, cp, "CSR indptr", None)?;
        let index_wide = integer(&i)?;
        let pointer_wide = integer(&p)?;
        if i.len != d.len || p.len != rows as u64 + 1 {
            return Err(PyValueError::new_err("inconsistent CSR buffer lengths"));
        }
        // This guard also covers a failed host copy after the histogram launch.
        let mut guard = BatchStreams::caller(cp)?;
        let _scope = guard.enter(0)?;
        let counts = empty(cp, &[cols], "uint64", "C", true)?;
        guard.retain(0, counts.clone());
        let count_array = vector(&counts, cp, "CSR column counts", Some(Dtype::U64))?;
        launch(
            cp,
            "sparse_csr_histogram",
            rows as u64,
            stream(cp)?,
            &[&d, &i, &p, &count_array],
            &mut [
                Arg::P(i.pointer),
                Arg::P(p.pointer),
                Arg::P(count_array.pointer),
                Arg::N(rows as u64),
                Arg::N(cols as u64),
                Arg::N(d.len),
                Arg::U(index_wide),
                Arg::U(pointer_wide),
            ],
        )?;
        let host = cp.call_method1("asnumpy", (&counts,))?;
        let counts = PyBuffer::<u64>::get(&host)?.to_vec(cp.py())?;
        let offsets = crate::host_parallel::run(cp.py(), cols, |_| {
            let mut offsets = Vec::with_capacity(cols + 1);
            offsets.push(0u64);
            for count in counts {
                let next = offsets
                    .last()
                    .unwrap()
                    .checked_add(count)
                    .ok_or_else(|| PyValueError::new_err("CSR column counts overflow"))?;
                offsets.push(next);
            }
            Ok::<_, PyErr>(offsets)
        })??;
        guard.finish()?;
        Ok(Self {
            data,
            indices,
            indptr,
            d,
            i,
            p,
            rows,
            offsets,
            buffers: (0..4).map(|_| None).collect(),
        })
    }

    pub fn window(
        &mut self,
        cp: &Bound<'py, PyModule>,
        first: usize,
        stop: usize,
        staging: &mut BatchStreams<'py>,
        slot: usize,
    ) -> PyResult<CscBlock<'py>> {
        if stop < first || stop >= self.offsets.len() || slot >= self.buffers.len() {
            return Err(PyValueError::new_err("invalid device CSR window"));
        }
        let base = self.offsets[first];
        let offsets: Vec<u64> = self.offsets[first..=stop]
            .iter()
            .map(|&v| v - base)
            .collect();
        let count = usize::try_from(*offsets.last().unwrap())
            .map_err(|_| PyValueError::new_err("CSR window is too large"))?;
        let row_dtype = if self.rows <= i32::MAX as usize {
            Dtype::I32
        } else {
            Dtype::I64
        };
        if self.buffers[slot]
            .as_ref()
            .is_none_or(|buffers| buffers.capacity < count)
        {
            let data = empty(cp, &[count], self.d.dtype.name(), "C", false)?;
            let indices = empty(cp, &[count], row_dtype.name(), "C", false)?;
            self.buffers[slot] = Some(Buffers {
                data,
                indices,
                capacity: count,
            });
        }
        let buffers = self.buffers[slot].as_ref().unwrap();
        for input in [
            &self.data,
            &self.indices,
            &self.indptr,
            &buffers.data,
            &buffers.indices,
        ] {
            staging.retain(slot, input.clone());
        }
        // SAFETY: initialized u64 offsets have no padding and their immutable
        // native owner outlives both synchronous pinned-memory copies.
        let bytes =
            unsafe { std::slice::from_raw_parts(offsets.as_ptr().cast::<u8>(), offsets.len() * 8) };
        let indptr = staging.upload(slot, 3, bytes, Dtype::I64, &[offsets.len()], "C")?;
        let positions = staging.upload(slot, 4, bytes, Dtype::U64, &[offsets.len()], "C")?;
        let out_d = floating(&buffers.data, cp, "CSC window data", Layout::C)?;
        let out_i = vector(&buffers.indices, cp, "CSC window indices", Some(row_dtype))?;
        let positions_array = vector(&positions, cp, "CSC write positions", Some(Dtype::U64))?;
        launch(
            cp,
            "sparse_csr_scatter",
            self.rows as u64,
            stream(cp)?,
            &[&self.d, &self.i, &self.p, &out_d, &out_i],
            &mut [
                Arg::P(self.d.pointer),
                Arg::P(self.i.pointer),
                Arg::P(self.p.pointer),
                Arg::P(positions_array.pointer),
                Arg::P(out_d.pointer),
                Arg::P(out_i.pointer),
                Arg::N(self.rows as u64),
                Arg::N(first as u64),
                Arg::N(stop as u64),
                Arg::N(self.d.len),
                Arg::N(count as u64),
                Arg::U(wide(self.d.dtype)?),
                Arg::U(integer(&self.i)?),
                Arg::U(integer(&self.p)?),
                Arg::U(u32::from(row_dtype == Dtype::I64)),
            ],
        )?;
        Ok(CscBlock::Prepared {
            arrays: CscArrays {
                data: buffers.data.clone(),
                indices: buffers.indices.clone(),
                indptr,
            },
            offsets,
        })
    }
}
