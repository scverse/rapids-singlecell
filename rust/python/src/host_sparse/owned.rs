//! Bounded CSR snapshots converted once into reusable native CSC storage.
use super::*;

const SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;

pub(super) struct Rows<T, I> {
    pub(super) values: Vec<T>,
    pub(super) columns: Vec<I>,
    pub(super) offsets: Vec<usize>,
}

pub(super) enum OwnedCsr {
    F32I32(Rows<f32, i32>),
    F32I64(Rows<f32, i64>),
    F64I32(Rows<f64, i32>),
    F64I64(Rows<f64, i64>),
}

fn copy_range<T: Scalar>(
    py: Python<'_>,
    source: &PyBuffer<T>,
    first: usize,
    output: &mut [T],
) -> PyResult<()> {
    if output.is_empty() {
        return Ok(());
    }
    let mut length = output.len() as isize;
    let mut stride = std::mem::size_of::<T>() as isize;
    let bytes = std::mem::size_of_val(output) as isize;
    // SAFETY: the typed contiguous buffer remains exported and attached. The
    // caller validated the entire source interval, and output is a disjoint
    // initialized native allocation. This borrowing descriptor is not released.
    let status = unsafe {
        let view = pyo3::ffi::Py_buffer {
            buf: source.get_ptr(&[first]),
            obj: std::ptr::null_mut(),
            len: bytes,
            itemsize: stride,
            readonly: 1,
            ndim: 1,
            format: std::ptr::null_mut(),
            shape: &mut length,
            strides: &mut stride,
            suboffsets: std::ptr::null_mut(),
            internal: std::ptr::null_mut(),
        };
        pyo3::ffi::PyBuffer_ToContiguous(output.as_mut_ptr().cast(), &view, bytes, b'C' as _)
    };
    if status == 0 {
        Ok(())
    } else {
        Err(PyErr::fetch(py))
    }
}

fn snapshot<T: Scalar, I: Index>(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    spans: &[(usize, usize)],
    n_columns: usize,
) -> PyResult<Option<Rows<T, I>>> {
    let values = PyBuffer::<T>::get(data)?;
    let columns = PyBuffer::<I>::get(indices)?;
    if values.dimensions() != 1
        || columns.dimensions() != 1
        || !values.is_c_contiguous()
        || !columns.is_c_contiguous()
        || values.item_count() != columns.item_count()
    {
        return Err(PyValueError::new_err(
            "invalid contiguous CSR snapshot buffers",
        ));
    }
    if !spans.is_empty() && I::try_from(spans.len() - 1).is_err() {
        return Err(PyValueError::new_err("CSC row indices exceed index dtype"));
    }
    let mut count = 0usize;
    for &(first, stop) in spans {
        if stop < first || stop > values.item_count() {
            return Err(PyValueError::new_err("invalid CSR row span"));
        }
        count = count
            .checked_add(stop - first)
            .ok_or_else(|| PyValueError::new_err("CSR snapshot count overflow"))?;
    }
    // Bound peak native payload, including the CSR snapshot, conversion
    // vectors, per-partition histograms/scatter slices, CSC output, and row
    // metadata. Two bounded window copies also fit within this allowance.
    let parts = count.div_ceil(8192).clamp(1, host_parallel::MAX_WORKERS);
    let bytes = count
        .checked_mul(2 * std::mem::size_of::<T>() + 3 * std::mem::size_of::<I>() + 8)
        .and_then(|bytes| {
            n_columns
                .checked_add(1)?
                .checked_mul(parts * 40 + 2 * (8 + std::mem::size_of::<I>()))?
                .checked_add(bytes)
        })
        .and_then(|bytes| {
            spans
                .len()
                .checked_add(1)?
                .checked_mul(32)?
                .checked_add(bytes)
        })
        .and_then(|bytes| bytes.checked_add(parts * 128));
    if bytes.is_none_or(|bytes| bytes > SNAPSHOT_BYTES) {
        return Ok(None);
    }
    let mut owned = Rows {
        values: allocate::<T>(count)?,
        columns: allocate::<I>(count)?,
        offsets: allocate::<usize>(spans.len() + 1)?,
    };
    for (row, &(first, stop)) in spans.iter().enumerate() {
        owned.offsets[row + 1] = owned.offsets[row] + stop - first;
    }
    // Full CSR inputs become two bulk copies; gapped spans merge only when
    // adjacent in the caller's storage, preserving every original row boundary.
    let mut row = 0;
    while row < spans.len() {
        let first_row = row;
        let (first, mut stop) = spans[row];
        row += 1;
        while row < spans.len() && spans[row].0 == stop {
            stop = spans[row].1;
            row += 1;
        }
        let target = owned.offsets[first_row]..owned.offsets[row];
        copy_range(py, &values, first, &mut owned.values[target.clone()])?;
        copy_range(py, &columns, first, &mut owned.columns[target])?;
    }
    Ok(Some(owned))
}

impl OwnedCsr {
    pub(super) fn snapshot(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        spans: &[(usize, usize)],
        columns: usize,
    ) -> PyResult<Option<Self>> {
        let dtype: String = data.getattr("dtype")?.getattr("name")?.extract()?;
        let index: String = indices.getattr("dtype")?.getattr("name")?.extract()?;
        match (dtype.as_str(), index.as_str()) {
            ("float32", "int32") => {
                snapshot(py, data, indices, spans, columns).map(|v| v.map(Self::F32I32))
            }
            ("float32", "int64") => {
                snapshot(py, data, indices, spans, columns).map(|v| v.map(Self::F32I64))
            }
            ("float64", "int32") => {
                snapshot(py, data, indices, spans, columns).map(|v| v.map(Self::F64I32))
            }
            ("float64", "int64") => {
                snapshot(py, data, indices, spans, columns).map(|v| v.map(Self::F64I64))
            }
            _ => Err(PyTypeError::new_err(
                "CSR staging requires float32/float64 and int32/int64",
            )),
        }
    }

    pub(super) fn into_csc(self, py: Python<'_>, columns: usize) -> PyResult<CscWindow> {
        match self {
            Self::F32I32(rows) => rows.into_csc(py, columns),
            Self::F32I64(rows) => rows.into_csc(py, columns),
            Self::F64I32(rows) => rows.into_csc(py, columns),
            Self::F64I64(rows) => rows.into_csc(py, columns),
        }
    }
}

impl<T: Scalar, I: Index> Rows<T, I> {
    fn into_csc(self, py: Python<'_>, columns: usize) -> PyResult<CscWindow> {
        host_parallel::run(py, self.values.len(), |plan| {
            let mut column_ids = allocate::<usize>(self.columns.len())?;
            for (output, value) in column_ids.iter_mut().zip(self.columns) {
                let column = usize::try_from(value.into())
                    .ok()
                    .filter(|&column| column < columns)
                    .ok_or_else(|| {
                        PyValueError::new_err("CSR column index changed during staging")
                    })?;
                *output = column;
            }
            let mut row_ids = allocate::<I>(self.values.len())?;
            plan.for_each_chunk(&mut row_ids, |first, output| {
                let mut row = self.offsets.partition_point(|&offset| offset <= first) - 1;
                for (local, value) in output.iter_mut().enumerate() {
                    while self.offsets[row + 1] <= first + local {
                        row += 1;
                    }
                    *value = I::try_from(row)
                        .ok()
                        .expect("row dtype validated before snapshot");
                }
            });
            convert_native(
                Snapshot {
                    values: self.values,
                    columns: column_ids,
                    rows: row_ids,
                },
                columns,
                plan,
            )
        })?
    }
}
