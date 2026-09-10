//! Validated PyO3 bindings for native preprocessing kernels.
#![allow(clippy::too_many_arguments)] // Existing public extension signatures.

use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use std::ffi::c_void;

pub(crate) struct Launch<'py> {
    cupy: Bound<'py, PyModule>,
    arrays: Vec<(Array, bool)>,
    pub(crate) value: Dtype,
    pub(crate) index: Option<Dtype>,
}

impl<'py> Launch<'py> {
    pub(crate) fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            cupy: py.import("cupy")?,
            arrays: Vec::new(),
            value: Dtype::F32,
            index: None,
        })
    }

    pub(crate) fn array(
        &mut self,
        object: &Bound<'_, PyAny>,
        name: &str,
        dtype: Option<Dtype>,
        len: Option<u64>,
        vector: bool,
        layout: Layout,
        writable: bool,
    ) -> PyResult<u64> {
        let array = Array::read(object, &self.cupy, name, dtype, None, layout)?;
        if vector {
            array.require_vector(name)?;
        }
        if len.is_some_and(|len| len != array.len) {
            return Err(PyValueError::new_err(format!(
                "{name} has an incompatible length"
            )));
        }
        let pointer = array.pointer;
        self.arrays.push((array, writable));
        Ok(pointer)
    }

    pub(crate) fn data(
        &mut self,
        object: &Bound<'_, PyAny>,
        len: Option<u64>,
        vector: bool,
        fortran: bool,
        writable: bool,
    ) -> PyResult<u64> {
        let pointer = self.array(
            object,
            "data",
            None,
            len,
            vector,
            if fortran { Layout::F } else { Layout::C },
            writable,
        )?;
        self.value = self.arrays.last().unwrap().0.dtype;
        if !matches!(self.value, Dtype::F32 | Dtype::F64) {
            return Err(PyTypeError::new_err(
                "data must have dtype float32 or float64",
            ));
        }
        if fortran
            && !object
                .getattr("flags")?
                .getattr("f_contiguous")?
                .extract::<bool>()?
        {
            return Err(PyValueError::new_err("data must be Fortran contiguous"));
        }
        Ok(pointer)
    }

    pub(crate) fn vector(
        &mut self,
        object: &Bound<'_, PyAny>,
        name: &str,
        dtype: Dtype,
        len: Option<u64>,
        writable: bool,
    ) -> PyResult<u64> {
        self.array(object, name, Some(dtype), len, true, Layout::C, writable)
    }

    fn indptr(&mut self, object: &Bound<'_, PyAny>, major: u64) -> PyResult<u64> {
        let pointer = self.array(
            object,
            "indptr",
            None,
            Some(
                major
                    .checked_add(1)
                    .ok_or_else(|| PyValueError::new_err("major is too large"))?,
            ),
            true,
            Layout::C,
            false,
        )?;
        let dtype = self.arrays.last().unwrap().0.dtype;
        if !matches!(dtype, Dtype::I32 | Dtype::I64) {
            return Err(PyTypeError::new_err(
                "indptr must have dtype int32 or int64",
            ));
        }
        self.index = Some(dtype);
        Ok(pointer)
    }

    pub(crate) fn indices(&mut self, object: &Bound<'_, PyAny>, nnz: u64) -> PyResult<u64> {
        let pointer = self.array(
            object,
            "indices",
            self.index,
            Some(nnz),
            true,
            Layout::C,
            false,
        )?;
        let dtype = self.arrays.last().unwrap().0.dtype;
        if !matches!(dtype, Dtype::I32 | Dtype::I64) {
            return Err(PyTypeError::new_err(
                "indices must have dtype int32 or int64",
            ));
        }
        self.index = Some(dtype);
        Ok(pointer)
    }

    pub(crate) fn len(&self) -> u64 {
        self.arrays.last().unwrap().0.len
    }

    pub(crate) fn run(
        self,
        family: &str,
        work: u64,
        block: u32,
        rows: bool,
        stream: usize,
        mut slots: Vec<u64>,
    ) -> PyResult<()> {
        let references = self
            .arrays
            .iter()
            .map(|(array, _)| array)
            .collect::<Vec<_>>();
        let device = current_device(&self.cupy, &references)?;
        for (i, (array, writable)) in self.arrays.iter().enumerate() {
            for (other, other_writable) in &self.arrays[..i] {
                if *writable || *other_writable {
                    array.require_disjoint(other)?;
                }
            }
        }
        if work == 0 {
            return Ok(());
        }
        let value = if self.value == Dtype::F32 {
            "f32"
        } else {
            "f64"
        };
        let name = match self.index {
            Some(Dtype::I32) => format!("prep_{family}_{value}_i32"),
            Some(Dtype::I64) => format!("prep_{family}_{value}_i64"),
            _ => format!("prep_{family}_{value}"),
        };
        let grid = if rows {
            work
        } else {
            work.div_ceil(block as u64)
        }
        .min(65_535) as u32;
        // Match the original dense QC tile: each warp covers 16 cells and
        // two genes, avoiding 32 competing atomics to a single cell sum.
        let (grid_shape, block_shape) = if family == "qc_dense" {
            (
                (
                    slots[6].div_ceil(16).min(65_535) as u32,
                    slots[7].div_ceil(16).min(65_535) as u32,
                    1,
                ),
                (16, 16, 1),
            )
        } else {
            ((grid, 1, 1), (block, 1, 1))
        };
        let mut args = slots
            .iter_mut()
            .map(|slot| (slot as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        // SAFETY: each wrapper builds slots in its kernel's ABI order. Pointer,
        // u64 and f64 slots occupy eight bytes; a u32 mode reads the low four
        // bytes on CUDA's little-endian ABI. Arrays are validated and kept alive
        // by the Python caller; kernels additionally bound-check sparse indices.
        unsafe { runtime::launch(device, &name, grid_shape, block_shape, stream, &mut args) }
    }
}

fn size(rows: u64, cols: u64) -> PyResult<u64> {
    rows.checked_mul(cols)
        .filter(|&n| n <= isize::MAX as u64 / 8)
        .ok_or_else(|| PyValueError::new_err("matrix size exceeds addressable memory"))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, means, vars, *, major, minor, stream=0))]
pub fn mean_var_major(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    vars: &Bound<'_, PyAny>,
    major: u64,
    minor: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(indices, nnz)?;
    let m = call.vector(means, "means", Dtype::F64, Some(major), true)?;
    let v = call.vector(vars, "vars", Dtype::F64, Some(major), true)?;
    call.run(
        "stats_major",
        major,
        64,
        true,
        stream,
        vec![p, i, d, m, v, 0, 0, major, minor, nnz],
    )
}

#[pyfunction]
#[pyo3(signature = (indices, data, means, vars, *, nnz, stream=0))]
pub fn mean_var_minor(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    vars: &Bound<'_, PyAny>,
    nnz: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, Some(nnz), true, false, false)?;
    let i = call.indices(indices, nnz)?;
    let m = call.vector(means, "means", Dtype::F64, None, true)?;
    let minor = call.len();
    let v = call.vector(vars, "vars", Dtype::F64, Some(minor), true)?;
    call.run(
        "stats_minor",
        nnz,
        256,
        false,
        stream,
        vec![i, d, m, v, 0, 0, 0, nnz, minor, 0],
    )
}

#[pyfunction]
#[pyo3(signature = (index, data, *, means, nans, mask, nnz, stream=0))]
pub fn nan_mean_minor(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    nans: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    nnz: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, Some(nnz), true, false, false)?;
    let i = call.indices(index, nnz)?;
    let m = call.vector(means, "means", Dtype::F64, None, true)?;
    let minor = call.len();
    let n = call.vector(nans, "nans", Dtype::I32, Some(minor), true)?;
    let mask = call.vector(mask, "mask", Dtype::Bool, Some(minor), false)?;
    call.run(
        "stats_minor",
        nnz,
        32,
        false,
        stream,
        vec![i, d, m, 0, n, mask, 0, nnz, minor, 1],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, means, nans, mask, major, minor, stream=0))]
pub fn nan_mean_major(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    nans: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    major: u64,
    minor: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let m = call.vector(means, "means", Dtype::F64, Some(major), true)?;
    let n = call.vector(nans, "nans", Dtype::I32, Some(major), true)?;
    let mask = call.vector(mask, "mask", Dtype::Bool, Some(minor), false)?;
    call.run(
        "nan_major",
        major,
        64,
        true,
        stream,
        vec![p, i, d, m, 0, n, mask, major, minor, nnz],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, nrows, ncols, target_sum, stream=0))]
pub fn mul_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    nrows: u64,
    ncols: u64,
    target_sum: f64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, Some(size(nrows, ncols)?), false, false, true)?;
    let scales = 0;
    call.run(
        "norm_dense",
        nrows,
        if ncols >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![d, scales, nrows, ncols, target_sum.to_bits(), 0],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, data, *, nrows, target_sum, stream=0))]
pub fn mul_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    nrows: u64,
    target_sum: f64,
    stream: usize,
) -> PyResult<()> {
    let major = nrows;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, true)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = 0;
    let mask = 0;
    let minor = 0;
    let sums = 0;
    let scales = 0;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            sums,
            scales,
            major,
            minor,
            nnz,
            target_sum.to_bits(),
            0,
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, data, *, sums, major, stream=0))]
pub fn sum_major(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums: &Bound<'_, PyAny>,
    major: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = 0;
    let mask = 0;
    let minor = 0;
    let sums = call.vector(sums, "sums", call.value, Some(major), true)?;
    let scales = 0;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            sums,
            scales,
            major,
            minor,
            nnz,
            0.0_f64.to_bits(),
            1,
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, *, gene_is_hi, max_fraction, nrows, stream=0))]
pub fn find_hi_genes_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    gene_is_hi: &Bound<'_, PyAny>,
    max_fraction: f64,
    nrows: u64,
    stream: usize,
) -> PyResult<()> {
    let major = nrows;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(indices, nnz)?;
    call.vector(gene_is_hi, "gene_is_hi", Dtype::Bool, None, true)?;
    let minor = call.len();
    // Use native integer atomics for shared flags, then commit one bool per
    // gene. Concurrent byte stores would be a data race even for equal values.
    let _scope = runtime::StreamScope::new(&call.cupy, stream)?;
    let flags = call.cupy.call_method1("zeros", (minor, "int32"))?;
    let mask = call.vector(&flags, "flags", Dtype::I32, Some(minor), true)?;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            0,
            0,
            major,
            minor,
            nnz,
            max_fraction.to_bits(),
            4,
        ],
    )?;
    let mut commit = Launch::new(py)?;
    let source = commit.vector(&flags, "flags", Dtype::I32, Some(minor), false)?;
    let out = commit.vector(gene_is_hi, "gene_is_hi", Dtype::Bool, Some(minor), true)?;
    commit.run(
        "high_flags",
        minor,
        256,
        false,
        stream,
        vec![source, out, minor],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, *, gene_mask, nrows, tsum, stream=0))]
pub fn masked_mul_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    gene_mask: &Bound<'_, PyAny>,
    nrows: u64,
    tsum: f64,
    stream: usize,
) -> PyResult<()> {
    let major = nrows;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, true)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(indices, nnz)?;
    let mask = call.vector(gene_mask, "gene_mask", Dtype::Bool, None, false)?;
    let minor = call.len();
    let sums = 0;
    let scales = 0;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            sums,
            scales,
            major,
            minor,
            nnz,
            tsum.to_bits(),
            2,
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, *, gene_mask, sums, major, stream=0))]
pub fn masked_sum_major(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    gene_mask: &Bound<'_, PyAny>,
    sums: &Bound<'_, PyAny>,
    major: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(indices, nnz)?;
    let mask = call.vector(gene_mask, "gene_mask", Dtype::Bool, None, false)?;
    let minor = call.len();
    let sums = call.vector(sums, "sums", call.value, Some(major), true)?;
    let scales = 0;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            sums,
            scales,
            major,
            minor,
            nnz,
            0.0_f64.to_bits(),
            3,
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, data, *, scales, nrows, stream=0))]
pub fn prescaled_mul_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    scales: &Bound<'_, PyAny>,
    nrows: u64,
    stream: usize,
) -> PyResult<()> {
    let major = nrows;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, true)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = 0;
    let mask = 0;
    let minor = 0;
    let sums = 0;
    let scales = call.vector(scales, "scales", call.value, Some(major), false)?;
    call.run(
        "norm",
        major,
        if nnz / major.max(1) >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![
            p,
            i,
            d,
            mask,
            sums,
            scales,
            major,
            minor,
            nnz,
            0.0_f64.to_bits(),
            5,
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, scales, nrows, ncols, stream=0))]
pub fn prescaled_mul_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    scales: &Bound<'_, PyAny>,
    nrows: u64,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, Some(size(nrows, ncols)?), false, false, true)?;
    let scales = call.vector(scales, "scales", call.value, Some(nrows), false)?;
    call.run(
        "norm_dense",
        nrows,
        if ncols >= 256 { 256 } else { 32 },
        true,
        stream,
        vec![d, scales, nrows, ncols, 0.0_f64.to_bits(), 1],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, data, std, *, ncols, stream=0))]
pub fn csc_scale_diff(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    std: &Bound<'_, PyAny>,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, true)?;
    let nnz = call.len();
    let p = call.indptr(indptr, ncols)?;
    let s = call.vector(std, "std", call.value, Some(ncols), false)?;
    call.run(
        "scale",
        ncols,
        64,
        true,
        stream,
        vec![p, 0, d, s, 0, ncols, ncols, nnz, 0, 0],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, std, mask, *, clipper, nrows, stream=0))]
pub fn csr_scale_diff(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    std: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    clipper: f64,
    nrows: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, true)?;
    let nnz = call.len();
    let p = call.indptr(indptr, nrows)?;
    let i = call.indices(indices, nnz)?;
    let s = call.vector(std, "std", call.value, None, false)?;
    let ncols = call.len();
    let mask = call.vector(mask, "mask", Dtype::I32, Some(nrows), false)?;
    call.run(
        "scale",
        nrows,
        64,
        true,
        stream,
        vec![p, i, d, s, mask, nrows, ncols, nnz, clipper.to_bits(), 1],
    )
}

#[pyfunction]
#[pyo3(signature = (data, mean, std, mask, *, clipper, nrows, ncols, stream=0))]
pub fn dense_scale_center_diff(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    std: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    clipper: f64,
    nrows: u64,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(nrows, ncols)?;
    let d = call.data(data, Some(length), false, false, true)?;
    let s = call.vector(std, "std", call.value, Some(ncols), false)?;
    let mask = call.vector(mask, "mask", Dtype::I32, Some(nrows), false)?;
    let m = call.vector(mean, "mean", call.value, Some(ncols), false)?;
    call.run(
        "scale_dense",
        length,
        256,
        false,
        stream,
        vec![d, m, s, mask, nrows, ncols, clipper.to_bits(), 1],
    )
}

#[pyfunction]
#[pyo3(signature = (data, std, mask, *, clipper, nrows, ncols, stream=0))]
pub fn dense_scale_diff(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    std: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    clipper: f64,
    nrows: u64,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(nrows, ncols)?;
    let d = call.data(data, Some(length), false, false, true)?;
    let s = call.vector(std, "std", call.value, Some(ncols), false)?;
    let mask = call.vector(mask, "mask", Dtype::I32, Some(nrows), false)?;
    let m = 0;
    call.run(
        "scale_dense",
        length,
        256,
        false,
        stream,
        vec![d, m, s, mask, nrows, ncols, clipper.to_bits(), 0],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, sums_genes, cell_ex, gene_ex, n_genes, stream=0))]
pub fn sparse_qc_csc(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    cell_ex: &Bound<'_, PyAny>,
    gene_ex: &Bound<'_, PyAny>,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_genes;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let a = call.vector(sums_genes, "sums_genes", call.value, Some(major), true)?;
    let b = call.vector(sums_cells, "sums_cells", call.value, None, true)?;
    let minor = call.len();
    let a_ex = call.vector(gene_ex, "gene_ex", Dtype::I32, Some(major), true)?;
    let b_ex = call.vector(cell_ex, "cell_ex", Dtype::I32, Some(minor), true)?;
    let mask = 0;
    call.run(
        "qc",
        major,
        32,
        false,
        stream,
        vec![p, i, d, a, b, a_ex, b_ex, mask, major, minor, nnz, 0],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, stream=0))]
pub fn sparse_qc_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    cell_ex: &Bound<'_, PyAny>,
    gene_ex: &Bound<'_, PyAny>,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_cells;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let a = call.vector(sums_cells, "sums_cells", call.value, Some(major), true)?;
    let b = call.vector(sums_genes, "sums_genes", call.value, None, true)?;
    let minor = call.len();
    let a_ex = call.vector(cell_ex, "cell_ex", Dtype::I32, Some(major), true)?;
    let b_ex = call.vector(gene_ex, "gene_ex", Dtype::I32, Some(minor), true)?;
    let mask = 0;
    call.run(
        "qc",
        major,
        32,
        false,
        stream,
        vec![p, i, d, a, b, a_ex, b_ex, mask, major, minor, nnz, 0],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, mask, n_genes, stream=0))]
pub fn sparse_qc_csc_sub(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_genes;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let a = 0;
    let b = call.vector(sums_cells, "sums_cells", call.value, None, true)?;
    let minor = call.len();
    let a_ex = 0;
    let b_ex = 0;
    let mask = call.vector(mask, "mask", Dtype::Bool, Some(major), false)?;
    call.run(
        "qc",
        major,
        32,
        false,
        stream,
        vec![p, i, d, a, b, a_ex, b_ex, mask, major, minor, nnz, 1],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, mask, n_cells, stream=0))]
pub fn sparse_qc_csr_sub(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_cells;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let a = call.vector(sums_cells, "sums_cells", call.value, Some(major), true)?;
    let minor = mask.getattr("size")?.extract::<u64>()?;
    let b = 0;
    let a_ex = 0;
    let b_ex = 0;
    let mask = call.vector(mask, "mask", Dtype::Bool, Some(minor), false)?;
    call.run(
        "qc",
        major,
        32,
        false,
        stream,
        vec![p, i, d, a, b, a_ex, b_ex, mask, major, minor, nnz, 2],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, n_genes, stream=0))]
pub fn sparse_qc_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    cell_ex: &Bound<'_, PyAny>,
    gene_ex: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(data, Some(length), false, false, false)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), true)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), true)?;
    let ce = call.vector(cell_ex, "cell_ex", Dtype::I32, Some(n_cells), true)?;
    let ge = call.vector(gene_ex, "gene_ex", Dtype::I32, Some(n_genes), true)?;
    let mask = 0;
    call.run(
        "qc_dense",
        length,
        256,
        false,
        stream,
        vec![d, c, g, ce, ge, mask, n_cells, n_genes, 0],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, sums_cells, mask, n_cells, n_genes, stream=0))]
pub fn sparse_qc_dense_sub(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(data, Some(length), false, false, false)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), true)?;
    let g = 0;
    let ce = 0;
    let ge = 0;
    let mask = call.vector(mask, "mask", Dtype::Bool, Some(n_genes), false)?;
    call.run(
        "qc_dense",
        length,
        256,
        false,
        stream,
        vec![d, c, g, ce, ge, mask, n_cells, n_genes, 1],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, cell_ex, n_cells, stream=0))]
pub fn sparse_qc_csr_cells(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    cell_ex: &Bound<'_, PyAny>,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_cells;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    call.indices(index, nnz)?;
    let a = call.vector(sums_cells, "sums_cells", call.value, Some(major), true)?;
    let a_ex = call.vector(cell_ex, "cell_ex", Dtype::I32, Some(major), true)?;
    call.run(
        "qc_cells",
        major,
        32,
        false,
        stream,
        vec![p, d, a, a_ex, major, nnz],
    )
}

#[pyfunction]
#[pyo3(signature = (index, data, *, sums_genes, gene_ex, nnz, stream=0))]
pub fn sparse_qc_csr_genes(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    gene_ex: &Bound<'_, PyAny>,
    nnz: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, Some(nnz), true, false, false)?;
    let i = call.indices(index, nnz)?;
    let s = call.vector(sums_genes, "sums_genes", call.value, None, true)?;
    let minor = call.len();
    let e = call.vector(gene_ex, "gene_ex", Dtype::I32, Some(minor), true)?;
    call.run(
        "stats_minor",
        nnz,
        256,
        false,
        stream,
        vec![i, d, 0, 0, e, 0, s, nnz, minor, 2],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, sums_cells, cell_ex, n_cells, n_genes, stream=0))]
pub fn sparse_qc_dense_cells(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    cell_ex: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(data, Some(length), false, false, false)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), true)?;
    let g = 0;
    let ce = call.vector(cell_ex, "cell_ex", Dtype::I32, Some(n_cells), true)?;
    let ge = 0;
    let mask = 0;
    call.run(
        "qc_dense",
        length,
        256,
        false,
        stream,
        vec![d, c, g, ce, ge, mask, n_cells, n_genes, 2],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, sums_genes, gene_ex, n_cells, n_genes, stream=0))]
pub fn sparse_qc_dense_genes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    gene_ex: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(data, Some(length), false, false, false)?;
    let c = 0;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), true)?;
    let ce = 0;
    let ge = call.vector(gene_ex, "gene_ex", Dtype::I32, Some(n_genes), true)?;
    let mask = 0;
    call.run(
        "qc_dense",
        length,
        256,
        false,
        stream,
        vec![d, c, g, ce, ge, mask, n_cells, n_genes, 3],
    )
}

#[pyfunction]
#[pyo3(signature = (scaled_means, total_counts, expected, n_genes, n_cells, stream=0))]
pub fn expected_zeros(
    py: Python<'_>,
    scaled_means: &Bound<'_, PyAny>,
    total_counts: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
    n_genes: u64,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let m = call.data(scaled_means, Some(n_genes), true, false, false)?;
    let c = call.vector(
        total_counts,
        "total_counts",
        call.value,
        Some(n_cells),
        false,
    )?;
    let e = call.vector(expected, "expected", call.value, Some(n_genes), true)?;
    call.run(
        "expected",
        n_genes,
        256,
        false,
        stream,
        vec![m, c, e, n_genes, n_cells],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, sums_genes, residuals, inv_sum_total, clip, inv_theta, n_cells, n_genes, stream=0))]
pub fn sparse_norm_res_csc(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
    inv_sum_total: f64,
    clip: f64,
    inv_theta: f64,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, n_genes)?;
    let i = call.indices(index, nnz)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), false)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), false)?;
    let r = call.array(
        residuals,
        "residuals",
        Some(call.value),
        Some(size(n_cells, n_genes)?),
        false,
        Layout::C,
        true,
    )?;
    call.run(
        "residual_csc",
        n_genes,
        32,
        false,
        stream,
        vec![
            p,
            i,
            d,
            c,
            g,
            r,
            n_cells,
            n_genes,
            nnz,
            inv_sum_total.to_bits(),
            clip.to_bits(),
            inv_theta.to_bits(),
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_cells, sums_genes, residuals, inv_sum_total, clip, inv_theta, n_cells, n_genes, stream=0))]
pub fn sparse_norm_res_csr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
    inv_sum_total: f64,
    clip: f64,
    inv_theta: f64,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, n_cells)?;
    let i = call.indices(index, nnz)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), false)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), false)?;
    let r = call.array(
        residuals,
        "residuals",
        Some(call.value),
        Some(size(n_cells, n_genes)?),
        false,
        Layout::C,
        true,
    )?;
    call.run(
        "residual_csr",
        n_cells,
        8,
        false,
        stream,
        vec![
            p,
            i,
            d,
            c,
            g,
            r,
            n_cells,
            n_genes,
            nnz,
            inv_sum_total.to_bits(),
            clip.to_bits(),
            inv_theta.to_bits(),
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (X, *, residuals, sums_cells, sums_genes, inv_sum_total, clip, inv_theta, n_cells, n_genes, stream=0))]
#[allow(non_snake_case)]
pub fn dense_norm_res(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    inv_sum_total: f64,
    clip: f64,
    inv_theta: f64,
    n_cells: u64,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(X, Some(length), false, false, false)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), false)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), false)?;
    let r = call.array(
        residuals,
        "residuals",
        Some(call.value),
        Some(length),
        false,
        Layout::C,
        true,
    )?;
    call.run(
        "residual_dense",
        length,
        256,
        false,
        stream,
        vec![
            d,
            c,
            g,
            r,
            n_cells,
            n_genes,
            inv_sum_total.to_bits(),
            clip.to_bits(),
            inv_theta.to_bits(),
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_genes, sums_cells, n_genes, stream=0))]
pub fn sparse_sum_csc(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    n_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let major = n_genes;
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, major)?;
    let i = call.indices(index, nnz)?;
    let a = call.vector(sums_genes, "sums_genes", call.value, Some(major), true)?;
    let b = call.vector(sums_cells, "sums_cells", call.value, None, true)?;
    let minor = call.len();
    let a_ex = 0;
    let b_ex = 0;
    let mask = 0;
    call.run(
        "qc",
        major,
        32,
        false,
        stream,
        vec![p, i, d, a, b, a_ex, b_ex, mask, major, minor, nnz, 4],
    )
}

#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, sums_genes, sums_cells, residuals, inv_sum_total, clip, inv_theta, n_genes, n_cells, stream=0))]
pub fn csc_hvg_res(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
    inv_sum_total: f64,
    clip: f64,
    inv_theta: f64,
    n_genes: u64,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let nnz = call.len();
    let p = call.indptr(indptr, n_genes)?;
    let i = call.indices(index, nnz)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), false)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), false)?;
    let r = call.array(
        residuals,
        "residuals",
        Some(call.value),
        None,
        false,
        Layout::C,
        true,
    )?;
    if call.len() < n_genes {
        return Err(PyValueError::new_err(
            "residuals buffer is shorter than n_genes",
        ));
    }
    call.run(
        "residual_hvg",
        n_genes,
        32,
        false,
        stream,
        vec![
            p,
            i,
            d,
            c,
            g,
            r,
            n_cells,
            n_genes,
            nnz,
            inv_sum_total.to_bits(),
            clip.to_bits(),
            inv_theta.to_bits(),
        ],
    )
}

#[pyfunction]
#[pyo3(signature = (data, *, sums_genes, sums_cells, residuals, inv_sum_total, clip, inv_theta, n_genes, n_cells, stream=0))]
pub fn dense_hvg_res(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sums_genes: &Bound<'_, PyAny>,
    sums_cells: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
    inv_sum_total: f64,
    clip: f64,
    inv_theta: f64,
    n_genes: u64,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let length = size(n_cells, n_genes)?;
    let d = call.data(data, Some(length), false, true, false)?;
    let c = call.vector(sums_cells, "sums_cells", call.value, Some(n_cells), false)?;
    let g = call.vector(sums_genes, "sums_genes", call.value, Some(n_genes), false)?;
    let r = call.array(
        residuals,
        "residuals",
        Some(call.value),
        None,
        false,
        Layout::C,
        true,
    )?;
    if call.len() < n_genes {
        return Err(PyValueError::new_err(
            "residuals buffer is shorter than n_genes",
        ));
    }
    call.run(
        "residual_dense_hvg",
        n_genes,
        32,
        false,
        stream,
        vec![
            d,
            c,
            g,
            r,
            n_cells,
            n_genes,
            inv_sum_total.to_bits(),
            clip.to_bits(),
            inv_theta.to_bits(),
        ],
    )
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._mean_var_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(mean_var_major, &module)?)?;
    module.add_function(wrap_pyfunction!(mean_var_minor, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._nanmean_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(nan_mean_minor, &module)?)?;
    module.add_function(wrap_pyfunction!(nan_mean_major, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._norm_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(mul_dense, &module)?)?;
    module.add_function(wrap_pyfunction!(mul_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(sum_major, &module)?)?;
    module.add_function(wrap_pyfunction!(find_hi_genes_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(masked_mul_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(masked_sum_major, &module)?)?;
    module.add_function(wrap_pyfunction!(prescaled_mul_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(prescaled_mul_dense, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._scale_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(csc_scale_diff, &module)?)?;
    module.add_function(wrap_pyfunction!(csr_scale_diff, &module)?)?;
    module.add_function(wrap_pyfunction!(dense_scale_center_diff, &module)?)?;
    module.add_function(wrap_pyfunction!(dense_scale_diff, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._qc_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(sparse_qc_csc, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_csc_sub, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_csr_sub, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_dense, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_dense_sub, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._qc_dask_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(sparse_qc_csr_cells, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_csr_genes, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_dense_cells, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_qc_dense_genes, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._hvg_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(expected_zeros, &module)?)?;
    parent.add_submodule(&module)?;
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._pr_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(sparse_norm_res_csc, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_norm_res_csr, &module)?)?;
    module.add_function(wrap_pyfunction!(dense_norm_res, &module)?)?;
    module.add_function(wrap_pyfunction!(sparse_sum_csc, &module)?)?;
    module.add_function(wrap_pyfunction!(csc_hvg_res, &module)?)?;
    module.add_function(wrap_pyfunction!(dense_hvg_res, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
