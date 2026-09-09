use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::ffi::c_void;

/// Write Jaccard weights on the borrowed stream. Keep arrays and stream alive
/// until the asynchronous work completes, and order cross-stream accesses.
#[pyfunction]
#[pyo3(signature = (knn, *, n_obs, k, jaccard_vals, stream=0))]
pub fn jaccard_shared_counts(
    py: Python<'_>,
    knn: &Bound<'_, PyAny>,
    n_obs: i64,
    k: i64,
    jaccard_vals: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    if !(0..=i32::MAX as i64).contains(&n_obs) || !(0..=i32::MAX as i64).contains(&k) {
        return Err(PyValueError::new_err(
            "n_obs and k must be nonnegative int32 dimensions",
        ));
    }
    let mut n_obs = n_obs as u64;
    let mut k = k as u64;
    let n_edges = n_obs * k;
    if n_edges > isize::MAX as u64 / 4 {
        return Err(PyValueError::new_err(
            "KNN array size exceeds addressable memory",
        ));
    }
    let cupy = py.import("cupy")?;
    let input = Array::read(
        knn,
        &cupy,
        "knn",
        Some(Dtype::I32),
        Some(&[n_obs as usize, k as usize]),
        Layout::C,
    )?;
    let output = Array::read(
        jaccard_vals,
        &cupy,
        "jaccard_vals",
        Some(Dtype::F32),
        Some(&[n_edges as usize]),
        Layout::C,
    )?;
    let device = current_device(&cupy, &[&input, &output])?;
    output.require_disjoint(&input)?;
    if n_edges == 0 {
        return Ok(());
    }
    let mut input_pointer = input.pointer;
    let mut output_pointer = output.pointer;
    let mut arguments = [
        (&mut input_pointer as *mut u64).cast::<c_void>(),
        (&mut n_obs as *mut u64).cast::<c_void>(),
        (&mut k as *mut u64).cast::<c_void>(),
        (&mut output_pointer as *mut u64).cast::<c_void>(),
    ];
    // SAFETY: validated disjoint arrays match the kernel's four u64 slots. The
    // one-dimensional grid is capped; device code grid-strides over all edges.
    unsafe {
        runtime::launch(
            device,
            "jaccard_shared_counts",
            (n_edges.div_ceil(256).min(65_535) as u32, 1, 1),
            (256, 1, 1),
            stream,
            &mut arguments,
        )
    }
}
