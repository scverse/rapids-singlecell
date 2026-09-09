//! pv bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(x, *, out, n_rows, m, stream=0))]
fn rev_cummin64(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n_rows: u64,
    m: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let x = read(Some(x), &cupy, "x", "double", false)?;
    let out = read(Some(out), &cupy, "out", "double", false)?;
    launch(
        &cupy,
        &[&x, &out],
        "domain_pv_rev_cummin64",
        n_rows,
        stream,
        vec![x.pointer(), out.pointer(), n_rows, m],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_pv_cuda")?;
    m.add_function(wrap_pyfunction!(rev_cummin64, &m)?)?;
    Ok(())
}
