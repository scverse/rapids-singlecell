//! kde bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(xy, *, out, n, a, b, c, stream=0))]
fn gaussian_kde_2d(
    py: Python<'_>,
    xy: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n: u64,
    a: f64,
    b: f64,
    c: f64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let xy = read(Some(xy), &cupy, "xy", "T", false)?;
    let out = read(Some(out), &cupy, "out", "T", false)?;
    same(&out, &xy)?;
    launch(
        &cupy,
        &[&xy, &out],
        "domain_kde_gaussian_kde_2d",
        n,
        stream,
        vec![
            xy.pointer(),
            out.pointer(),
            n,
            a.to_bits(),
            b.to_bits(),
            c.to_bits(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_kde_cuda")?;
    m.add_function(wrap_pyfunction!(gaussian_kde_2d, &m)?)?;
    Ok(())
}
