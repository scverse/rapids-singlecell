//! guide_assignment bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(X, valid_guides, lam, mu, sigma, pi0, assignments, thresholds, *, n_cells, n_guides, n_valid_guides, posterior_threshold, stream=0))]
fn assign_threshold_dense(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    valid_guides: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    mu: &Bound<'_, PyAny>,
    sigma: &Bound<'_, PyAny>,
    pi0: &Bound<'_, PyAny>,
    assignments: &Bound<'_, PyAny>,
    thresholds: &Bound<'_, PyAny>,
    n_cells: u64,
    n_guides: u64,
    n_valid_guides: u64,
    posterior_threshold: f64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "float", true)?;
    let valid_guides = read(Some(valid_guides), &cupy, "valid_guides", "int", false)?;
    let lam = read(Some(lam), &cupy, "lam", "float", false)?;
    let mu = read(Some(mu), &cupy, "mu", "float", false)?;
    let sigma = read(Some(sigma), &cupy, "sigma", "float", false)?;
    let pi0 = read(Some(pi0), &cupy, "pi0", "float", false)?;
    let assignments = read(Some(assignments), &cupy, "assignments", "bool", false)?;
    let thresholds = read(Some(thresholds), &cupy, "thresholds", "float", false)?;
    launch(
        &cupy,
        &[
            &X,
            &valid_guides,
            &lam,
            &mu,
            &sigma,
            &pi0,
            &assignments,
            &thresholds,
        ],
        "domain_guide_assignment_assign_threshold_dense",
        n_valid_guides * 256,
        stream,
        vec![
            X.pointer(),
            valid_guides.pointer(),
            lam.pointer(),
            mu.pointer(),
            sigma.pointer(),
            pi0.pointer(),
            assignments.pointer(),
            thresholds.pointer(),
            n_cells,
            n_guides,
            n_valid_guides,
            posterior_threshold.to_bits(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X, assignments, thresholds, lam, mu, sigma, pi0, valid_mask, nonzero_counts, max_counts, *, n_cells, n_guides, max_iter, tol, posterior_threshold, stream=0))]
fn fit_assign_dense(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    assignments: &Bound<'_, PyAny>,
    thresholds: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    mu: &Bound<'_, PyAny>,
    sigma: &Bound<'_, PyAny>,
    pi0: &Bound<'_, PyAny>,
    valid_mask: &Bound<'_, PyAny>,
    nonzero_counts: &Bound<'_, PyAny>,
    max_counts: &Bound<'_, PyAny>,
    n_cells: u64,
    n_guides: u64,
    max_iter: u64,
    tol: f64,
    posterior_threshold: f64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "float", true)?;
    let assignments = read(Some(assignments), &cupy, "assignments", "bool", false)?;
    let thresholds = read(Some(thresholds), &cupy, "thresholds", "float", false)?;
    let lam = read(Some(lam), &cupy, "lam", "float", false)?;
    let mu = read(Some(mu), &cupy, "mu", "float", false)?;
    let sigma = read(Some(sigma), &cupy, "sigma", "float", false)?;
    let pi0 = read(Some(pi0), &cupy, "pi0", "float", false)?;
    let valid_mask = read(Some(valid_mask), &cupy, "valid_mask", "bool", false)?;
    let nonzero_counts = read(Some(nonzero_counts), &cupy, "nonzero_counts", "int", false)?;
    let max_counts = read(Some(max_counts), &cupy, "max_counts", "int", false)?;
    launch(
        &cupy,
        &[
            &X,
            &assignments,
            &thresholds,
            &lam,
            &mu,
            &sigma,
            &pi0,
            &valid_mask,
            &nonzero_counts,
            &max_counts,
        ],
        "domain_guide_assignment_fit_assign_dense",
        n_guides * 256,
        stream,
        vec![
            X.pointer(),
            assignments.pointer(),
            thresholds.pointer(),
            lam.pointer(),
            mu.pointer(),
            sigma.pointer(),
            pi0.pointer(),
            valid_mask.pointer(),
            nonzero_counts.pointer(),
            max_counts.pointer(),
            n_cells,
            n_guides,
            max_iter,
            tol.to_bits(),
            posterior_threshold.to_bits(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_guide_assignment_cuda")?;
    m.add_function(wrap_pyfunction!(assign_threshold_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_assign_dense, &m)?)?;
    Ok(())
}
