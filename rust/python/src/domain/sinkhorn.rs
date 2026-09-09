//! sinkhorn bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(cost, cost_off, n, m, scale, floor, eps, stream=0))]
fn auto_eps(
    py: Python<'_>,
    cost: &Bound<'_, PyAny>,
    cost_off: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    scale: f64,
    floor: f64,
    eps: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let cost = read(Some(cost), &cupy, "cost", "T", false)?;
    let cost_off = read(Some(cost_off), &cupy, "cost_off", "int64_t", false)?;
    let n = read(Some(n), &cupy, "n", "int", false)?;
    let m = read(Some(m), &cupy, "m", "int", false)?;
    let eps = read(Some(eps), &cupy, "eps", "T", false)?;
    same(&eps, &cost)?;
    launch(
        &cupy,
        &[&cost, &cost_off, &n, &m, &eps],
        "domain_sinkhorn_auto_eps",
        n.len() * 32,
        stream,
        vec![
            cost.pointer(),
            cost_off.pointer(),
            n.pointer(),
            m.pointer(),
            scale.to_bits(),
            floor.to_bits(),
            eps.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(cost, cost_off, n, m, f, f_off, g, g_off, col2pair, eps, log_b, conv, omega=1.0, stream=0))]
fn update_g(
    py: Python<'_>,
    cost: &Bound<'_, PyAny>,
    cost_off: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
    f_off: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    g_off: &Bound<'_, PyAny>,
    col2pair: &Bound<'_, PyAny>,
    eps: &Bound<'_, PyAny>,
    log_b: &Bound<'_, PyAny>,
    conv: &Bound<'_, PyAny>,
    omega: f64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let cost = read(Some(cost), &cupy, "cost", "T", false)?;
    let cost_off = read(Some(cost_off), &cupy, "cost_off", "int64_t", false)?;
    let n = read(Some(n), &cupy, "n", "int", false)?;
    let m = read(Some(m), &cupy, "m", "int", false)?;
    let f = read(Some(f), &cupy, "f", "T", false)?;
    same(&f, &cost)?;
    let f_off = read(Some(f_off), &cupy, "f_off", "int64_t", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &cost)?;
    let g_off = read(Some(g_off), &cupy, "g_off", "int64_t", false)?;
    let col2pair = read(Some(col2pair), &cupy, "col2pair", "int", false)?;
    let eps = read(Some(eps), &cupy, "eps", "T", false)?;
    same(&eps, &cost)?;
    let log_b = read(Some(log_b), &cupy, "log_b", "T", false)?;
    same(&log_b, &cost)?;
    let conv = read(Some(conv), &cupy, "conv", "int", false)?;
    launch(
        &cupy,
        &[
            &cost, &cost_off, &n, &m, &f, &f_off, &g, &g_off, &col2pair, &eps, &log_b, &conv,
        ],
        "domain_sinkhorn_update_g",
        col2pair.len(),
        stream,
        vec![
            cost.pointer(),
            cost_off.pointer(),
            n.pointer(),
            m.pointer(),
            f.pointer(),
            f_off.pointer(),
            g.pointer(),
            g_off.pointer(),
            col2pair.pointer(),
            eps.pointer(),
            log_b.pointer(),
            conv.pointer(),
            omega.to_bits(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(cost, cost_off, m, g, g_off, f, f_off, row2pair, eps, log_a, conv, omega=1.0, stream=0))]
fn update_f(
    py: Python<'_>,
    cost: &Bound<'_, PyAny>,
    cost_off: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    g_off: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
    f_off: &Bound<'_, PyAny>,
    row2pair: &Bound<'_, PyAny>,
    eps: &Bound<'_, PyAny>,
    log_a: &Bound<'_, PyAny>,
    conv: &Bound<'_, PyAny>,
    omega: f64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let cost = read(Some(cost), &cupy, "cost", "T", false)?;
    let cost_off = read(Some(cost_off), &cupy, "cost_off", "int64_t", false)?;
    let m = read(Some(m), &cupy, "m", "int", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &cost)?;
    let g_off = read(Some(g_off), &cupy, "g_off", "int64_t", false)?;
    let f = read(Some(f), &cupy, "f", "T", false)?;
    same(&f, &cost)?;
    let f_off = read(Some(f_off), &cupy, "f_off", "int64_t", false)?;
    let row2pair = read(Some(row2pair), &cupy, "row2pair", "int", false)?;
    let eps = read(Some(eps), &cupy, "eps", "T", false)?;
    same(&eps, &cost)?;
    let log_a = read(Some(log_a), &cupy, "log_a", "T", false)?;
    same(&log_a, &cost)?;
    let conv = read(Some(conv), &cupy, "conv", "int", false)?;
    launch(
        &cupy,
        &[
            &cost, &cost_off, &m, &g, &g_off, &f, &f_off, &row2pair, &eps, &log_a, &conv,
        ],
        "domain_sinkhorn_update_f",
        row2pair.len() * 32,
        stream,
        vec![
            cost.pointer(),
            cost_off.pointer(),
            m.pointer(),
            g.pointer(),
            g_off.pointer(),
            f.pointer(),
            f_off.pointer(),
            row2pair.pointer(),
            eps.pointer(),
            log_a.pointer(),
            conv.pointer(),
            omega.to_bits(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(f, f_prev, f_off, n, g, g_prev, g_off, m, tol, conv, stream=0))]
fn check_convergence(
    py: Python<'_>,
    f: &Bound<'_, PyAny>,
    f_prev: &Bound<'_, PyAny>,
    f_off: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    g_prev: &Bound<'_, PyAny>,
    g_off: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    tol: f64,
    conv: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let f = read(Some(f), &cupy, "f", "T", false)?;
    let f_prev = read(Some(f_prev), &cupy, "f_prev", "T", false)?;
    same(&f_prev, &f)?;
    let f_off = read(Some(f_off), &cupy, "f_off", "int64_t", false)?;
    let n = read(Some(n), &cupy, "n", "int", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &f)?;
    let g_prev = read(Some(g_prev), &cupy, "g_prev", "T", false)?;
    same(&g_prev, &f)?;
    let g_off = read(Some(g_off), &cupy, "g_off", "int64_t", false)?;
    let m = read(Some(m), &cupy, "m", "int", false)?;
    let conv = read(Some(conv), &cupy, "conv", "int", false)?;
    launch(
        &cupy,
        &[&f, &f_prev, &f_off, &n, &g, &g_prev, &g_off, &m, &conv],
        "domain_sinkhorn_check_convergence",
        n.len() * 32,
        stream,
        vec![
            f.pointer(),
            f_prev.pointer(),
            f_off.pointer(),
            n.pointer(),
            g.pointer(),
            g_prev.pointer(),
            g_off.pointer(),
            m.pointer(),
            tol.to_bits(),
            conv.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(emb, cidx_l, f_off, cidx_r, g_off, n, m, cost_off, tile_pair, tile_i0, tile_j0, cost, stream=0))]
fn build_cost(
    py: Python<'_>,
    emb: &Bound<'_, PyAny>,
    cidx_l: &Bound<'_, PyAny>,
    f_off: &Bound<'_, PyAny>,
    cidx_r: &Bound<'_, PyAny>,
    g_off: &Bound<'_, PyAny>,
    n: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    cost_off: &Bound<'_, PyAny>,
    tile_pair: &Bound<'_, PyAny>,
    tile_i0: &Bound<'_, PyAny>,
    tile_j0: &Bound<'_, PyAny>,
    cost: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let emb = read(Some(emb), &cupy, "emb", "T", false)?;
    let cidx_l = read(Some(cidx_l), &cupy, "cidx_l", "int", false)?;
    let f_off = read(Some(f_off), &cupy, "f_off", "int64_t", false)?;
    let cidx_r = read(Some(cidx_r), &cupy, "cidx_r", "int", false)?;
    let g_off = read(Some(g_off), &cupy, "g_off", "int64_t", false)?;
    let n = read(Some(n), &cupy, "n", "int", false)?;
    let m = read(Some(m), &cupy, "m", "int", false)?;
    let cost_off = read(Some(cost_off), &cupy, "cost_off", "int64_t", false)?;
    let tile_pair = read(Some(tile_pair), &cupy, "tile_pair", "int", false)?;
    let tile_i0 = read(Some(tile_i0), &cupy, "tile_i0", "int", false)?;
    let tile_j0 = read(Some(tile_j0), &cupy, "tile_j0", "int", false)?;
    let cost = read(Some(cost), &cupy, "cost", "T", false)?;
    same(&cost, &emb)?;
    launch(
        &cupy,
        &[
            &emb, &cidx_l, &f_off, &cidx_r, &g_off, &n, &m, &cost_off, &tile_pair, &tile_i0,
            &tile_j0, &cost,
        ],
        "domain_sinkhorn_build_cost",
        tile_pair.len() * 256,
        stream,
        vec![
            emb.pointer(),
            cidx_l.pointer(),
            f_off.pointer(),
            cidx_r.pointer(),
            g_off.pointer(),
            n.pointer(),
            m.pointer(),
            cost_off.pointer(),
            tile_pair.pointer(),
            tile_i0.pointer(),
            tile_j0.pointer(),
            cost.pointer(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_sinkhorn_cuda")?;
    m.add_function(wrap_pyfunction!(auto_eps, &m)?)?;
    m.add_function(wrap_pyfunction!(update_g, &m)?)?;
    m.add_function(wrap_pyfunction!(update_f, &m)?)?;
    m.add_function(wrap_pyfunction!(check_convergence, &m)?)?;
    m.add_function(wrap_pyfunction!(build_cost, &m)?)?;
    Ok(())
}
