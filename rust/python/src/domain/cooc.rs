//! cooc bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(spatial, *, thresholds, cat_offsets, cell_indices, pair_left, pair_right, counts, num_pairs, k, l_val, blocks_per_pair, cell_tile, block_size, shared_mem, stream=0))]
fn count_csr_catpairs(
    py: Python<'_>,
    spatial: &Bound<'_, PyAny>,
    thresholds: &Bound<'_, PyAny>,
    cat_offsets: &Bound<'_, PyAny>,
    cell_indices: &Bound<'_, PyAny>,
    pair_left: &Bound<'_, PyAny>,
    pair_right: &Bound<'_, PyAny>,
    counts: &Bound<'_, PyAny>,
    num_pairs: u64,
    k: u64,
    l_val: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    block_size: u64,
    shared_mem: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let spatial = read(Some(spatial), &cupy, "spatial", "float", false)?;
    let thresholds = read(Some(thresholds), &cupy, "thresholds", "float", false)?;
    let cat_offsets = read(Some(cat_offsets), &cupy, "cat_offsets", "int", false)?;
    let cell_indices = read(Some(cell_indices), &cupy, "cell_indices", "int", false)?;
    let pair_left = read(Some(pair_left), &cupy, "pair_left", "int", false)?;
    let pair_right = read(Some(pair_right), &cupy, "pair_right", "int", false)?;
    let counts = read(Some(counts), &cupy, "counts", "cooc_count_t", false)?;
    launch(
        &cupy,
        &[
            &spatial,
            &thresholds,
            &cat_offsets,
            &cell_indices,
            &pair_left,
            &pair_right,
            &counts,
        ],
        "domain_cooc_count_csr_catpairs",
        num_pairs * blocks_per_pair * 128,
        stream,
        vec![
            spatial.pointer(),
            thresholds.pointer(),
            cat_offsets.pointer(),
            cell_indices.pointer(),
            pair_left.pointer(),
            pair_right.pointer(),
            counts.pointer(),
            num_pairs,
            k,
            l_val,
            blocks_per_pair,
            cell_tile,
            block_size,
            shared_mem,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(spatial, *, thresholds, labels, result, n, k, l_val, stream=0))]
fn count_pairwise(
    py: Python<'_>,
    spatial: &Bound<'_, PyAny>,
    thresholds: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    result: &Bound<'_, PyAny>,
    n: u64,
    k: u64,
    l_val: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let spatial = read(Some(spatial), &cupy, "spatial", "float", false)?;
    let thresholds = read(Some(thresholds), &cupy, "thresholds", "float", false)?;
    let labels = read(Some(labels), &cupy, "labels", "int", false)?;
    let result = read(Some(result), &cupy, "result", "cooc_count_t", false)?;
    launch(
        &cupy,
        &[&spatial, &thresholds, &labels, &result],
        "domain_cooc_count_pairwise",
        n * 128,
        stream,
        vec![
            spatial.pointer(),
            thresholds.pointer(),
            labels.pointer(),
            result.pointer(),
            n,
            k,
            l_val,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(result, *, out, k, l_val, format, stream=0))]
fn reduce_shared(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    k: u64,
    l_val: u64,
    format: u64,
    stream: usize,
) -> PyResult<bool> {
    let cupy = py.import("cupy")?;
    let _stream_scope = runtime::StreamScope::new(&cupy, stream)?;
    let workspace_object = cupy.call_method1("empty", ((k + 1) * l_val, "float64"))?;
    let workspace = &workspace_object;
    let result = read(Some(result), &cupy, "result", "cooc_count_t", false)?;
    let out = read(Some(out), &cupy, "out", "float", false)?;
    let workspace = read(Some(workspace), &cupy, "workspace", "double", false)?;
    launch(
        &cupy,
        &[&result, &out, &workspace],
        "domain_cooc_reduce_shared",
        l_val * 128,
        stream,
        vec![
            result.pointer(),
            out.pointer(),
            k,
            l_val,
            format,
            workspace.pointer(),
        ],
    )?;
    Ok(true)
}
#[pyfunction]
#[pyo3(signature=(result, *, inter_out, out, k, l_val, format, stream=0))]
fn reduce_global(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    inter_out: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    k: u64,
    l_val: u64,
    format: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;
    let _stream_scope = runtime::StreamScope::new(&cupy, stream)?;
    let workspace_object = cupy.call_method1("empty", ((k + 1) * l_val, "float64"))?;
    let workspace = &workspace_object;
    let result = read(Some(result), &cupy, "result", "cooc_count_t", false)?;
    let inter_out = read(Some(inter_out), &cupy, "inter_out", "float", false)?;
    let out = read(Some(out), &cupy, "out", "float", false)?;
    let workspace = read(Some(workspace), &cupy, "workspace", "double", false)?;
    launch(
        &cupy,
        &[&result, &inter_out, &out, &workspace],
        "domain_cooc_reduce_global",
        l_val * 128,
        stream,
        vec![
            result.pointer(),
            inter_out.pointer(),
            out.pointer(),
            k,
            l_val,
            format,
            workspace.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
fn get_kernel_config(l_val: u64, n_cells: u64, k: u64) -> PyResult<Option<(u64, u64, u64, u64)>> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be positive"));
    }
    let _ = n_cells;
    Ok(Some((128, l_val.div_ceil(32) * 32, 128, 0)))
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_cooc_cuda")?;
    m.add_function(wrap_pyfunction!(count_csr_catpairs, &m)?)?;
    m.add_function(wrap_pyfunction!(count_pairwise, &m)?)?;
    m.add_function(wrap_pyfunction!(reduce_shared, &m)?)?;
    m.add_function(wrap_pyfunction!(reduce_global, &m)?)?;
    m.add_function(wrap_pyfunction!(get_kernel_config, &m)?)?;
    Ok(())
}
