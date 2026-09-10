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
    if !(32..=1024).contains(&block_size) || !block_size.is_multiple_of(32) {
        return Err(PyValueError::new_err(
            "block_size must be a multiple of 32 from 32 through 1024",
        ));
    }
    let cell_tile = if [16, 32, 64, 128, 256, 512, 1024].contains(&cell_tile) {
        cell_tile
    } else {
        16 // Preserve the original launcher's default tile specialization.
    };
    let padded = l_val
        .checked_add(31)
        .ok_or_else(|| PyValueError::new_err("too many thresholds"))?
        / 32
        * 32;
    let warps = block_size / 32;
    // Bound every launch by the supported 48 KiB floor. Large threshold sets
    // use independent cumulative windows instead of an unbounded histogram.
    let window = padded.min((48 * 1024 - cell_tile * 8) / (warps * 8) / 32 * 32);
    let _ = shared_mem; // A caller's tuning hint cannot enlarge the safe budget.
    let shared_mem = cell_tile * 8 + warps * window * 8;
    let work = num_pairs
        .checked_mul(blocks_per_pair)
        .and_then(|n| n.checked_mul(block_size))
        .ok_or_else(|| PyValueError::new_err("cooccurrence launch extent overflow"))?;
    if thresholds.len() < l_val
        || pair_left.len() < num_pairs
        || pair_right.len() < num_pairs
        || cat_offsets.len() <= k
        || k.checked_mul(k)
            .and_then(|n| n.checked_mul(l_val))
            .is_none_or(|n| counts.len() < n)
    {
        return Err(PyValueError::new_err(
            "cooccurrence dimensions exceed array extents",
        ));
    }
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
        work,
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
fn get_kernel_config(
    py: Python<'_>,
    l_val: u64,
    n_cells: u64,
    k: u64,
) -> PyResult<Option<(u64, u64, u64, u64)>> {
    if k == 0 {
        return Err(PyValueError::new_err("k must be positive"));
    }
    let cp = py.import("cupy")?;
    let runtime = cp.getattr("cuda")?.getattr("runtime")?;
    let device = runtime.call_method0("getDevice")?;
    let properties = runtime.call_method1("getDeviceProperties", (device,))?;
    let maximum = properties
        .get_item("sharedMemPerBlock")?
        .extract::<u64>()?
        .min(48 * 1024);
    let padded = l_val
        .checked_add(31)
        .ok_or_else(|| PyValueError::new_err("too many thresholds"))?
        / 32
        * 32;
    let target = match n_cells / k {
        10000.. => 1024,
        5000.. => 512,
        2500.. => 256,
        _ => 128,
    };
    for block in [target, 128, 256, 512, 1024] {
        for tile in [1024, 512, 256, 128, 64, 32, 16] {
            if let Some(bytes) = padded
                .checked_mul(block / 32 * 8)
                .and_then(|n| n.checked_add(tile * 8))
                && bytes <= maximum
            {
                return Ok(Some((tile, padded, block, bytes)));
            }
        }
    }
    // The bounded window path also supports threshold sets that exhausted the
    // original launcher's shared-memory configurations.
    let window = padded.min((maximum - 128 * 8) / (4 * 8) / 32 * 32);
    Ok(Some((128, padded, 128, 128 * 8 + 4 * window * 8)))
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
