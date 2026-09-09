//! edistance bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(indptr, indices, data, cat_offsets, cell_indices, pair_left, pair_right, pairwise_sums, num_pairs, n_features, blocks_per_pair, cell_tile, feat_tile, block_size, shared_mem, stream=0))]
fn compute_distances_sparse(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    cat_offsets: &Bound<'_, PyAny>,
    cell_indices: &Bound<'_, PyAny>,
    pair_left: &Bound<'_, PyAny>,
    pair_right: &Bound<'_, PyAny>,
    pairwise_sums: &Bound<'_, PyAny>,
    num_pairs: u64,
    n_features: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    feat_tile: u64,
    block_size: u64,
    shared_mem: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;
    let _stream_scope = runtime::StreamScope::new(&cupy, stream)?;
    let workspace_object = cupy.call_method1("empty", (num_pairs * blocks_per_pair, "float64"))?;
    let workspace = &workspace_object;
    let indptr = read(Some(indptr), &cupy, "indptr", "IndptrT", false)?;
    let indices = read(Some(indices), &cupy, "indices", "int", false)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let cat_offsets = read(Some(cat_offsets), &cupy, "cat_offsets", "int", false)?;
    let cell_indices = read(Some(cell_indices), &cupy, "cell_indices", "int", false)?;
    let pair_left = read(Some(pair_left), &cupy, "pair_left", "int", false)?;
    let pair_right = read(Some(pair_right), &cupy, "pair_right", "int", false)?;
    let pairwise_sums = read(Some(pairwise_sums), &cupy, "pairwise_sums", "T", false)?;
    same(&pairwise_sums, &data)?;
    let workspace = read(Some(workspace), &cupy, "workspace", "double", false)?;
    for phase in 0..2_u64 {
        launch(
            &cupy,
            &[
                &indptr,
                &indices,
                &data,
                &cat_offsets,
                &cell_indices,
                &pair_left,
                &pair_right,
                &pairwise_sums,
                &workspace,
            ],
            "domain_edistance_compute_distances_sparse",
            if phase == 0 {
                num_pairs * blocks_per_pair * 128
            } else {
                num_pairs * 32
            },
            stream,
            vec![
                indptr.pointer(),
                indices.pointer(),
                data.pointer(),
                cat_offsets.pointer(),
                cell_indices.pointer(),
                pair_left.pointer(),
                pair_right.pointer(),
                pairwise_sums.pointer(),
                num_pairs,
                n_features,
                blocks_per_pair,
                cell_tile,
                feat_tile,
                block_size,
                shared_mem,
                workspace.pointer(),
                phase,
            ],
        )?;
    }
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(embedding, cat_offsets, cell_indices, pair_left, pair_right, pairwise_sums, num_pairs, n_features, blocks_per_pair, cell_tile, feat_tile, block_size, shared_mem, stream=0))]
fn compute_distances(
    py: Python<'_>,
    embedding: &Bound<'_, PyAny>,
    cat_offsets: &Bound<'_, PyAny>,
    cell_indices: &Bound<'_, PyAny>,
    pair_left: &Bound<'_, PyAny>,
    pair_right: &Bound<'_, PyAny>,
    pairwise_sums: &Bound<'_, PyAny>,
    num_pairs: u64,
    n_features: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    feat_tile: u64,
    block_size: u64,
    shared_mem: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;
    let _stream_scope = runtime::StreamScope::new(&cupy, stream)?;
    let workspace_object = cupy.call_method1("empty", (num_pairs * blocks_per_pair, "float64"))?;
    let workspace = &workspace_object;
    let embedding = read(Some(embedding), &cupy, "embedding", "T", false)?;
    let cat_offsets = read(Some(cat_offsets), &cupy, "cat_offsets", "int", false)?;
    let cell_indices = read(Some(cell_indices), &cupy, "cell_indices", "int", false)?;
    let pair_left = read(Some(pair_left), &cupy, "pair_left", "int", false)?;
    let pair_right = read(Some(pair_right), &cupy, "pair_right", "int", false)?;
    let pairwise_sums = read(Some(pairwise_sums), &cupy, "pairwise_sums", "T", false)?;
    same(&pairwise_sums, &embedding)?;
    let workspace = read(Some(workspace), &cupy, "workspace", "double", false)?;
    for phase in 0..2_u64 {
        launch(
            &cupy,
            &[
                &embedding,
                &cat_offsets,
                &cell_indices,
                &pair_left,
                &pair_right,
                &pairwise_sums,
                &workspace,
            ],
            "domain_edistance_compute_distances",
            if phase == 0 {
                num_pairs * blocks_per_pair * 128
            } else {
                num_pairs * 32
            },
            stream,
            vec![
                embedding.pointer(),
                cat_offsets.pointer(),
                cell_indices.pointer(),
                pair_left.pointer(),
                pair_right.pointer(),
                pairwise_sums.pointer(),
                num_pairs,
                n_features,
                blocks_per_pair,
                cell_tile,
                feat_tile,
                block_size,
                shared_mem,
                workspace.pointer(),
                phase,
            ],
        )?;
    }
    Ok(())
}
#[pyfunction]
fn get_kernel_config(n_features: u64, is_double: bool) -> Option<(u64, u64, u64, u64)> {
    let _ = (n_features, is_double);
    Some((32, 32, 128, 0))
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_edistance_cuda")?;
    m.add_function(wrap_pyfunction!(compute_distances_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_distances, &m)?)?;
    m.add_function(wrap_pyfunction!(get_kernel_config, &m)?)?;
    Ok(())
}
