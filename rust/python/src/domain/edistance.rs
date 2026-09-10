//! edistance bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
fn kernel_name(
    sparse: bool,
    dtype: u64,
    cell_tile: u64,
    feat_tile: u64,
    block_size: u64,
) -> String {
    let (kind, cells, features) = if dtype == 0 {
        if cell_tile == 64 {
            ("f32", 64, if feat_tile == 25 { 25 } else { 16 })
        } else {
            (
                "f32",
                32,
                if matches!(feat_tile, 50 | 64) {
                    feat_tile
                } else {
                    32
                },
            )
        }
    } else {
        (
            "f64",
            16,
            if matches!(feat_tile, 50 | 64) {
                feat_tile
            } else {
                32
            },
        )
    };
    let layout = if sparse { "sparse" } else { "dense" };
    let block_suffix = if block_size > 512 { "_b1024" } else { "" };
    format!("domain_edistance_{layout}_{kind}_c{cells}_f{features}{block_suffix}")
}

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
    let indptr = read(Some(indptr), &cupy, "indptr", "IndptrT", false)?;
    let indices = read(Some(indices), &cupy, "indices", "int", false)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let cat_offsets = read(Some(cat_offsets), &cupy, "cat_offsets", "int", false)?;
    let cell_indices = read(Some(cell_indices), &cupy, "cell_indices", "int", false)?;
    let pair_left = read(Some(pair_left), &cupy, "pair_left", "int", false)?;
    let pair_right = read(Some(pair_right), &cupy, "pair_right", "int", false)?;
    let pairwise_sums = read(Some(pairwise_sums), &cupy, "pairwise_sums", "T", false)?;
    same(&pairwise_sums, &data)?;
    if block_size == 0 || block_size > 1024 || !block_size.is_multiple_of(32) {
        return Err(PyValueError::new_err(
            "block_size must be a multiple of 32 in 32..=1024",
        ));
    }
    if blocks_per_pair == 0 {
        return Err(PyValueError::new_err("blocks_per_pair must be positive"));
    }
    let work = num_pairs
        .checked_mul(blocks_per_pair)
        .and_then(|n| n.checked_mul(block_size))
        .ok_or_else(|| PyValueError::new_err("energy launch dimensions overflow"))?;
    {
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
            ],
            &kernel_name(true, data.kind(), cell_tile, feat_tile, block_size),
            work,
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
    let embedding = read(Some(embedding), &cupy, "embedding", "T", false)?;
    let cat_offsets = read(Some(cat_offsets), &cupy, "cat_offsets", "int", false)?;
    let cell_indices = read(Some(cell_indices), &cupy, "cell_indices", "int", false)?;
    let pair_left = read(Some(pair_left), &cupy, "pair_left", "int", false)?;
    let pair_right = read(Some(pair_right), &cupy, "pair_right", "int", false)?;
    let pairwise_sums = read(Some(pairwise_sums), &cupy, "pairwise_sums", "T", false)?;
    same(&pairwise_sums, &embedding)?;
    if block_size == 0 || block_size > 1024 || !block_size.is_multiple_of(32) {
        return Err(PyValueError::new_err(
            "block_size must be a multiple of 32 in 32..=1024",
        ));
    }
    if blocks_per_pair == 0 {
        return Err(PyValueError::new_err("blocks_per_pair must be positive"));
    }
    let work = num_pairs
        .checked_mul(blocks_per_pair)
        .and_then(|n| n.checked_mul(block_size))
        .ok_or_else(|| PyValueError::new_err("energy launch dimensions overflow"))?;
    {
        launch(
            &cupy,
            &[
                &embedding,
                &cat_offsets,
                &cell_indices,
                &pair_left,
                &pair_right,
                &pairwise_sums,
            ],
            &kernel_name(false, embedding.kind(), cell_tile, feat_tile, block_size),
            work,
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
            ],
        )?;
    }
    Ok(())
}
#[pyfunction]
fn get_kernel_config(
    py: Python<'_>,
    n_features: u64,
    is_double: bool,
) -> PyResult<Option<(u64, u64, u64, u64)>> {
    let cp = py.import("cupy")?;
    let runtime = cp.getattr("cuda")?.getattr("runtime")?;
    let device = runtime.call_method0("getDevice")?;
    let properties = runtime.call_method1("getDeviceProperties", (device,))?;
    let shared: u64 = properties.get_item("sharedMemPerBlock")?.extract()?;
    let major: u32 = properties.get_item("major")?.extract()?;
    let (cells, features, block, bytes) = if is_double {
        let available = shared.saturating_sub(32 * 8);
        let mut chosen = 32;
        for tile in [64, 50, 32] {
            if tile * 16 * 8 <= available {
                if n_features.is_multiple_of(tile) {
                    chosen = tile;
                    break;
                }
                if chosen == 32 || tile > chosen {
                    chosen = tile;
                }
            }
        }
        (16, chosen, if major >= 8 { 512 } else { 256 }, 8)
    } else {
        (
            64,
            if n_features.is_multiple_of(25) {
                25
            } else {
                16
            },
            512,
            4,
        )
    };
    let scratch = cells * features * bytes;
    Ok((scratch <= shared).then_some((cells, features, block, scratch)))
}

pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_edistance_cuda")?;
    m.add_function(wrap_pyfunction!(compute_distances_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_distances, &m)?)?;
    m.add_function(wrap_pyfunction!(get_kernel_config, &m)?)?;
    Ok(())
}
