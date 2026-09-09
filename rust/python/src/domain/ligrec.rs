//! ligrec bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(data, *, clusters, sum, count, rows, cols, ncls, stream=0))]
fn sum_count_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    clusters: &Bound<'_, PyAny>,
    sum: &Bound<'_, PyAny>,
    count: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "T", false)?;
    let clusters = read(Some(clusters), &cupy, "clusters", "int", false)?;
    let sum = read(Some(sum), &cupy, "sum", "T", false)?;
    same(&sum, &data)?;
    let count = read(Some(count), &cupy, "count", "int", false)?;
    launch(
        &cupy,
        &[&data, &clusters, &sum, &count],
        "domain_ligrec_sum_count_dense",
        rows.div_ceil(32) * cols.div_ceil(32) * 1024,
        stream,
        vec![
            data.pointer(),
            clusters.pointer(),
            sum.pointer(),
            count.pointer(),
            rows,
            cols,
            ncls,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, clusters, sum, count, rows, ncls, stream=0))]
fn sum_count_sparse(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    clusters: &Bound<'_, PyAny>,
    sum: &Bound<'_, PyAny>,
    count: &Bound<'_, PyAny>,
    rows: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let clusters = read(Some(clusters), &cupy, "clusters", "int", false)?;
    let sum = read(Some(sum), &cupy, "sum", "T", false)?;
    same(&sum, &data)?;
    let count = read(Some(count), &cupy, "count", "int", false)?;
    launch(
        &cupy,
        &[&indptr, &index, &data, &clusters, &sum, &count],
        "domain_ligrec_sum_count_sparse",
        rows * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            clusters.pointer(),
            sum.pointer(),
            count.pointer(),
            rows,
            ncls,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, *, clusters, g, rows, cols, ncls, stream=0))]
fn mean_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    clusters: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "T", false)?;
    let clusters = read(Some(clusters), &cupy, "clusters", "int", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &data)?;
    launch(
        &cupy,
        &[&data, &clusters, &g],
        "domain_ligrec_mean_dense",
        rows.div_ceil(32) * cols.div_ceil(32) * 1024,
        stream,
        vec![
            data.pointer(),
            clusters.pointer(),
            g.pointer(),
            rows,
            cols,
            ncls,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, clusters, g, rows, ncls, stream=0))]
fn mean_sparse(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    clusters: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    rows: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let clusters = read(Some(clusters), &cupy, "clusters", "int", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &data)?;
    launch(
        &cupy,
        &[&indptr, &index, &data, &clusters, &g],
        "domain_ligrec_mean_sparse",
        rows * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            clusters.pointer(),
            g.pointer(),
            rows,
            ncls,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(g, *, total_counts, n_genes, n_clusters, stream=0))]
fn elementwise_diff(
    py: Python<'_>,
    g: &Bound<'_, PyAny>,
    total_counts: &Bound<'_, PyAny>,
    n_genes: u64,
    n_clusters: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let g = read(Some(g), &cupy, "g", "T", false)?;
    let total_counts = read(Some(total_counts), &cupy, "total_counts", "T", false)?;
    same(&total_counts, &g)?;
    launch(
        &cupy,
        &[&g, &total_counts],
        "domain_ligrec_elementwise_diff",
        n_genes * n_clusters,
        stream,
        vec![g.pointer(), total_counts.pointer(), n_genes, n_clusters],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(interactions, *, interaction_clusters, mean, res, mask, g, n_iter, n_inter_clust, ncls, stream=0))]
fn interaction(
    py: Python<'_>,
    interactions: &Bound<'_, PyAny>,
    interaction_clusters: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    res: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    n_iter: u64,
    n_inter_clust: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let interactions = read(Some(interactions), &cupy, "interactions", "int", false)?;
    let interaction_clusters = read(
        Some(interaction_clusters),
        &cupy,
        "interaction_clusters",
        "Index",
        false,
    )?;
    let mean = read(Some(mean), &cupy, "mean", "T", false)?;
    let res = read(Some(res), &cupy, "res", "T", false)?;
    same(&res, &mean)?;
    let mask = read(Some(mask), &cupy, "mask", "bool", false)?;
    let g = read(Some(g), &cupy, "g", "T", false)?;
    same(&g, &mean)?;
    launch(
        &cupy,
        &[&interactions, &interaction_clusters, &mean, &res, &mask, &g],
        "domain_ligrec_interaction",
        n_iter * n_inter_clust,
        stream,
        vec![
            interactions.pointer(),
            interaction_clusters.pointer(),
            mean.pointer(),
            res.pointer(),
            mask.pointer(),
            g.pointer(),
            n_iter,
            n_inter_clust,
            ncls,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(interactions, *, interaction_clusters, mean, res_mean, n_inter, n_inter_clust, ncls, stream=0))]
fn res_mean(
    py: Python<'_>,
    interactions: &Bound<'_, PyAny>,
    interaction_clusters: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    res_mean: &Bound<'_, PyAny>,
    n_inter: u64,
    n_inter_clust: u64,
    ncls: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let interactions = read(Some(interactions), &cupy, "interactions", "int", false)?;
    let interaction_clusters = read(
        Some(interaction_clusters),
        &cupy,
        "interaction_clusters",
        "Index",
        false,
    )?;
    let mean = read(Some(mean), &cupy, "mean", "T", false)?;
    let res_mean = read(Some(res_mean), &cupy, "res_mean", "T", false)?;
    same(&res_mean, &mean)?;
    launch(
        &cupy,
        &[&interactions, &interaction_clusters, &mean, &res_mean],
        "domain_ligrec_res_mean",
        n_inter * n_inter_clust,
        stream,
        vec![
            interactions.pointer(),
            interaction_clusters.pointer(),
            mean.pointer(),
            res_mean.pointer(),
            n_inter,
            n_inter_clust,
            ncls,
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_ligrec_cuda")?;
    m.add_function(wrap_pyfunction!(sum_count_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(sum_count_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(elementwise_diff, &m)?)?;
    m.add_function(wrap_pyfunction!(interaction, &m)?)?;
    m.add_function(wrap_pyfunction!(res_mean, &m)?)?;
    Ok(())
}
