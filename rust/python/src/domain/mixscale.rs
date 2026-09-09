//! mixscale bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(X, n_vars, row_ids, col_ids, n_per_gene, k_per_gene, cell_offsets, feat_offsets, is_guide, nt_in_all, pvec_scratch, scores_out, *, n_genes, max_k, do_scale, stream=0))]
fn project_score(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    n_vars: u64,
    row_ids: &Bound<'_, PyAny>,
    col_ids: &Bound<'_, PyAny>,
    n_per_gene: &Bound<'_, PyAny>,
    k_per_gene: &Bound<'_, PyAny>,
    cell_offsets: &Bound<'_, PyAny>,
    feat_offsets: &Bound<'_, PyAny>,
    is_guide: &Bound<'_, PyAny>,
    nt_in_all: &Bound<'_, PyAny>,
    pvec_scratch: &Bound<'_, PyAny>,
    scores_out: &Bound<'_, PyAny>,
    n_genes: u64,
    max_k: u64,
    do_scale: bool,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;
    let _stream_scope = runtime::StreamScope::new(&cupy, stream)?;
    let workspace_object = cupy.call_method1(
        "empty",
        (3 * col_ids.getattr("size")?.extract::<u64>()?, "float64"),
    )?;
    let workspace = &workspace_object;
    let X = read(Some(X), &cupy, "X", "T", false)?;
    let row_ids = read(Some(row_ids), &cupy, "row_ids", "int", false)?;
    let col_ids = read(Some(col_ids), &cupy, "col_ids", "int", false)?;
    let n_per_gene = read(Some(n_per_gene), &cupy, "n_per_gene", "int", false)?;
    let k_per_gene = read(Some(k_per_gene), &cupy, "k_per_gene", "int", false)?;
    let cell_offsets = read(Some(cell_offsets), &cupy, "cell_offsets", "int", false)?;
    let feat_offsets = read(Some(feat_offsets), &cupy, "feat_offsets", "int", false)?;
    let is_guide = read(Some(is_guide), &cupy, "is_guide", "bool", false)?;
    let nt_in_all = read(Some(nt_in_all), &cupy, "nt_in_all", "bool", false)?;
    let pvec_scratch = read(Some(pvec_scratch), &cupy, "pvec_scratch", "double", false)?;
    let scores_out = read(Some(scores_out), &cupy, "scores_out", "T", false)?;
    same(&scores_out, &X)?;
    let workspace = read(Some(workspace), &cupy, "workspace", "double", false)?;
    launch(
        &cupy,
        &[
            &X,
            &row_ids,
            &col_ids,
            &n_per_gene,
            &k_per_gene,
            &cell_offsets,
            &feat_offsets,
            &is_guide,
            &nt_in_all,
            &pvec_scratch,
            &scores_out,
            &workspace,
        ],
        "domain_mixscale_project_score",
        n_genes * 256,
        stream,
        vec![
            X.pointer(),
            n_vars,
            row_ids.pointer(),
            col_ids.pointer(),
            n_per_gene.pointer(),
            k_per_gene.pointer(),
            cell_offsets.pointer(),
            feat_offsets.pointer(),
            is_guide.pointer(),
            nt_in_all.pointer(),
            pvec_scratch.pointer(),
            scores_out.pointer(),
            n_genes,
            max_k,
            u64::from(do_scale),
            workspace.pointer(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_mixscale_cuda")?;
    m.add_function(wrap_pyfunction!(project_score, &m)?)?;
    Ok(())
}
