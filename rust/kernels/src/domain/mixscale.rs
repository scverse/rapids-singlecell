//! Native mixscale CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_mixscale_project_score(
    X: u64,
    n_vars: u64,
    row_ids: u64,
    col_ids: u64,
    n_per_gene: u64,
    k_per_gene: u64,
    cell_offsets: u64,
    feat_offsets: u64,
    is_guide: u64,
    nt_in_all: u64,
    pvec_scratch: u64,
    scores_out: u64,
    n_genes: u64,
    max_k: u64,
    do_scale: u64,
    workspace: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    row_ids_len: u64,
    row_ids_kind: u64,
    row_ids_order: u64,
    row_ids_rows: u64,
    row_ids_cols: u64,
    col_ids_len: u64,
    col_ids_kind: u64,
    col_ids_order: u64,
    col_ids_rows: u64,
    col_ids_cols: u64,
    n_per_gene_len: u64,
    n_per_gene_kind: u64,
    n_per_gene_order: u64,
    n_per_gene_rows: u64,
    n_per_gene_cols: u64,
    k_per_gene_len: u64,
    k_per_gene_kind: u64,
    k_per_gene_order: u64,
    k_per_gene_rows: u64,
    k_per_gene_cols: u64,
    cell_offsets_len: u64,
    cell_offsets_kind: u64,
    cell_offsets_order: u64,
    cell_offsets_rows: u64,
    cell_offsets_cols: u64,
    feat_offsets_len: u64,
    feat_offsets_kind: u64,
    feat_offsets_order: u64,
    feat_offsets_rows: u64,
    feat_offsets_cols: u64,
    is_guide_len: u64,
    is_guide_kind: u64,
    is_guide_order: u64,
    is_guide_rows: u64,
    is_guide_cols: u64,
    nt_in_all_len: u64,
    nt_in_all_kind: u64,
    nt_in_all_order: u64,
    nt_in_all_rows: u64,
    nt_in_all_cols: u64,
    pvec_scratch_len: u64,
    pvec_scratch_kind: u64,
    pvec_scratch_order: u64,
    pvec_scratch_rows: u64,
    pvec_scratch_cols: u64,
    scores_out_len: u64,
    scores_out_kind: u64,
    scores_out_order: u64,
    scores_out_rows: u64,
    scores_out_cols: u64,
    workspace_len: u64,
    workspace_kind: u64,
    workspace_order: u64,
    workspace_rows: u64,
    workspace_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let row_ids = Buffer {
        pointer: row_ids,
        len: row_ids_len,
        kind: row_ids_kind,
        order: row_ids_order,
        rows: row_ids_rows,
        cols: row_ids_cols,
    };
    let col_ids = Buffer {
        pointer: col_ids,
        len: col_ids_len,
        kind: col_ids_kind,
        order: col_ids_order,
        rows: col_ids_rows,
        cols: col_ids_cols,
    };
    let n_per_gene = Buffer {
        pointer: n_per_gene,
        len: n_per_gene_len,
        kind: n_per_gene_kind,
        order: n_per_gene_order,
        rows: n_per_gene_rows,
        cols: n_per_gene_cols,
    };
    let k_per_gene = Buffer {
        pointer: k_per_gene,
        len: k_per_gene_len,
        kind: k_per_gene_kind,
        order: k_per_gene_order,
        rows: k_per_gene_rows,
        cols: k_per_gene_cols,
    };
    let cell_offsets = Buffer {
        pointer: cell_offsets,
        len: cell_offsets_len,
        kind: cell_offsets_kind,
        order: cell_offsets_order,
        rows: cell_offsets_rows,
        cols: cell_offsets_cols,
    };
    let feat_offsets = Buffer {
        pointer: feat_offsets,
        len: feat_offsets_len,
        kind: feat_offsets_kind,
        order: feat_offsets_order,
        rows: feat_offsets_rows,
        cols: feat_offsets_cols,
    };
    let is_guide = Buffer {
        pointer: is_guide,
        len: is_guide_len,
        kind: is_guide_kind,
        order: is_guide_order,
        rows: is_guide_rows,
        cols: is_guide_cols,
    };
    let nt_in_all = Buffer {
        pointer: nt_in_all,
        len: nt_in_all_len,
        kind: nt_in_all_kind,
        order: nt_in_all_order,
        rows: nt_in_all_rows,
        cols: nt_in_all_cols,
    };
    let pvec_scratch = Buffer {
        pointer: pvec_scratch,
        len: pvec_scratch_len,
        kind: pvec_scratch_kind,
        order: pvec_scratch_order,
        rows: pvec_scratch_rows,
        cols: pvec_scratch_cols,
    };
    let scores_out = Buffer {
        pointer: scores_out,
        len: scores_out_len,
        kind: scores_out_kind,
        order: scores_out_order,
        rows: scores_out_rows,
        cols: scores_out_cols,
    };
    let workspace = Buffer {
        pointer: workspace,
        len: workspace_len,
        kind: workspace_kind,
        order: workspace_order,
        rows: workspace_rows,
        cols: workspace_cols,
    };
    let mut gene = tid() / 256;
    let lane = tid() % 256;
    while gene < n_genes {
        let n = n_per_gene.i(gene);
        let k = k_per_gene.i(gene);
        let co = cell_offsets.i(gene);
        let feature_offset = feat_offsets.i(gene);
        if n > 0 && k > 0 {
            let mut ng = 0.0f64;
            let mut nt = 0.0f64;
            let mut cell = lane;
            while cell < n {
                ng += f64::from(is_guide.i(co + cell) != 0);
                nt += f64::from(nt_in_all.i(co + cell) != 0);
                cell += 256;
            }
            ng = domain_block_total(ng).max(1.0);
            nt = domain_block_total(nt).max(1.0);
            let mut j = lane / 32;
            while j < k {
                let col = col_ids.i(feature_offset + j);
                let mut sum = 0.0;
                let mut square = 0.0;
                let mut gs = 0.0;
                let mut ts = 0.0;
                cell = lane % 32;
                while cell < n {
                    let v = X.f(row_ids.i(co + cell) * n_vars + col);
                    sum += v;
                    square += v * v;
                    if is_guide.i(co + cell) != 0 {
                        gs += v;
                    }
                    if nt_in_all.i(co + cell) != 0 {
                        ts += v;
                    }
                    cell += 32;
                }
                sum = domain_warp_total(sum);
                square = domain_warp_total(square);
                gs = domain_warp_total(gs);
                ts = domain_warp_total(ts);
                let mean = if do_scale != 0 { sum / n as f64 } else { 0.0 };
                let variance = if n > 1 {
                    (square - sum * sum / n as f64) / (n - 1) as f64
                } else {
                    0.0
                };
                let sd = if do_scale != 0 {
                    let s = variance.max(0.0).sqrt();
                    if s == 0.0 { 1.0 } else { s }
                } else {
                    1.0
                };
                if lane.is_multiple_of(32) {
                    workspace.put((feature_offset + j) * 3, mean);
                    workspace.put((feature_offset + j) * 3 + 1, sd);
                    workspace.put((feature_offset + j) * 3 + 2, (gs / ng - ts / nt) / sd);
                }
                j += 8;
            }
            cuda_device::thread::sync_threads();
            let mut dot = 0.0;
            j = lane;
            while j < k {
                let v = workspace.f((feature_offset + j) * 3 + 2);
                dot += v * v;
                j += 256;
            }
            dot = domain_block_total(dot);
            cell = lane;
            while cell < n {
                let row = row_ids.i(co + cell);
                let mut p = 0.0;
                j = 0;
                while j < k {
                    let v = (X.f(row * n_vars + col_ids.i(feature_offset + j))
                        - workspace.f((feature_offset + j) * 3))
                        / workspace.f((feature_offset + j) * 3 + 1);
                    p += v * workspace.f((feature_offset + j) * 3 + 2);
                    j += 1;
                }
                pvec_scratch.put(co + cell, p / dot.max(1e-12));
                cell += 256;
            }
            cuda_device::thread::sync_threads();
            let mut total = 0.0;
            let mut square = 0.0;
            cell = lane;
            while cell < n {
                if nt_in_all.i(co + cell) != 0 {
                    let v = pvec_scratch.f(co + cell);
                    total += v;
                    square += v * v;
                }
                cell += 256;
            }
            total = domain_block_total(total);
            square = domain_block_total(square);
            let mean = total / nt;
            let variance = if nt > 1.0 {
                (square / nt - mean * mean) * nt / (nt - 1.0)
            } else {
                0.0
            };
            let sd = variance.max(0.0).sqrt();
            let sd = if sd == 0.0 { 1.0 } else { sd };
            cell = lane;
            while cell < n {
                if is_guide.i(co + cell) != 0 {
                    scores_out.put(
                        row_ids.i(co + cell),
                        (pvec_scratch.f(co + cell) - mean) / sd,
                    );
                }
                cell += 256;
            }
            cuda_device::thread::sync_threads();
        }
        gene += stride() / 256;
    }
}
