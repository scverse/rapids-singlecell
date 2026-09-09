//! Native aucell CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aucell_auc(
    ranks: u64,
    R: u64,
    C: u64,
    cnct: u64,
    starts: u64,
    lens: u64,
    n_sets: u64,
    n_up: u64,
    max_aucs: u64,
    es: u64,
    ranks_len: u64,
    ranks_kind: u64,
    ranks_order: u64,
    ranks_rows: u64,
    ranks_cols: u64,
    cnct_len: u64,
    cnct_kind: u64,
    cnct_order: u64,
    cnct_rows: u64,
    cnct_cols: u64,
    starts_len: u64,
    starts_kind: u64,
    starts_order: u64,
    starts_rows: u64,
    starts_cols: u64,
    lens_len: u64,
    lens_kind: u64,
    lens_order: u64,
    lens_rows: u64,
    lens_cols: u64,
    max_aucs_len: u64,
    max_aucs_kind: u64,
    max_aucs_order: u64,
    max_aucs_rows: u64,
    max_aucs_cols: u64,
    es_len: u64,
    es_kind: u64,
    es_order: u64,
    es_rows: u64,
    es_cols: u64,
) {
    let ranks = Buffer {
        pointer: ranks,
        len: ranks_len,
        kind: ranks_kind,
        order: ranks_order,
        rows: ranks_rows,
        cols: ranks_cols,
    };
    let cnct = Buffer {
        pointer: cnct,
        len: cnct_len,
        kind: cnct_kind,
        order: cnct_order,
        rows: cnct_rows,
        cols: cnct_cols,
    };
    let starts = Buffer {
        pointer: starts,
        len: starts_len,
        kind: starts_kind,
        order: starts_order,
        rows: starts_rows,
        cols: starts_cols,
    };
    let lens = Buffer {
        pointer: lens,
        len: lens_len,
        kind: lens_kind,
        order: lens_order,
        rows: lens_rows,
        cols: lens_cols,
    };
    let max_aucs = Buffer {
        pointer: max_aucs,
        len: max_aucs_len,
        kind: max_aucs_kind,
        order: max_aucs_order,
        rows: max_aucs_rows,
        cols: max_aucs_cols,
    };
    let es = Buffer {
        pointer: es,
        len: es_len,
        kind: es_kind,
        order: es_order,
        rows: es_rows,
        cols: es_cols,
    };
    let mut p = tid();
    while p < R * n_sets {
        let row = p / n_sets;
        let s = p % n_sets;
        let start = starts.i(s);
        let end = (start + lens.i(s)).min(cnct.len);
        let mut j = start;
        let mut score = 0i64;
        while j < end {
            let g = cnct.i(j);
            if g < C {
                let rank = ranks.i(row * C + g);
                if rank <= n_up {
                    score += (n_up - rank) as i64;
                }
            }
            j += 1;
        }
        es.put_single(p, score as f32 / max_aucs.single(s));
        p += stride();
    }
}
