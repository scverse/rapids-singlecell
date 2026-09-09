//! Native pv CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pv_rev_cummin64(
    x: u64,
    out: u64,
    n_rows: u64,
    m: u64,
    x_len: u64,
    x_kind: u64,
    x_order: u64,
    x_rows: u64,
    x_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let x = Buffer {
        pointer: x,
        len: x_len,
        kind: x_kind,
        order: x_order,
        rows: x_rows,
        cols: x_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let mut r = tid();
    while r < n_rows {
        let mut j = m;
        let mut v = f64::INFINITY;
        while j > 0 {
            j -= 1;
            let q = x.f(r * m + j);
            v = if q < v { q } else { v };
            out.put(r * m + j, v);
        }
        r += stride();
    }
}
