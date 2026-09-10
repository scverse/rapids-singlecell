//! Native pseudobulk CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pseudobulk_paired_squared(
    X: u64,
    Y: u64,
    out: u64,
    n_pairs: u64,
    n_features: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    Y_len: u64,
    Y_kind: u64,
    Y_order: u64,
    Y_rows: u64,
    Y_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let Y = Buffer {
        pointer: Y,
        len: Y_len,
        kind: Y_kind,
        order: Y_order,
        rows: Y_rows,
        cols: Y_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let block = thread::blockDim_x() as u64;
    let mut p = tid() / block;
    let lane = thread::threadIdx_x() as u64;
    while p < n_pairs {
        let xi = p;
        let yi = p;
        let mut d = lane;
        let mut acc = 0.0;
        while d < n_features {
            let diff = unsafe {
                *(X.pointer as *const f64).add((xi * n_features + d) as usize)
                    - *(Y.pointer as *const f64).add((yi * n_features + d) as usize)
            };
            acc += diff * diff;
            d += block;
        }
        let acc = domain_block_total(acc);
        if lane == 0 {
            out.put(p, acc);
        }
        p += stride() / block;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pseudobulk_paired_abs_mean(
    X: u64,
    Y: u64,
    out: u64,
    n_pairs: u64,
    n_features: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    Y_len: u64,
    Y_kind: u64,
    Y_order: u64,
    Y_rows: u64,
    Y_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let Y = Buffer {
        pointer: Y,
        len: Y_len,
        kind: Y_kind,
        order: Y_order,
        rows: Y_rows,
        cols: Y_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let block = thread::blockDim_x() as u64;
    let mut p = tid() / block;
    let lane = thread::threadIdx_x() as u64;
    while p < n_pairs {
        let xi = p;
        let yi = p;
        let mut d = lane;
        let mut acc = 0.0;
        while d < n_features {
            let diff = unsafe {
                *(X.pointer as *const f64).add((xi * n_features + d) as usize)
                    - *(Y.pointer as *const f64).add((yi * n_features + d) as usize)
            };
            acc += diff.abs();
            d += block;
        }
        let acc = domain_block_total(acc);
        if lane == 0 {
            out.put(p, acc / n_features as f64);
        }
        p += stride() / block;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pseudobulk_pairwise_squared(
    X: u64,
    Y: u64,
    out: u64,
    n_x: u64,
    n_y: u64,
    n_features: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    Y_len: u64,
    Y_kind: u64,
    Y_order: u64,
    Y_rows: u64,
    Y_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let Y = Buffer {
        pointer: Y,
        len: Y_len,
        kind: Y_kind,
        order: Y_order,
        rows: Y_rows,
        cols: Y_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let block = thread::blockDim_x() as u64;
    let mut p = tid() / block;
    let lane = thread::threadIdx_x() as u64;
    while p < n_x * n_y {
        let xi = p / n_y;
        let yi = p % n_y;
        let mut d = lane;
        let mut acc = 0.0;
        while d < n_features {
            let diff = unsafe {
                *(X.pointer as *const f64).add((xi * n_features + d) as usize)
                    - *(Y.pointer as *const f64).add((yi * n_features + d) as usize)
            };
            acc += diff * diff;
            d += block;
        }
        let acc = domain_block_total(acc);
        if lane == 0 {
            out.put(p, acc);
        }
        p += stride() / block;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pseudobulk_pairwise_abs_mean(
    X: u64,
    Y: u64,
    out: u64,
    n_x: u64,
    n_y: u64,
    n_features: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    Y_len: u64,
    Y_kind: u64,
    Y_order: u64,
    Y_rows: u64,
    Y_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let Y = Buffer {
        pointer: Y,
        len: Y_len,
        kind: Y_kind,
        order: Y_order,
        rows: Y_rows,
        cols: Y_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let block = thread::blockDim_x() as u64;
    let mut p = tid() / block;
    let lane = thread::threadIdx_x() as u64;
    while p < n_x * n_y {
        let xi = p / n_y;
        let yi = p % n_y;
        let mut d = lane;
        let mut acc = 0.0;
        while d < n_features {
            let diff = unsafe {
                *(X.pointer as *const f64).add((xi * n_features + d) as usize)
                    - *(Y.pointer as *const f64).add((yi * n_features + d) as usize)
            };
            acc += diff.abs();
            d += block;
        }
        let acc = domain_block_total(acc);
        if lane == 0 {
            out.put(p, acc / n_features as f64);
        }
        p += stride() / block;
    }
}
