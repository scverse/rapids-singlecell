//! Native nn_descent CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_nn_descent_sqeuclidean(
    data: u64,
    out: u64,
    pairs: u64,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
    pairs_len: u64,
    pairs_kind: u64,
    pairs_order: u64,
    pairs_rows: u64,
    pairs_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let pairs = Buffer {
        pointer: pairs,
        len: pairs_len,
        kind: pairs_kind,
        order: pairs_order,
        rows: pairs_rows,
        cols: pairs_cols,
    };
    let mut p = tid();
    while p < n_samples * n_neighbors {
        let a = p / n_neighbors;
        let b = pairs.i(p);
        if b < n_samples {
            let mut d = 0;
            let mut val = 0f32;
            while d < n_features {
                let x = data.single(a * n_features + d);
                let y = data.single(b * n_features + d);
                val += (x - y) * (x - y);
                d += 1;
            }
            let dist = val;
            out.put_single(p, dist);
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_nn_descent_cosine(
    data: u64,
    out: u64,
    pairs: u64,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
    pairs_len: u64,
    pairs_kind: u64,
    pairs_order: u64,
    pairs_rows: u64,
    pairs_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let pairs = Buffer {
        pointer: pairs,
        len: pairs_len,
        kind: pairs_kind,
        order: pairs_order,
        rows: pairs_rows,
        cols: pairs_cols,
    };
    let mut p = tid();
    while p < n_samples * n_neighbors {
        let a = p / n_neighbors;
        let b = pairs.i(p);
        if b < n_samples {
            let mut d = 0;
            let mut val = 0f32;
            let mut norm_a = 0f32;
            let mut norm_b = 0f32;
            while d < n_features {
                let x = data.single(a * n_features + d);
                let y = data.single(b * n_features + d);
                val += x * y;
                norm_a += x * x;
                norm_b += y * y;
                d += 1;
            }
            let dist = 1.0
                - val
                    * if norm_a > 0.0 {
                        1.0 / norm_a.sqrt()
                    } else {
                        0.0
                    }
                    * if norm_b > 0.0 {
                        1.0 / norm_b.sqrt()
                    } else {
                        0.0
                    };
            out.put_single(p, dist);
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_nn_descent_inner(
    data: u64,
    out: u64,
    pairs: u64,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
    pairs_len: u64,
    pairs_kind: u64,
    pairs_order: u64,
    pairs_rows: u64,
    pairs_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let pairs = Buffer {
        pointer: pairs,
        len: pairs_len,
        kind: pairs_kind,
        order: pairs_order,
        rows: pairs_rows,
        cols: pairs_cols,
    };
    let mut p = tid();
    while p < n_samples * n_neighbors {
        let a = p / n_neighbors;
        let b = pairs.i(p);
        if b < n_samples {
            let mut d = 0;
            let mut val = 0f32;
            while d < n_features {
                let x = data.single(a * n_features + d);
                let y = data.single(b * n_features + d);
                val += x * y;
                d += 1;
            }
            let dist = val;
            out.put_single(p, dist);
        }
        p += stride();
    }
}
