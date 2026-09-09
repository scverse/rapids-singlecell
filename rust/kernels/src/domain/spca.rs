//! Native spca CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_spca_gram_csr_upper(
    indptr: u64,
    index: u64,
    data: u64,
    nrows: u64,
    ncols: u64,
    out: u64,
    indptr_len: u64,
    indptr_kind: u64,
    indptr_order: u64,
    indptr_rows: u64,
    indptr_cols: u64,
    index_len: u64,
    index_kind: u64,
    index_order: u64,
    index_rows: u64,
    index_cols: u64,
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
) {
    let indptr = Buffer {
        pointer: indptr,
        len: indptr_len,
        kind: indptr_kind,
        order: indptr_order,
        rows: indptr_rows,
        cols: indptr_cols,
    };
    let index = Buffer {
        pointer: index,
        len: index_len,
        kind: index_kind,
        order: index_order,
        rows: index_rows,
        cols: index_cols,
    };
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
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < nrows {
        let start = indptr.i(row);
        let end = indptr.i(row + 1).min(data.len).min(index.len);
        let mut p = start;
        while p < end {
            let a = index.i(p);
            let mut q = p + lane;
            while q < end {
                let b = index.i(q);
                if a < ncols && b < ncols {
                    out.add(
                        a.min(b) * ncols + a.max(b),
                        if data.kind == 0 {
                            (data.single(p) * data.single(q)) as f64
                        } else {
                            data.f(p) * data.f(q)
                        },
                    );
                }
                q += 128;
            }
            p += 1;
        }
        row += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_spca_copy_upper_to_lower(
    out: u64,
    ncols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let mut p = tid();
    while p < ncols * ncols {
        let r = p / ncols;
        let c = p % ncols;
        if r > c {
            out.put(p, out.f(c * ncols + r));
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_spca_cov_from_gram(
    gram: u64,
    meanx: u64,
    meany: u64,
    cov: u64,
    ncols: u64,
    gram_len: u64,
    gram_kind: u64,
    gram_order: u64,
    gram_rows: u64,
    gram_cols: u64,
    meanx_len: u64,
    meanx_kind: u64,
    meanx_order: u64,
    meanx_rows: u64,
    meanx_cols: u64,
    meany_len: u64,
    meany_kind: u64,
    meany_order: u64,
    meany_rows: u64,
    meany_cols: u64,
    cov_len: u64,
    cov_kind: u64,
    cov_order: u64,
    cov_rows: u64,
    cov_cols: u64,
) {
    let gram = Buffer {
        pointer: gram,
        len: gram_len,
        kind: gram_kind,
        order: gram_order,
        rows: gram_rows,
        cols: gram_cols,
    };
    let meanx = Buffer {
        pointer: meanx,
        len: meanx_len,
        kind: meanx_kind,
        order: meanx_order,
        rows: meanx_rows,
        cols: meanx_cols,
    };
    let meany = Buffer {
        pointer: meany,
        len: meany_len,
        kind: meany_kind,
        order: meany_order,
        rows: meany_rows,
        cols: meany_cols,
    };
    let cov = Buffer {
        pointer: cov,
        len: cov_len,
        kind: cov_kind,
        order: cov_order,
        rows: cov_rows,
        cols: cov_cols,
    };
    let mut p = tid();
    while p < ncols * ncols {
        cov.put(
            p,
            if gram.kind == 0 {
                (gram.single(p) - meanx.single(p / ncols) * meany.single(p % ncols)) as f64
            } else {
                gram.f(p) - meanx.f(p / ncols) * meany.f(p % ncols)
            },
        );
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_spca_check_zero_genes(
    indices: u64,
    out: u64,
    nnz: u64,
    num_genes: u64,
    indices_len: u64,
    indices_kind: u64,
    indices_order: u64,
    indices_rows: u64,
    indices_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let indices = Buffer {
        pointer: indices,
        len: indices_len,
        kind: indices_kind,
        order: indices_order,
        rows: indices_rows,
        cols: indices_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let mut p = tid();
    while p < nnz {
        let g = indices.i(p);
        if g < num_genes {
            out.add(g, 1.0);
        }
        p += stride();
    }
}
