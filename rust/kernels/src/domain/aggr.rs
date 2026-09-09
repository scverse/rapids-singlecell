//! Native aggr CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aggr_sparse_aggr(
    indptr: u64,
    index: u64,
    data: u64,
    out_sum: u64,
    out_count: u64,
    out_sqsum: u64,
    cats: u64,
    mask: u64,
    n_cells: u64,
    n_genes: u64,
    is_csc: u64,
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
    out_sum_len: u64,
    out_sum_kind: u64,
    out_sum_order: u64,
    out_sum_rows: u64,
    out_sum_cols: u64,
    out_count_len: u64,
    out_count_kind: u64,
    out_count_order: u64,
    out_count_rows: u64,
    out_count_cols: u64,
    out_sqsum_len: u64,
    out_sqsum_kind: u64,
    out_sqsum_order: u64,
    out_sqsum_rows: u64,
    out_sqsum_cols: u64,
    cats_len: u64,
    cats_kind: u64,
    cats_order: u64,
    cats_rows: u64,
    cats_cols: u64,
    mask_len: u64,
    mask_kind: u64,
    mask_order: u64,
    mask_rows: u64,
    mask_cols: u64,
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
    let out_sum = Buffer {
        pointer: out_sum,
        len: out_sum_len,
        kind: out_sum_kind,
        order: out_sum_order,
        rows: out_sum_rows,
        cols: out_sum_cols,
    };
    let out_count = Buffer {
        pointer: out_count,
        len: out_count_len,
        kind: out_count_kind,
        order: out_count_order,
        rows: out_count_rows,
        cols: out_count_cols,
    };
    let out_sqsum = Buffer {
        pointer: out_sqsum,
        len: out_sqsum_len,
        kind: out_sqsum_kind,
        order: out_sqsum_order,
        rows: out_sqsum_rows,
        cols: out_sqsum_cols,
    };
    let cats = Buffer {
        pointer: cats,
        len: cats_len,
        kind: cats_kind,
        order: cats_order,
        rows: cats_rows,
        cols: cats_cols,
    };
    let mask = Buffer {
        pointer: mask,
        len: mask_len,
        kind: mask_kind,
        order: mask_order,
        rows: mask_rows,
        cols: mask_cols,
    };
    let major = if is_csc != 0 { n_genes } else { n_cells };
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < major {
        let mut p = indptr.i(row) + lane;
        let end = indptr.i(row + 1).min(data.len).min(index.len);
        while p < end {
            let cell = if is_csc != 0 { index.i(p) } else { row };
            let gene = if is_csc != 0 { row } else { index.i(p) };
            if cell < n_cells && gene < n_genes && mask.i(cell) != 0 {
                let group = cats.i(cell);
                let out = group * n_genes + gene;
                let v = data.f(p);
                out_sum.add(out, v);
                out_count.add(out, 1.0);
                out_sqsum.add(out, v * v);
            }
            p += 128;
        }
        row += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aggr_dense_aggr(
    data: u64,
    out_sum: u64,
    out_count: u64,
    out_sqsum: u64,
    cats: u64,
    mask: u64,
    n_cells: u64,
    n_genes: u64,
    is_fortran: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    out_sum_len: u64,
    out_sum_kind: u64,
    out_sum_order: u64,
    out_sum_rows: u64,
    out_sum_cols: u64,
    out_count_len: u64,
    out_count_kind: u64,
    out_count_order: u64,
    out_count_rows: u64,
    out_count_cols: u64,
    out_sqsum_len: u64,
    out_sqsum_kind: u64,
    out_sqsum_order: u64,
    out_sqsum_rows: u64,
    out_sqsum_cols: u64,
    cats_len: u64,
    cats_kind: u64,
    cats_order: u64,
    cats_rows: u64,
    cats_cols: u64,
    mask_len: u64,
    mask_kind: u64,
    mask_order: u64,
    mask_rows: u64,
    mask_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let out_sum = Buffer {
        pointer: out_sum,
        len: out_sum_len,
        kind: out_sum_kind,
        order: out_sum_order,
        rows: out_sum_rows,
        cols: out_sum_cols,
    };
    let out_count = Buffer {
        pointer: out_count,
        len: out_count_len,
        kind: out_count_kind,
        order: out_count_order,
        rows: out_count_rows,
        cols: out_count_cols,
    };
    let out_sqsum = Buffer {
        pointer: out_sqsum,
        len: out_sqsum_len,
        kind: out_sqsum_kind,
        order: out_sqsum_order,
        rows: out_sqsum_rows,
        cols: out_sqsum_cols,
    };
    let cats = Buffer {
        pointer: cats,
        len: cats_len,
        kind: cats_kind,
        order: cats_order,
        rows: cats_rows,
        cols: cats_cols,
    };
    let mask = Buffer {
        pointer: mask,
        len: mask_len,
        kind: mask_kind,
        order: mask_order,
        rows: mask_rows,
        cols: mask_cols,
    };
    let mut p = tid();
    while p < n_cells * n_genes {
        let cell = if data.order == 0 {
            p / n_genes
        } else {
            p % n_cells
        };
        let gene = if data.order == 0 {
            p % n_genes
        } else {
            p / n_cells
        };
        if mask.i(cell) != 0 {
            let v = data.f(p);
            let group = cats.i(cell);
            if v != 0.0 && group < out_sum.len.max(out_count.len).max(out_sqsum.len) / n_genes {
                let out = group * n_genes + gene;
                out_sum.add(out, v);
                out_count.add(out, 1.0);
                out_sqsum.add(out, v * v);
            }
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aggr_csr_to_coo(
    indptr: u64,
    index: u64,
    data: u64,
    out_row: u64,
    out_col: u64,
    out_data: u64,
    cats: u64,
    mask: u64,
    n_cells: u64,
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
    out_row_len: u64,
    out_row_kind: u64,
    out_row_order: u64,
    out_row_rows: u64,
    out_row_cols: u64,
    out_col_len: u64,
    out_col_kind: u64,
    out_col_order: u64,
    out_col_rows: u64,
    out_col_cols: u64,
    out_data_len: u64,
    out_data_kind: u64,
    out_data_order: u64,
    out_data_rows: u64,
    out_data_cols: u64,
    cats_len: u64,
    cats_kind: u64,
    cats_order: u64,
    cats_rows: u64,
    cats_cols: u64,
    mask_len: u64,
    mask_kind: u64,
    mask_order: u64,
    mask_rows: u64,
    mask_cols: u64,
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
    let out_row = Buffer {
        pointer: out_row,
        len: out_row_len,
        kind: out_row_kind,
        order: out_row_order,
        rows: out_row_rows,
        cols: out_row_cols,
    };
    let out_col = Buffer {
        pointer: out_col,
        len: out_col_len,
        kind: out_col_kind,
        order: out_col_order,
        rows: out_col_rows,
        cols: out_col_cols,
    };
    let out_data = Buffer {
        pointer: out_data,
        len: out_data_len,
        kind: out_data_kind,
        order: out_data_order,
        rows: out_data_rows,
        cols: out_data_cols,
    };
    let cats = Buffer {
        pointer: cats,
        len: cats_len,
        kind: cats_kind,
        order: cats_order,
        rows: cats_rows,
        cols: cats_cols,
    };
    let mask = Buffer {
        pointer: mask,
        len: mask_len,
        kind: mask_kind,
        order: mask_order,
        rows: mask_rows,
        cols: mask_cols,
    };
    let mut cell = tid() / 128;
    let lane = tid() % 128;
    while cell < n_cells {
        if mask.i(cell) != 0 {
            let mut p = indptr.i(cell) + lane;
            let end = indptr.i(cell + 1).min(data.len).min(index.len);
            while p < end {
                out_data.put(p, data.f(p));
                out_row.put_i(p, cats.i(cell));
                out_col.put_i(p, index.i(p));
                p += 128;
            }
        }
        cell += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aggr_sparse_var(
    indptr: u64,
    index: u64,
    data: u64,
    means: u64,
    n_cells: u64,
    dof: u64,
    n_groups: u64,
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
    means_len: u64,
    means_kind: u64,
    means_order: u64,
    means_rows: u64,
    means_cols: u64,
    n_cells_len: u64,
    n_cells_kind: u64,
    n_cells_order: u64,
    n_cells_rows: u64,
    n_cells_cols: u64,
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
    let means = Buffer {
        pointer: means,
        len: means_len,
        kind: means_kind,
        order: means_order,
        rows: means_rows,
        cols: means_cols,
    };
    let n_cells = Buffer {
        pointer: n_cells,
        len: n_cells_len,
        kind: n_cells_kind,
        order: n_cells_order,
        rows: n_cells_rows,
        cols: n_cells_cols,
    };
    let mut group = tid() / 128;
    let lane = tid() % 128;
    while group < n_groups {
        let nc = n_cells.f(group);
        let corr = nc / (nc - dof as f64);
        let mut p = indptr.i(group) + lane;
        let end = indptr.i(group + 1).min(data.len);
        while p < end {
            let m = means.f(p);
            data.put(p, (data.f(p) - m * m) * corr);
            p += 128;
        }
        group += stride() / 128;
    }
}
