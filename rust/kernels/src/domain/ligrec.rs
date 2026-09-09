//! Native ligrec CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_ligrec_sum_count_dense(
    data: u64,
    clusters: u64,
    sum: u64,
    count: u64,
    rows: u64,
    cols: u64,
    ncls: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    clusters_len: u64,
    clusters_kind: u64,
    clusters_order: u64,
    clusters_rows: u64,
    clusters_cols: u64,
    sum_len: u64,
    sum_kind: u64,
    sum_order: u64,
    sum_rows: u64,
    sum_cols: u64,
    count_len: u64,
    count_kind: u64,
    count_order: u64,
    count_rows: u64,
    count_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let clusters = Buffer {
        pointer: clusters,
        len: clusters_len,
        kind: clusters_kind,
        order: clusters_order,
        rows: clusters_rows,
        cols: clusters_cols,
    };
    let sum = Buffer {
        pointer: sum,
        len: sum_len,
        kind: sum_kind,
        order: sum_order,
        rows: sum_rows,
        cols: sum_cols,
    };
    let count = Buffer {
        pointer: count,
        len: count_len,
        kind: count_kind,
        order: count_order,
        rows: count_rows,
        cols: count_cols,
    };
    let row_tiles = rows.div_ceil(32);
    let col_tiles = cols.div_ceil(32);
    let mut tile = tid() / 1024;
    let lane = tid() % 1024;
    while tile < row_tiles * col_tiles {
        let row = (tile % row_tiles) * 32 + lane % 32;
        let gene = (tile / row_tiles) * 32 + lane / 32;
        let cl = clusters.i(row);
        let v = data.f(row * cols + gene);
        if row < rows && gene < cols && cl < ncls && v > 0.0 {
            let o = gene * ncls + cl;
            sum.add(o, v);
            count.add(o, 1.0);
        }
        tile += stride() / 1024;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_ligrec_sum_count_sparse(
    indptr: u64,
    index: u64,
    data: u64,
    clusters: u64,
    sum: u64,
    count: u64,
    rows: u64,
    ncls: u64,
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
    clusters_len: u64,
    clusters_kind: u64,
    clusters_order: u64,
    clusters_rows: u64,
    clusters_cols: u64,
    sum_len: u64,
    sum_kind: u64,
    sum_order: u64,
    sum_rows: u64,
    sum_cols: u64,
    count_len: u64,
    count_kind: u64,
    count_order: u64,
    count_rows: u64,
    count_cols: u64,
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
    let clusters = Buffer {
        pointer: clusters,
        len: clusters_len,
        kind: clusters_kind,
        order: clusters_order,
        rows: clusters_rows,
        cols: clusters_cols,
    };
    let sum = Buffer {
        pointer: sum,
        len: sum_len,
        kind: sum_kind,
        order: sum_order,
        rows: sum_rows,
        cols: sum_cols,
    };
    let count = Buffer {
        pointer: count,
        len: count_len,
        kind: count_kind,
        order: count_order,
        rows: count_rows,
        cols: count_cols,
    };
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < rows {
        let cl = clusters.i(row);
        let mut p = indptr.i(row) + lane;
        let end = indptr.i(row + 1).min(index.len).min(data.len);
        while p < end {
            let gene = index.i(p);
            let v = data.f(p);
            if cl < ncls && v > 0.0 {
                let o = gene * ncls + cl;
                sum.add(o, v);
                count.add(o, 1.0);
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
pub unsafe fn domain_ligrec_mean_dense(
    data: u64,
    clusters: u64,
    g: u64,
    rows: u64,
    cols: u64,
    ncls: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    clusters_len: u64,
    clusters_kind: u64,
    clusters_order: u64,
    clusters_rows: u64,
    clusters_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let clusters = Buffer {
        pointer: clusters,
        len: clusters_len,
        kind: clusters_kind,
        order: clusters_order,
        rows: clusters_rows,
        cols: clusters_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let row_tiles = rows.div_ceil(32);
    let col_tiles = cols.div_ceil(32);
    let mut tile = tid() / 1024;
    let lane = tid() % 1024;
    while tile < row_tiles * col_tiles {
        let row = (tile % row_tiles) * 32 + lane % 32;
        let gene = (tile / row_tiles) * 32 + lane / 32;
        let cl = clusters.i(row);
        let v = data.f(row * cols + gene);
        if row < rows && gene < cols && cl < ncls {
            let o = gene * ncls + cl;
            g.add(o, v);
        }
        tile += stride() / 1024;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_ligrec_mean_sparse(
    indptr: u64,
    index: u64,
    data: u64,
    clusters: u64,
    g: u64,
    rows: u64,
    ncls: u64,
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
    clusters_len: u64,
    clusters_kind: u64,
    clusters_order: u64,
    clusters_rows: u64,
    clusters_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
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
    let clusters = Buffer {
        pointer: clusters,
        len: clusters_len,
        kind: clusters_kind,
        order: clusters_order,
        rows: clusters_rows,
        cols: clusters_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < rows {
        let cl = clusters.i(row);
        let mut p = indptr.i(row) + lane;
        let end = indptr.i(row + 1).min(index.len).min(data.len);
        while p < end {
            let gene = index.i(p);
            let v = data.f(p);
            if cl < ncls && v > 0.0 {
                let o = gene * ncls + cl;
                g.add(o, v);
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
pub unsafe fn domain_ligrec_elementwise_diff(
    g: u64,
    total_counts: u64,
    n_genes: u64,
    n_clusters: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
    total_counts_len: u64,
    total_counts_kind: u64,
    total_counts_order: u64,
    total_counts_rows: u64,
    total_counts_cols: u64,
) {
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let total_counts = Buffer {
        pointer: total_counts,
        len: total_counts_len,
        kind: total_counts_kind,
        order: total_counts_order,
        rows: total_counts_rows,
        cols: total_counts_cols,
    };
    let mut p = tid();
    while p < n_genes * n_clusters {
        if g.kind == 0 {
            g.put_single(p, g.single(p) / total_counts.single(p % n_clusters));
        } else {
            g.put(p, g.f(p) / total_counts.f(p % n_clusters));
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_ligrec_interaction(
    interactions: u64,
    interaction_clusters: u64,
    mean: u64,
    res: u64,
    mask: u64,
    g: u64,
    n_iter: u64,
    n_inter_clust: u64,
    ncls: u64,
    interactions_len: u64,
    interactions_kind: u64,
    interactions_order: u64,
    interactions_rows: u64,
    interactions_cols: u64,
    interaction_clusters_len: u64,
    interaction_clusters_kind: u64,
    interaction_clusters_order: u64,
    interaction_clusters_rows: u64,
    interaction_clusters_cols: u64,
    mean_len: u64,
    mean_kind: u64,
    mean_order: u64,
    mean_rows: u64,
    mean_cols: u64,
    res_len: u64,
    res_kind: u64,
    res_order: u64,
    res_rows: u64,
    res_cols: u64,
    mask_len: u64,
    mask_kind: u64,
    mask_order: u64,
    mask_rows: u64,
    mask_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
) {
    let interactions = Buffer {
        pointer: interactions,
        len: interactions_len,
        kind: interactions_kind,
        order: interactions_order,
        rows: interactions_rows,
        cols: interactions_cols,
    };
    let interaction_clusters = Buffer {
        pointer: interaction_clusters,
        len: interaction_clusters_len,
        kind: interaction_clusters_kind,
        order: interaction_clusters_order,
        rows: interaction_clusters_rows,
        cols: interaction_clusters_cols,
    };
    let mean = Buffer {
        pointer: mean,
        len: mean_len,
        kind: mean_kind,
        order: mean_order,
        rows: mean_rows,
        cols: mean_cols,
    };
    let res = Buffer {
        pointer: res,
        len: res_len,
        kind: res_kind,
        order: res_order,
        rows: res_rows,
        cols: res_cols,
    };
    let mask = Buffer {
        pointer: mask,
        len: mask_len,
        kind: mask_kind,
        order: mask_order,
        rows: mask_rows,
        cols: mask_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let mut p = tid();
    while p < n_iter * n_inter_clust {
        let i = p / n_inter_clust;
        let j = p % n_inter_clust;
        let a = interactions.i(i * 2) * ncls + interaction_clusters.i(j * 2);
        let b = interactions.i(i * 2 + 1) * ncls + interaction_clusters.i(j * 2 + 1);
        let ma = mean.f(a);
        let mb = mean.f(b);
        let old = res.f(p);
        if !old.is_nan() {
            if ma > 0.0 && mb > 0.0 && mask.i(a) != 0 && mask.i(b) != 0 {
                res.put(
                    p,
                    old + f64::from(if g.kind == 0 {
                        g.single(a) + g.single(b) > ma as f32 + mb as f32
                    } else {
                        g.f(a) + g.f(b) > ma + mb
                    }),
                );
            } else {
                res.put(p, f64::NAN);
            }
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_ligrec_res_mean(
    interactions: u64,
    interaction_clusters: u64,
    mean: u64,
    res_mean: u64,
    n_inter: u64,
    n_inter_clust: u64,
    ncls: u64,
    interactions_len: u64,
    interactions_kind: u64,
    interactions_order: u64,
    interactions_rows: u64,
    interactions_cols: u64,
    interaction_clusters_len: u64,
    interaction_clusters_kind: u64,
    interaction_clusters_order: u64,
    interaction_clusters_rows: u64,
    interaction_clusters_cols: u64,
    mean_len: u64,
    mean_kind: u64,
    mean_order: u64,
    mean_rows: u64,
    mean_cols: u64,
    res_mean_len: u64,
    res_mean_kind: u64,
    res_mean_order: u64,
    res_mean_rows: u64,
    res_mean_cols: u64,
) {
    let interactions = Buffer {
        pointer: interactions,
        len: interactions_len,
        kind: interactions_kind,
        order: interactions_order,
        rows: interactions_rows,
        cols: interactions_cols,
    };
    let interaction_clusters = Buffer {
        pointer: interaction_clusters,
        len: interaction_clusters_len,
        kind: interaction_clusters_kind,
        order: interaction_clusters_order,
        rows: interaction_clusters_rows,
        cols: interaction_clusters_cols,
    };
    let mean = Buffer {
        pointer: mean,
        len: mean_len,
        kind: mean_kind,
        order: mean_order,
        rows: mean_rows,
        cols: mean_cols,
    };
    let res_mean = Buffer {
        pointer: res_mean,
        len: res_mean_len,
        kind: res_mean_kind,
        order: res_mean_order,
        rows: res_mean_rows,
        cols: res_mean_cols,
    };
    let mut p = tid();
    while p < n_inter * n_inter_clust {
        let i = p / n_inter_clust;
        let j = p % n_inter_clust;
        let a = interactions.i(i * 2) * ncls + interaction_clusters.i(j * 2);
        let b = interactions.i(i * 2 + 1) * ncls + interaction_clusters.i(j * 2 + 1);
        let ma = mean.f(a);
        let mb = mean.f(b);
        if ma > 0.0 && mb > 0.0 {
            if mean.kind == 0 {
                res_mean.put_single(p, (ma as f32 + mb as f32) / 2.0);
            } else {
                res_mean.put(p, (ma + mb) / 2.0);
            }
        }
        p += stride();
    }
}
