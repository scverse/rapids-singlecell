//! Native edistance CUDA kernels.
use super::*;

// Expand the fixed tile in the frontend so each squared distance remains in a
// register. Dynamically indexed local arrays spill to device memory on CUDA.
macro_rules! tile_columns {
    ($column:ident, $body:block) => {
        tile_columns!(@each $column, $body; 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31);
    };
    (@each $column:ident, $body:block; $($index:literal),*) => {
        $({ let $column = $index; $body })*
    };
}

#[inline(always)]
fn sparse_distance_f32(
    indptr: Buffer,
    index: Buffer,
    data: Buffer,
    a: u64,
    b: u64,
    features: u64,
) -> f64 {
    let mut p = indptr.i(a);
    let pe = indptr.i(a + 1).min(data.len).min(index.len);
    let mut q = indptr.i(b);
    let qe = indptr.i(b + 1).min(data.len).min(index.len);
    let mut square = 0.0f32;
    while p < pe || q < qe {
        let pi = if p < pe { index.i(p) } else { u64::MAX };
        let qi = if q < qe { index.i(q) } else { u64::MAX };
        let (gene, diff) = if pi == qi {
            let d = data.f(p) as f32 - data.f(q) as f32;
            p += 1;
            q += 1;
            (pi, d)
        } else if pi < qi {
            let d = data.f(p) as f32;
            p += 1;
            (pi, d)
        } else {
            let d = -(data.f(q) as f32);
            q += 1;
            (qi, d)
        };
        if gene < features {
            square += diff * diff;
        }
    }
    square.sqrt() as f64
}
#[inline(always)]
fn sparse_distance_f64(
    indptr: Buffer,
    index: Buffer,
    data: Buffer,
    a: u64,
    b: u64,
    features: u64,
) -> f64 {
    let mut p = indptr.i(a);
    let pe = indptr.i(a + 1).min(data.len).min(index.len);
    let mut q = indptr.i(b);
    let qe = indptr.i(b + 1).min(data.len).min(index.len);
    let mut square = 0.0f64;
    while p < pe || q < qe {
        let pi = if p < pe { index.i(p) } else { u64::MAX };
        let qi = if q < qe { index.i(q) } else { u64::MAX };
        let (gene, diff) = if pi == qi {
            let d = data.f(p) - data.f(q);
            p += 1;
            q += 1;
            (pi, d)
        } else if pi < qi {
            let d = data.f(p);
            p += 1;
            (pi, d)
        } else {
            let d = -(data.f(q));
            q += 1;
            (qi, d)
        };
        if gene < features {
            square += diff * diff;
        }
    }
    square.sqrt()
}

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_edistance_compute_distances_sparse(
    indptr: u64,
    indices: u64,
    data: u64,
    cat_offsets: u64,
    cell_indices: u64,
    pair_left: u64,
    pair_right: u64,
    pairwise_sums: u64,
    num_pairs: u64,
    n_features: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    feat_tile: u64,
    block_size: u64,
    shared_mem: u64,
    workspace: u64,
    phase: u64,
    indptr_len: u64,
    indptr_kind: u64,
    indptr_order: u64,
    indptr_rows: u64,
    indptr_cols: u64,
    indices_len: u64,
    indices_kind: u64,
    indices_order: u64,
    indices_rows: u64,
    indices_cols: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    cat_offsets_len: u64,
    cat_offsets_kind: u64,
    cat_offsets_order: u64,
    cat_offsets_rows: u64,
    cat_offsets_cols: u64,
    cell_indices_len: u64,
    cell_indices_kind: u64,
    cell_indices_order: u64,
    cell_indices_rows: u64,
    cell_indices_cols: u64,
    pair_left_len: u64,
    pair_left_kind: u64,
    pair_left_order: u64,
    pair_left_rows: u64,
    pair_left_cols: u64,
    pair_right_len: u64,
    pair_right_kind: u64,
    pair_right_order: u64,
    pair_right_rows: u64,
    pair_right_cols: u64,
    pairwise_sums_len: u64,
    pairwise_sums_kind: u64,
    pairwise_sums_order: u64,
    pairwise_sums_rows: u64,
    pairwise_sums_cols: u64,
    workspace_len: u64,
    workspace_kind: u64,
    workspace_order: u64,
    workspace_rows: u64,
    workspace_cols: u64,
) {
    let indptr = Buffer {
        pointer: indptr,
        len: indptr_len,
        kind: indptr_kind,
        order: indptr_order,
        rows: indptr_rows,
        cols: indptr_cols,
    };
    let indices = Buffer {
        pointer: indices,
        len: indices_len,
        kind: indices_kind,
        order: indices_order,
        rows: indices_rows,
        cols: indices_cols,
    };
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let cat_offsets = Buffer {
        pointer: cat_offsets,
        len: cat_offsets_len,
        kind: cat_offsets_kind,
        order: cat_offsets_order,
        rows: cat_offsets_rows,
        cols: cat_offsets_cols,
    };
    let cell_indices = Buffer {
        pointer: cell_indices,
        len: cell_indices_len,
        kind: cell_indices_kind,
        order: cell_indices_order,
        rows: cell_indices_rows,
        cols: cell_indices_cols,
    };
    let pair_left = Buffer {
        pointer: pair_left,
        len: pair_left_len,
        kind: pair_left_kind,
        order: pair_left_order,
        rows: pair_left_rows,
        cols: pair_left_cols,
    };
    let pair_right = Buffer {
        pointer: pair_right,
        len: pair_right_len,
        kind: pair_right_kind,
        order: pair_right_order,
        rows: pair_right_rows,
        cols: pair_right_cols,
    };
    let pairwise_sums = Buffer {
        pointer: pairwise_sums,
        len: pairwise_sums_len,
        kind: pairwise_sums_kind,
        order: pairwise_sums_order,
        rows: pairwise_sums_rows,
        cols: pairwise_sums_cols,
    };
    let workspace = Buffer {
        pointer: workspace,
        len: workspace_len,
        kind: workspace_kind,
        order: workspace_order,
        rows: workspace_rows,
        cols: workspace_cols,
    };
    static mut PARTIAL: cuda_device::SharedArray<f64, 4> = cuda_device::SharedArray::UNINIT;
    if phase == 0 {
        let mut tile = tid() / 128;
        let lane = tid() % 128;
        while tile < num_pairs * blocks_per_pair {
            let pair = tile / blocks_per_pair;
            let part = tile % blocks_per_pair;
            let l = pair_left.i(pair);
            let r = pair_right.i(pair);
            let ls = cat_offsets.i(l);
            let le = cat_offsets.i(l + 1);
            let rs = cat_offsets.i(r);
            let re = cat_offsets.i(r + 1);
            let nr = re.saturating_sub(rs);
            let mut p = part * 128 + lane;
            let count = le.saturating_sub(ls) * nr;
            let mut sum = 0.0;
            while p < count {
                let a = cell_indices.i(ls + p / nr);
                let b = cell_indices.i(rs + p % nr);
                if l != r || p / nr < p % nr {
                    sum += if data.kind == 0 {
                        sparse_distance_f32(indptr, indices, data, a, b, n_features)
                    } else {
                        sparse_distance_f64(indptr, indices, data, a, b, n_features)
                    };
                }
                p += blocks_per_pair * 128;
            }
            let sum = sum_warp(sum);
            if lane.is_multiple_of(32) {
                unsafe {
                    PARTIAL[(lane / 32) as usize] = sum;
                }
            }
            thread::sync_threads();
            if lane == 0 {
                let total = unsafe { PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3] };
                workspace.put(tile, total);
            }
            thread::sync_threads();
            tile += stride() / 128;
        }
    } else {
        let mut pair = tid() / 32;
        let lane = tid() % 32;
        while pair < num_pairs {
            let mut q = lane;
            let mut total = 0.0;
            while q < blocks_per_pair {
                total += workspace.f(pair * blocks_per_pair + q);
                q += 32;
            }
            total = sum_warp(total);
            if lane == 0 {
                pairwise_sums.put(pair, pairwise_sums.f(pair) + total);
            }
            pair += stride() / 32;
        }
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_edistance_compute_distances(
    embedding: u64,
    cat_offsets: u64,
    cell_indices: u64,
    pair_left: u64,
    pair_right: u64,
    pairwise_sums: u64,
    num_pairs: u64,
    n_features: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    feat_tile: u64,
    block_size: u64,
    shared_mem: u64,
    workspace: u64,
    phase: u64,
    embedding_len: u64,
    embedding_kind: u64,
    embedding_order: u64,
    embedding_rows: u64,
    embedding_cols: u64,
    cat_offsets_len: u64,
    cat_offsets_kind: u64,
    cat_offsets_order: u64,
    cat_offsets_rows: u64,
    cat_offsets_cols: u64,
    cell_indices_len: u64,
    cell_indices_kind: u64,
    cell_indices_order: u64,
    cell_indices_rows: u64,
    cell_indices_cols: u64,
    pair_left_len: u64,
    pair_left_kind: u64,
    pair_left_order: u64,
    pair_left_rows: u64,
    pair_left_cols: u64,
    pair_right_len: u64,
    pair_right_kind: u64,
    pair_right_order: u64,
    pair_right_rows: u64,
    pair_right_cols: u64,
    pairwise_sums_len: u64,
    pairwise_sums_kind: u64,
    pairwise_sums_order: u64,
    pairwise_sums_rows: u64,
    pairwise_sums_cols: u64,
    workspace_len: u64,
    workspace_kind: u64,
    workspace_order: u64,
    workspace_rows: u64,
    workspace_cols: u64,
) {
    let embedding = Buffer {
        pointer: embedding,
        len: embedding_len,
        kind: embedding_kind,
        order: embedding_order,
        rows: embedding_rows,
        cols: embedding_cols,
    };
    let cat_offsets = Buffer {
        pointer: cat_offsets,
        len: cat_offsets_len,
        kind: cat_offsets_kind,
        order: cat_offsets_order,
        rows: cat_offsets_rows,
        cols: cat_offsets_cols,
    };
    let cell_indices = Buffer {
        pointer: cell_indices,
        len: cell_indices_len,
        kind: cell_indices_kind,
        order: cell_indices_order,
        rows: cell_indices_rows,
        cols: cell_indices_cols,
    };
    let pair_left = Buffer {
        pointer: pair_left,
        len: pair_left_len,
        kind: pair_left_kind,
        order: pair_left_order,
        rows: pair_left_rows,
        cols: pair_left_cols,
    };
    let pair_right = Buffer {
        pointer: pair_right,
        len: pair_right_len,
        kind: pair_right_kind,
        order: pair_right_order,
        rows: pair_right_rows,
        cols: pair_right_cols,
    };
    let pairwise_sums = Buffer {
        pointer: pairwise_sums,
        len: pairwise_sums_len,
        kind: pairwise_sums_kind,
        order: pairwise_sums_order,
        rows: pairwise_sums_rows,
        cols: pairwise_sums_cols,
    };
    let workspace = Buffer {
        pointer: workspace,
        len: workspace_len,
        kind: workspace_kind,
        order: workspace_order,
        rows: workspace_rows,
        cols: workspace_cols,
    };
    static mut CACHE_F32: cuda_device::SharedArray<f32, 1024> = cuda_device::SharedArray::UNINIT;
    static mut CACHE_F64: cuda_device::SharedArray<f64, 1024> = cuda_device::SharedArray::UNINIT;
    static mut PARTIAL: cuda_device::SharedArray<f64, 4> = cuda_device::SharedArray::UNINIT;
    if phase == 0 {
        let mut tile = tid() / 128;
        let lane = tid() % 128;
        while tile < num_pairs * blocks_per_pair {
            let pair = tile / blocks_per_pair;
            let part = tile % blocks_per_pair;
            let l = pair_left.i(pair);
            let r = pair_right.i(pair);
            let ls = cat_offsets.i(l);
            let le = cat_offsets.i(l + 1);
            let rs = cat_offsets.i(r);
            let re = cat_offsets.i(r + 1);
            let nr = re.saturating_sub(rs);
            let na = le.saturating_sub(ls);
            let mut sum = 0.0;
            let mut abase = part * 128;
            while abase < na {
                let a_local = abase + lane;
                let a = cell_indices.i(ls + a_local);
                if embedding.kind == 0 {
                    let mut bbase = 0;
                    while bbase < nr {
                        let width = (nr - bbase).min(32);
                        let mut squares = [0.0f32; 32];
                        let mut fbase = 0;
                        while fbase < n_features {
                            let nf = (n_features - fbase).min(32);
                            let mut q = lane;
                            while q < 1024 {
                                let bc = q / 32;
                                let bf = q % 32;
                                let mut value = 0.0f32;
                                if bc < width && bf < nf {
                                    let br = cell_indices.i(rs + bbase + bc);
                                    let at = br * n_features + fbase + bf;
                                    if at < embedding.len {
                                        value = unsafe {
                                            *(embedding.pointer as *const f32).add(at as usize)
                                        };
                                    }
                                }
                                unsafe {
                                    CACHE_F32[(bf * 32 + bc) as usize] = value;
                                }
                                q += 128;
                            }
                            thread::sync_threads();
                            if a_local < na {
                                let mut f = 0;
                                while f < nf {
                                    let at = a * n_features + fbase + f;
                                    let av = if at < embedding.len {
                                        unsafe {
                                            *(embedding.pointer as *const f32).add(at as usize)
                                        }
                                    } else {
                                        0.0
                                    };
                                    tile_columns!(c, {
                                        let bv = unsafe { CACHE_F32[(f * 32) as usize + c] };
                                        let diff = av - bv;
                                        squares[c] += diff * diff;
                                    });
                                    f += 1;
                                }
                            }
                            thread::sync_threads();
                            fbase += 32;
                        }
                        if a_local < na {
                            tile_columns!(c, {
                                if (c as u64) < width && (l != r || a_local < bbase + c as u64) {
                                    sum += squares[c].sqrt() as f64;
                                }
                            });
                        }
                        bbase += 32;
                    }
                } else {
                    let mut bbase = 0;
                    while bbase < nr {
                        let width = (nr - bbase).min(32);
                        let mut squares = [0.0f64; 32];
                        let mut fbase = 0;
                        while fbase < n_features {
                            let nf = (n_features - fbase).min(32);
                            let mut q = lane;
                            while q < 1024 {
                                let bc = q / 32;
                                let bf = q % 32;
                                let mut value = 0.0f64;
                                if bc < width && bf < nf {
                                    let br = cell_indices.i(rs + bbase + bc);
                                    let at = br * n_features + fbase + bf;
                                    if at < embedding.len {
                                        value = unsafe {
                                            *(embedding.pointer as *const f64).add(at as usize)
                                        };
                                    }
                                }
                                unsafe {
                                    CACHE_F64[(bf * 32 + bc) as usize] = value;
                                }
                                q += 128;
                            }
                            thread::sync_threads();
                            if a_local < na {
                                let mut f = 0;
                                while f < nf {
                                    let at = a * n_features + fbase + f;
                                    let av = if at < embedding.len {
                                        unsafe {
                                            *(embedding.pointer as *const f64).add(at as usize)
                                        }
                                    } else {
                                        0.0
                                    };
                                    tile_columns!(c, {
                                        let bv = unsafe { CACHE_F64[(f * 32) as usize + c] };
                                        let diff = av - bv;
                                        squares[c] += diff * diff;
                                    });
                                    f += 1;
                                }
                            }
                            thread::sync_threads();
                            fbase += 32;
                        }
                        if a_local < na {
                            tile_columns!(c, {
                                if (c as u64) < width && (l != r || a_local < bbase + c as u64) {
                                    sum += squares[c].sqrt();
                                }
                            });
                        }
                        bbase += 32;
                    }
                }
                abase += blocks_per_pair * 128;
            }
            let sum = sum_warp(sum);
            if lane.is_multiple_of(32) {
                unsafe {
                    PARTIAL[(lane / 32) as usize] = sum;
                }
            }
            thread::sync_threads();
            if lane == 0 {
                let total = unsafe { PARTIAL[0] + PARTIAL[1] + PARTIAL[2] + PARTIAL[3] };
                workspace.put(tile, total);
            }
            thread::sync_threads();
            tile += stride() / 128;
        }
    } else {
        let mut pair = tid() / 32;
        let lane = tid() % 32;
        while pair < num_pairs {
            let mut q = lane;
            let mut total = 0.0;
            while q < blocks_per_pair {
                total += workspace.f(pair * blocks_per_pair + q);
                q += 32;
            }
            total = sum_warp(total);
            if lane == 0 {
                pairwise_sums.put(pair, pairwise_sums.f(pair) + total);
            }
            pair += stride() / 32;
        }
    }
}
