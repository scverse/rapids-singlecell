//! Native edistance CUDA kernels.
use super::*;

// Category, pair and column indices have an int32 binding contract. Avoid
// carrying a runtime dtype branch through every tile load; indptr remains
// independently dispatched because sparse offsets support int32 and int64.
#[inline(always)]
fn index32(buffer: Buffer, index: u64) -> u64 {
    if index >= buffer.len {
        return 0;
    }
    unsafe { *(buffer.pointer as *const i32).add(index as usize) as u64 }
}

// Expand the fixed tile in the frontend so each squared distance remains in a
// register. Dynamically indexed local arrays spill to device memory on CUDA.
macro_rules! tile_columns {
    ($column:ident, $body:block) => {
        tile_columns!(@each $column, $body; 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63);
    };
    (@each $column:ident, $body:block; $($index:literal),*) => {
        $({ let $column = $index; $body })*
    };
}

macro_rules! sparse_tile {
    ($value:ty, $cells:literal, $features:literal, $shuffle:ident, $indptr:ident, $indices:ident, $data:ident, $cat_offsets:ident, $cell_indices:ident, $pair_left:ident, $pair_right:ident, $pairwise_sums:ident, $num_pairs:ident, $blocks_per_pair:ident, $n_features:ident) => {{
        static mut CACHE: cuda_device::SharedArray<u64, 1024> = cuda_device::SharedArray::UNINIT;
        static mut PARTIAL: cuda_device::SharedArray<f64, 32> = cuda_device::SharedArray::UNINIT;
        let cache = unsafe { core::ptr::addr_of_mut!(CACHE[0]) as *mut $value };
        let block = thread::blockDim_x() as u64;
        let lane = thread::threadIdx_x() as u64;
        let mut tile = tid() / block;
        while tile < $num_pairs * $blocks_per_pair {
            // Preserve CUDA's (pair, partition) grid order. Interleaving
            // pairs spreads useful partitions across SMs when groups are
            // smaller than blocks_per_pair * block_size.
            let pair = tile % $num_pairs;
            let part = tile / $num_pairs;
            let l = index32($pair_left, pair);
            let r = index32($pair_right, pair);
            let (mut ls, mut le) = (index32($cat_offsets, l), index32($cat_offsets, l + 1));
            let (mut rs, mut re) = (index32($cat_offsets, r), index32($cat_offsets, r + 1));
            // The larger group spans threads; the smaller group remains in
            // the shared/L2 tile across the larger group's iterations.
            if le.saturating_sub(ls) < re.saturating_sub(rs) {
                core::mem::swap(&mut ls, &mut rs);
                core::mem::swap(&mut le, &mut re);
            }
            let na = le.saturating_sub(ls);
            let nb = re.saturating_sub(rs);
            let mut sum = 0.0 as $value;
            let mut abase = part * block;
            while abase < na {
                let a_local = abase + lane;
                let a = index32($cell_indices, ls + a_local);
                let mut bbase = 0;
                while bbase < nb {
                    let width = (nb - bbase).min($cells);
                    let mut squares = [0.0 as $value; $cells as usize];
                    let mut a_cur = $indptr.i(a);
                    let a_end = $indptr.i(a + 1).min($indices.len).min($data.len);
                    let b_row = index32($cell_indices, rs + bbase + lane);
                    let mut b_cur = if lane < width { $indptr.i(b_row) } else { 0 };
                    let b_end = if lane < width {
                        $indptr.i(b_row + 1).min($indices.len).min($data.len)
                    } else {
                        0
                    };
                    let mut fbase = 0;
                    while fbase < $n_features {
                        let nf = ($n_features - fbase).min($features);
                        let mut q = lane;
                        while q < $cells * $features {
                            unsafe {
                                *cache.add(q as usize) = 0.0;
                            }
                            q += block;
                        }
                        thread::sync_threads();
                        if lane < width {
                            while b_cur < b_end {
                                let column = index32($indices, b_cur);
                                if column >= fbase + nf {
                                    break;
                                }
                                if column >= fbase {
                                    unsafe {
                                        *cache.add(((column - fbase) * $cells + lane) as usize) =
                                            *($data.pointer as *const $value).add(b_cur as usize);
                                    }
                                }
                                b_cur += 1;
                            }
                        }
                        let mut a_window = [0.0 as $value; $features as usize];
                        if a_local < na {
                            while a_cur < a_end {
                                let column = index32($indices, a_cur);
                                if column >= fbase + nf {
                                    break;
                                }
                                if column >= fbase {
                                    a_window[(column - fbase) as usize] = unsafe {
                                        *($data.pointer as *const $value).add(a_cur as usize)
                                    };
                                }
                                a_cur += 1;
                            }
                        }
                        thread::sync_threads();
                        if a_local < na {
                            let mut f = 0;
                            while f < nf {
                                let av = a_window[f as usize];
                                tile_columns!(c, {
                                    if c < $cells {
                                        let bv = unsafe { *cache.add((f * $cells) as usize + c) };
                                        let diff = av - bv;
                                        squares[c] += diff * diff;
                                    }
                                });
                                f += 1;
                            }
                        }
                        thread::sync_threads();
                        fbase += $features;
                    }
                    if a_local < na {
                        tile_columns!(c, {
                            if c < $cells
                                && (c as u64) < width
                                && (l != r || a_local < bbase + c as u64)
                            {
                                sum += squares[c].sqrt();
                            }
                        });
                    }
                    bbase += $cells;
                }
                abase += $blocks_per_pair * block;
            }
            let mut d = 16;
            while d > 0 {
                sum += warp::$shuffle(u32::MAX, sum, d);
                d /= 2;
            }
            if lane.is_multiple_of(32) {
                unsafe {
                    PARTIAL[(lane / 32) as usize] = sum as f64;
                }
            }
            thread::sync_threads();
            if lane < 32 {
                let mut total = if lane < block / 32 {
                    unsafe { PARTIAL[lane as usize] as $value }
                } else {
                    0.0 as $value
                };
                let mut d = 16;
                while d > 0 {
                    total += warp::$shuffle(u32::MAX, total, d);
                    d /= 2;
                }
                if lane == 0 {
                    $pairwise_sums.add(pair, total as f64);
                }
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }};
}
macro_rules! energy_sparse_entry {
    ($name:ident, $value:ty, $cells:literal, $features:literal, $shuffle:ident, $max_threads:literal) => {
        /// # Safety
        /// The host validates all direct allocations and launch dimensions.
        /// Indirect sparse/category indices are checked before dereferencing.
        #[kernel]
        #[cuda_device::launch_bounds($max_threads)]
        pub unsafe fn $name(
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

            sparse_tile!(
                $value,
                $cells,
                $features,
                $shuffle,
                indptr,
                indices,
                data,
                cat_offsets,
                cell_indices,
                pair_left,
                pair_right,
                pairwise_sums,
                num_pairs,
                blocks_per_pair,
                n_features
            );
        }
    };
}

energy_sparse_entry!(
    domain_edistance_sparse_f32_c64_f25,
    f32,
    64,
    25,
    shuffle_down_f32_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f32_c64_f16,
    f32,
    64,
    16,
    shuffle_down_f32_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f64,
    f32,
    32,
    64,
    shuffle_down_f32_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f50,
    f32,
    32,
    50,
    shuffle_down_f32_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f32,
    f32,
    32,
    32,
    shuffle_down_f32_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f64,
    f64,
    16,
    64,
    shuffle_down_f64_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f50,
    f64,
    16,
    50,
    shuffle_down_f64_sync,
    512
);

energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f32,
    f64,
    16,
    32,
    shuffle_down_f64_sync,
    512
);

macro_rules! dense_tile {
    ($value:ty, $cells:literal, $features:literal, $shuffle:ident, $embedding:ident, $cat_offsets:ident, $cell_indices:ident, $pair_left:ident, $pair_right:ident, $pairwise_sums:ident, $num_pairs:ident, $blocks_per_pair:ident, $n_features:ident) => {{
        static mut CACHE: cuda_device::SharedArray<u64, 1024> = cuda_device::SharedArray::UNINIT;
        static mut PARTIAL: cuda_device::SharedArray<f64, 32> = cuda_device::SharedArray::UNINIT;
        let cache = unsafe { core::ptr::addr_of_mut!(CACHE[0]) as *mut $value };
        let block = thread::blockDim_x() as u64;
        let lane = thread::threadIdx_x() as u64;
        let mut tile = tid() / block;
        while tile < $num_pairs * $blocks_per_pair {
            // Preserve CUDA's (pair, partition) grid order. Interleaving
            // pairs spreads useful partitions across SMs when groups are
            // smaller than blocks_per_pair * block_size.
            let pair = tile % $num_pairs;
            let part = tile / $num_pairs;
            let l = index32($pair_left, pair);
            let r = index32($pair_right, pair);
            let (mut ls, mut le) = (index32($cat_offsets, l), index32($cat_offsets, l + 1));
            let (mut rs, mut re) = (index32($cat_offsets, r), index32($cat_offsets, r + 1));
            // The larger group spans threads; the smaller group remains in
            // the shared/L2 tile across the larger group's iterations.
            if le.saturating_sub(ls) < re.saturating_sub(rs) {
                core::mem::swap(&mut ls, &mut rs);
                core::mem::swap(&mut le, &mut re);
            }
            let na = le.saturating_sub(ls);
            let nb = re.saturating_sub(rs);
            let mut sum = 0.0 as $value;
            let mut abase = part * block;
            while abase < na {
                let a_local = abase + lane;
                let a = index32($cell_indices, ls + a_local);
                let mut bbase = 0;
                while bbase < nb {
                    let width = (nb - bbase).min($cells);
                    let mut squares = [0.0 as $value; $cells as usize];
                    let mut fbase = 0;
                    while fbase < $n_features {
                        let nf = ($n_features - fbase).min($features);
                        let mut q = lane;
                        while q < $cells * $features {
                            let bc = q / $features;
                            let bf = q % $features;
                            let mut value = 0.0 as $value;
                            if bc < width && bf < nf {
                                let br = index32($cell_indices, rs + bbase + bc);
                                // Check the row before multiplication to
                                // exclude invalid/negative sparse metadata.
                                if $n_features != 0 && br < $embedding.len / $n_features {
                                    value = unsafe {
                                        *($embedding.pointer as *const $value)
                                            .add((br * $n_features + fbase + bf) as usize)
                                    };
                                }
                            }
                            unsafe {
                                *cache.add((bf * $cells + bc) as usize) = value;
                            }
                            q += block;
                        }
                        thread::sync_threads();
                        if a_local < na && $n_features != 0 && a < $embedding.len / $n_features {
                            let mut f = 0;
                            while f < nf {
                                let av = unsafe {
                                    *($embedding.pointer as *const $value)
                                        .add((a * $n_features + fbase + f) as usize)
                                };
                                tile_columns!(c, {
                                    if c < $cells {
                                        let bv = unsafe { *cache.add((f * $cells) as usize + c) };
                                        let diff = av - bv;
                                        squares[c] += diff * diff;
                                    }
                                });
                                f += 1;
                            }
                        }
                        thread::sync_threads();
                        fbase += $features;
                    }
                    if a_local < na {
                        tile_columns!(c, {
                            if c < $cells
                                && (c as u64) < width
                                && (l != r || a_local < bbase + c as u64)
                            {
                                sum += squares[c].sqrt();
                            }
                        });
                    }
                    bbase += $cells;
                }
                abase += $blocks_per_pair * block;
            }
            let mut d = 16;
            while d > 0 {
                sum += warp::$shuffle(u32::MAX, sum, d);
                d /= 2;
            }
            if lane.is_multiple_of(32) {
                unsafe {
                    PARTIAL[(lane / 32) as usize] = sum as f64;
                }
            }
            thread::sync_threads();
            if lane < 32 {
                let mut total = if lane < block / 32 {
                    unsafe { PARTIAL[lane as usize] as $value }
                } else {
                    0.0 as $value
                };
                let mut d = 16;
                while d > 0 {
                    total += warp::$shuffle(u32::MAX, total, d);
                    d /= 2;
                }
                if lane == 0 {
                    $pairwise_sums.add(pair, total as f64);
                }
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }};
}
macro_rules! energy_dense_entry {
    ($name:ident, $value:ty, $cells:literal, $features:literal, $shuffle:ident, $max_threads:literal) => {
        /// # Safety
        /// The host validates all direct allocations and launch dimensions.
        /// Indirect sparse/category indices are checked before dereferencing.
        #[kernel]
        #[cuda_device::launch_bounds($max_threads)]
        pub unsafe fn $name(
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

            dense_tile!(
                $value,
                $cells,
                $features,
                $shuffle,
                embedding,
                cat_offsets,
                cell_indices,
                pair_left,
                pair_right,
                pairwise_sums,
                num_pairs,
                blocks_per_pair,
                n_features
            );
        }
    };
}

energy_dense_entry!(
    domain_edistance_dense_f32_c64_f25,
    f32,
    64,
    25,
    shuffle_down_f32_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f32_c64_f16,
    f32,
    64,
    16,
    shuffle_down_f32_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f32_c32_f64,
    f32,
    32,
    64,
    shuffle_down_f32_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f32_c32_f50,
    f32,
    32,
    50,
    shuffle_down_f32_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f32_c32_f32,
    f32,
    32,
    32,
    shuffle_down_f32_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f64_c16_f64,
    f64,
    16,
    64,
    shuffle_down_f64_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f64_c16_f50,
    f64,
    16,
    50,
    shuffle_down_f64_sync,
    512
);

energy_dense_entry!(
    domain_edistance_dense_f64_c16_f32,
    f64,
    16,
    32,
    shuffle_down_f64_sync,
    512
);

// Keep register budgets for large explicit launch sizes separate from the
// default 512-thread specializations. This preserves accepted launch choices
// without forcing their spill budget onto ordinary calls.
energy_sparse_entry!(
    domain_edistance_sparse_f32_c64_f25_b1024,
    f32,
    64,
    25,
    shuffle_down_f32_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f32_c64_f16_b1024,
    f32,
    64,
    16,
    shuffle_down_f32_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f64_b1024,
    f32,
    32,
    64,
    shuffle_down_f32_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f50_b1024,
    f32,
    32,
    50,
    shuffle_down_f32_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f32_c32_f32_b1024,
    f32,
    32,
    32,
    shuffle_down_f32_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f64_b1024,
    f64,
    16,
    64,
    shuffle_down_f64_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f50_b1024,
    f64,
    16,
    50,
    shuffle_down_f64_sync,
    1024
);
energy_sparse_entry!(
    domain_edistance_sparse_f64_c16_f32_b1024,
    f64,
    16,
    32,
    shuffle_down_f64_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f32_c64_f25_b1024,
    f32,
    64,
    25,
    shuffle_down_f32_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f32_c64_f16_b1024,
    f32,
    64,
    16,
    shuffle_down_f32_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f32_c32_f64_b1024,
    f32,
    32,
    64,
    shuffle_down_f32_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f32_c32_f50_b1024,
    f32,
    32,
    50,
    shuffle_down_f32_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f32_c32_f32_b1024,
    f32,
    32,
    32,
    shuffle_down_f32_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f64_c16_f64_b1024,
    f64,
    16,
    64,
    shuffle_down_f64_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f64_c16_f50_b1024,
    f64,
    16,
    50,
    shuffle_down_f64_sync,
    1024
);
energy_dense_entry!(
    domain_edistance_dense_f64_c16_f32_b1024,
    f64,
    16,
    32,
    shuffle_down_f64_sync,
    1024
);
