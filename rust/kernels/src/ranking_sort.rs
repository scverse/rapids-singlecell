//! Stable segmented radix sorting for contiguous float32 or uint32 columns.
#![allow(clippy::too_many_arguments)]
use cuda_device::{SharedArray, kernel, thread, warp};

#[inline(always)]
fn ordered_key(value: f32) -> u32 {
    let bits = value.to_bits();
    // CUB floating-key transformation, branchless except for signed-zero
    // equivalence. NaN signs/payloads retain their deterministic bit ordering.
    if bits & 0x7fff_ffff == 0 {
        0x8000_0000
    } else {
        bits ^ (0u32.wrapping_sub(bits >> 31) | 0x8000_0000)
    }
}

#[inline(always)]
fn radix_digit(bits: u32, shift: u32) -> u32 {
    let key = if shift & 256 != 0 {
        bits
    } else {
        ordered_key(f32::from_bits(bits))
    };
    (key >> (shift & 31)) & 255
}

#[inline(always)]
unsafe fn segment_for_tile(tile_offsets: *const u64, count: u64, tile: u64) -> u64 {
    let mut lo = 0;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *tile_offsets.add(mid as usize + 1) } <= tile {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// Four register-held items per thread amortize histogram work and allow a
// shared permutation to turn random global scatter stores into ordered runs.
const TILE_ROWS: u64 = 1024;

#[inline(always)]
unsafe fn histogram_body<const SEGMENTED: bool, const VALUES: bool>(
    values: *const u32,
    histogram: *mut u32,
    permutation: *mut u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
    output: *mut u32,
) {
    unsafe {
        static mut COUNTS: SharedArray<u32, 8192> = SharedArray::UNINIT;
        static mut ORDER: SharedArray<u32, 1024> = SharedArray::UNINIT;
        let counts = SharedArray::as_raw_mut_ptr(&raw mut COUNTS);
        let order = SharedArray::as_raw_mut_ptr(&raw mut ORDER);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let wid = tid / 32;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < cols * tiles {
            for w in 0..32 {
                *counts.add(w * 256 + tid) = 0;
            }
            thread::sync_threads();
            let col = tile / tiles;
            let local_tile = tile % tiles;
            let (start, end) = if SEGMENTED {
                let segment = segment_for_tile(tile_offsets, segments, local_tile);
                (
                    *begins.add(segment as usize)
                        + (local_tile - *tile_offsets.add(segment as usize)) * TILE_ROWS,
                    *ends.add(segment as usize),
                )
            } else {
                (local_tile * TILE_ROWS, rows)
            };
            let mut digits = [256u32; 4];
            let mut keys = [0u32; 4];
            let mut ranks = [0u32; 4];
            for item in 0..4 {
                let row = start + item as u64 * 256 + tid as u64;
                let value = if row < end {
                    *values.add((col * rows + row) as usize)
                } else {
                    0
                };
                keys[item] = value;
                let digit = if row < end {
                    radix_digit(value, shift)
                } else {
                    256
                };
                digits[item] = digit;
                let peers = warp::match_any_sync(u32::MAX, digit);
                if digit < 256 && lane as u32 == peers.trailing_zeros() {
                    *counts.add((item * 8 + wid) * 256 + digit as usize) = peers.count_ones();
                }
                ranks[item] = (peers & warp::lanemask_lt()).count_ones();
            }
            thread::sync_threads();
            // One thread prefixes each digit's 32 warp/item counts. Each
            // value then needs a single shared load for its stable rank,
            // avoiding repeated, bank-conflicting prefix walks per value.
            let mut total = 0;
            for w in 0..32 {
                let count = *counts.add(w * 256 + tid);
                *counts.add(w * 256 + tid) = total;
                total += count;
            }
            thread::sync_threads();
            for item in 0..4 {
                if digits[item] < 256 {
                    ranks[item] += *counts.add((item * 8 + wid) * 256 + digits[item] as usize);
                }
            }
            *histogram.add((tile * 256 + tid as u64) as usize) = total;
            // Every thread has consumed the warp counts before they are reused
            // for digit prefixes; rank registers remain live across this scan.
            thread::sync_threads();
            let mut inclusive = total;
            let mut distance = 1;
            while distance < 32 {
                let previous = warp::shuffle_up(inclusive, distance);
                if lane >= distance as usize {
                    inclusive += previous;
                }
                distance *= 2;
            }
            if lane == 31 {
                *counts.add(wid) = inclusive;
            }
            thread::sync_threads();
            let mut prefix = inclusive - total;
            for w in 0..wid {
                prefix += *counts.add(w);
            }
            *counts.add(16 + tid) = prefix;
            if VALUES {
                *permutation.add((tile * 256 + tid as u64) as usize) = prefix;
            }
            thread::sync_threads();
            for item in 0..4 {
                if digits[item] < 256 {
                    let pos = *counts.add(16 + digits[item] as usize) + ranks[item];
                    *order.add(pos as usize) = if VALUES {
                        keys[item]
                    } else {
                        (item * 256 + tid) as u32 | (ranks[item] << 16)
                    };
                }
            }
            thread::sync_threads();
            for item in 0..4 {
                let local = item * 256 + tid;
                let row = start + local as u64;
                if row < end {
                    let destination = if VALUES { output } else { permutation };
                    *destination.add((col * rows + row) as usize) = *order.add(local);
                }
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Values/permutation contain rows*cols elements. Histogram contains
/// cols*ceil(rows/1024)*256 counts. Launch 256 threads per block.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_hist(
    values: *const u32,
    histogram: *mut u32,
    permutation: *mut u32,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        histogram_body::<false, false>(
            values,
            histogram,
            permutation,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            rows,
            cols,
            0,
            tiles,
            shift,
            core::ptr::null_mut(),
        );
    }
}

/// # Safety
/// Selected segments are disjoint and in range; tile_offsets contains their
/// cumulative ceil(length/1024) counts. Values/permutation contain rows*cols
/// elements, histogram contains cols*tiles*256. Launch 256 threads per block.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_segments_hist(
    values: *const u32,
    histogram: *mut u32,
    permutation: *mut u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        histogram_body::<true, false>(
            values,
            histogram,
            permutation,
            begins,
            ends,
            tile_offsets,
            rows,
            cols,
            segments,
            tiles,
            shift,
            core::ptr::null_mut(),
        );
    }
}

#[inline(always)]
unsafe fn scatter_body<const SEGMENTED: bool>(
    values: *const u32,
    indices: *const i64,
    output: *mut u32,
    output_indices: *mut i64,
    histogram: *const u32,
    permutation: *const u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        let tid = thread::threadIdx_x() as u64;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < cols * tiles {
            let col = tile / tiles;
            let local_tile = tile % tiles;
            let (start, end, segment_start) = if SEGMENTED {
                let segment = segment_for_tile(tile_offsets, segments, local_tile);
                let segment_start = *begins.add(segment as usize);
                (
                    segment_start + (local_tile - *tile_offsets.add(segment as usize)) * TILE_ROWS,
                    *ends.add(segment as usize),
                    segment_start,
                )
            } else {
                (local_tile * TILE_ROWS, rows, 0)
            };
            for item in 0..4 {
                let row = start + item * 256 + tid;
                if row < end {
                    let p = *permutation.add((col * rows + row) as usize);
                    let source_row = start + (p & 65535) as u64;
                    let source = col * rows + source_row;
                    let value = *values.add(source as usize);
                    let digit = radix_digit(value, shift);
                    let prefix = *histogram.add((tile * 256 + digit as u64) as usize);
                    let pos = col * rows + segment_start + prefix as u64 + (p >> 16) as u64;
                    *output.add(pos as usize) = value;
                    if !output_indices.is_null() {
                        if shift & 512 != 0 {
                            *output_indices.cast::<u32>().add(pos as usize) = if indices.is_null() {
                                source_row as u32
                            } else {
                                *indices.cast::<u32>().add(source as usize)
                            };
                        } else {
                            *output_indices.add(pos as usize) = if indices.is_null() {
                                source_row as i64
                            } else {
                                *indices.add(source as usize)
                            };
                        }
                    }
                }
            }
            tile += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Disjoint value/index buffers have rows*cols elements. Histogram and
/// permutation describe this radix pass, with ceil(rows/1024) tiles per column.
/// Index pointers may be null; shift bit 9 selects u32 indices, otherwise i64.
/// Launch 256 threads per block.
#[kernel]
pub unsafe fn rank_radix_scatter(
    values: *const u32,
    indices: *const i64,
    output: *mut u32,
    output_indices: *mut i64,
    histogram: *const u32,
    permutation: *const u32,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        scatter_body::<false>(
            values,
            indices,
            output,
            output_indices,
            histogram,
            permutation,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            rows,
            cols,
            0,
            tiles,
            shift,
        );
    }
}

/// # Safety
/// Active segments, histogram and permutation match the segmented histogram
/// pass. Disjoint value buffers contain rows*cols elements; block size is 256.
#[kernel]
pub unsafe fn rank_radix_segments_scatter(
    values: *const u32,
    output: *mut u32,
    histogram: *const u32,
    permutation: *const u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        scatter_body::<true>(
            values,
            core::ptr::null(),
            output,
            core::ptr::null_mut(),
            histogram,
            permutation,
            begins,
            ends,
            tile_offsets,
            rows,
            cols,
            segments,
            tiles,
            shift,
        );
    }
}

/// # Safety
/// Histogram contains cols*tiles*256 valid counts; block size is 256.
#[kernel]
pub unsafe fn rank_radix_prefix(histogram: *mut u32, cols: u64, tiles: u64) {
    unsafe {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let warps = SharedArray::as_raw_mut_ptr(&raw mut WARPS);
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let wid = tid / 32;
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            let mut total = 0u32;
            for tile in 0..tiles {
                total += *histogram.add(((col * tiles + tile) * 256 + tid as u64) as usize);
            }
            let mut inclusive = total;
            let mut d = 1;
            while d < 32 {
                let previous = warp::shuffle_up(inclusive, d);
                if lane >= d {
                    inclusive += previous;
                }
                d *= 2;
            }
            if lane == 31 {
                *warps.add(wid as usize) = inclusive;
            }
            thread::sync_threads();
            let mut prefix = inclusive - total;
            for w in 0..wid {
                prefix += *warps.add(w as usize);
            }
            for tile in 0..tiles {
                let ptr = histogram.add(((col * tiles + tile) * 256 + tid as u64) as usize);
                let count = *ptr;
                *ptr = prefix;
                prefix += count;
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Histogram and segment tile offsets satisfy the segmented histogram contract.
#[kernel]
pub unsafe fn rank_radix_segments_prefix(
    histogram: *mut u32,
    tile_offsets: *const u64,
    cols: u64,
    segments: u64,
    tiles: u64,
) {
    unsafe {
        static mut WARPS: SharedArray<u32, 8> = SharedArray::UNINIT;
        let warps = SharedArray::as_raw_mut_ptr(&raw mut WARPS);
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let wid = tid / 32;
        let mut seg = thread::blockIdx_x() as u64;
        while seg < cols * segments {
            let col = seg / segments;
            let group = seg % segments;
            let first = col * tiles + *tile_offsets.add(group as usize);
            let last = col * tiles + *tile_offsets.add(group as usize + 1);
            let mut total = 0u32;
            for tile in first..last {
                total += *histogram.add((tile * 256 + tid as u64) as usize);
            }
            let mut inclusive = total;
            let mut d = 1;
            while d < 32 {
                let previous = warp::shuffle_up(inclusive, d);
                if lane >= d {
                    inclusive += previous;
                }
                d *= 2;
            }
            if lane == 31 {
                *warps.add(wid as usize) = inclusive;
            }
            thread::sync_threads();
            let mut prefix = inclusive - total;
            for w in 0..wid {
                prefix += *warps.add(w as usize);
            }
            for tile in first..last {
                let ptr = histogram.add((tile * 256 + tid as u64) as usize);
                let count = *ptr;
                *ptr = prefix;
                prefix += count;
            }
            thread::sync_threads();
            seg += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Input/output contain rows*cols elements, rows<=1024. Enabled index output has
/// the same shape. Values are F-order; launch256 threads/block. Values and index
/// tie-breakers implement stable CUB key ordering, including signed-zero equality.
/// raw_keys bit 8 selects unsigned keys; bit 9 selects u32 instead of i64 indices.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_sort_small(
    values: *const u32,
    output: *mut u32,
    output_indices: *mut i64,
    rows: u64,
    cols: u64,
    raw_keys: u32,
) {
    unsafe {
        static mut VALUES: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut INDICES: SharedArray<u32, 1024> = SharedArray::UNINIT;
        let keys = SharedArray::as_raw_mut_ptr(&raw mut VALUES);
        let indices = SharedArray::as_raw_mut_ptr(&raw mut INDICES);
        let tid = thread::threadIdx_x();
        let padded = (rows as u32).next_power_of_two();
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            let mut row = tid;
            while row < padded {
                *keys.add(row as usize) = if (row as u64) < rows {
                    *values.add((col * rows + row as u64) as usize)
                } else {
                    0
                };
                *indices.add(row as usize) = if (row as u64) < rows { row } else { u32::MAX };
                row += 256;
            }
            thread::sync_threads();
            let mut size = 2;
            while size <= padded {
                let mut distance = size / 2;
                while distance > 0 {
                    row = tid;
                    while row < padded {
                        let other = row ^ distance;
                        if other > row {
                            let ai = *indices.add(row as usize);
                            let bi = *indices.add(other as usize);
                            let a = *keys.add(row as usize);
                            let b = *keys.add(other as usize);
                            let ak = if ai == u32::MAX {
                                u32::MAX
                            } else if raw_keys & 256 != 0 {
                                a
                            } else {
                                ordered_key(f32::from_bits(a))
                            };
                            let bk = if bi == u32::MAX {
                                u32::MAX
                            } else if raw_keys & 256 != 0 {
                                b
                            } else {
                                ordered_key(f32::from_bits(b))
                            };
                            let greater = ak > bk || (ak == bk && ai > bi);
                            let less = ak < bk || (ak == bk && ai < bi);
                            if if row & size == 0 { greater } else { less } {
                                *keys.add(row as usize) = b;
                                *keys.add(other as usize) = a;
                                *indices.add(row as usize) = bi;
                                *indices.add(other as usize) = ai;
                            }
                        }
                        row += 256;
                    }
                    thread::sync_threads();
                    distance /= 2;
                }
                size *= 2;
            }
            row = tid;
            while (row as u64) < rows {
                let pos = (col * rows + row as u64) as usize;
                *output.add(pos) = *keys.add(row as usize);
                if !output_indices.is_null() {
                    if raw_keys & 512 != 0 {
                        *output_indices.cast::<u32>().add(pos) = *indices.add(row as usize);
                    } else {
                        *output_indices.add(pos) = *indices.add(row as usize) as i64;
                    }
                }
                row += 256;
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Float32 input has rows*cols elements in F order; masks has cols u32 entries.
/// Detect byte differences in magnitude bits. Sign ordering is always retained
/// in the high-byte pass. Launch 256 threads per block.
#[kernel]
pub unsafe fn rank_sort_byte_mask(values: *const u32, masks: *mut u32, rows: u64, cols: u64) {
    unsafe {
        static mut PARTIAL: SharedArray<u32, 16> = SharedArray::UNINIT;
        let partial = SharedArray::as_raw_mut_ptr(&raw mut PARTIAL);
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let wid = tid / 32;
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            let mut row = tid as u64;
            let mut any = 0u32;
            let mut all = u32::MAX;
            while row < rows {
                let value = *values.add((col * rows + row) as usize) & 0x7fff_ffff;
                any |= value;
                all &= value;
                row += 256;
            }
            for distance in [16, 8, 4, 2, 1] {
                any |= warp::shuffle_xor(any, distance);
                all &= warp::shuffle_xor(all, distance);
            }
            if lane == 0 {
                *partial.add(wid as usize) = any;
                *partial.add(8 + wid as usize) = all;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut any = 0;
                let mut all = u32::MAX;
                for w in 0..8 {
                    any |= *partial.add(w);
                    all &= *partial.add(8 + w);
                }
                let variable = any ^ all;
                let mut mask = 8;
                for byte in 0..3 {
                    if (variable >> (byte * 8)) & 255 != 0 {
                        mask |= 1 << byte;
                    }
                }
                *masks.add(col as usize) = mask;
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}

#[inline(always)]
unsafe fn scatter_values_body<const SEGMENTED: bool>(
    values: *const u32,
    output: *mut u32,
    histogram: *const u32,
    permutation: *const u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        let tid = thread::threadIdx_x() as u64;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < cols * tiles {
            let col = tile / tiles;
            let local_tile = tile % tiles;
            let (start, end, segment_start) = if SEGMENTED {
                let segment = segment_for_tile(tile_offsets, segments, local_tile);
                let segment_start = *begins.add(segment as usize);
                (
                    segment_start + (local_tile - *tile_offsets.add(segment as usize)) * TILE_ROWS,
                    *ends.add(segment as usize),
                    segment_start,
                )
            } else {
                (local_tile * TILE_ROWS, rows, 0)
            };
            for item in 0..4 {
                let row = start + item * 256 + tid;
                if row < end {
                    let value = *values.add((col * rows + row) as usize);
                    let digit = radix_digit(value, shift) as u64;
                    let local_base = *permutation.add((tile * 256 + digit) as usize);
                    let global_base = *histogram.add((tile * 256 + digit) as usize);
                    let local = row - start;
                    let pos =
                        col * rows + segment_start + global_base as u64 + local - local_base as u64;
                    *output.add(pos as usize) = value;
                }
            }
            tile += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Input/output are disjoint F-order matrices. Histogram/local prefixes contain
/// cols*tiles*256 u32 entries; segment metadata, when present, is validated.
/// Launch 256 threads per block. Output contains stable locally ordered keys.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_hist_values(
    values: *const u32,
    output: *mut u32,
    histogram: *mut u32,
    local_prefix: *mut u32,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        histogram_body::<false, true>(
            values,
            histogram,
            local_prefix,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            rows,
            cols,
            0,
            tiles,
            shift,
            output,
        );
    }
}
/// # Safety
/// Inputs contain locally ordered keys and matching per-tile digit prefixes;
/// output is disjoint and contains rows*cols entries. Launch 256 threads/block.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_scatter_values(
    values: *const u32,
    output: *mut u32,
    histogram: *const u32,
    local_prefix: *const u32,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        scatter_values_body::<false>(
            values,
            output,
            histogram,
            local_prefix,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            rows,
            cols,
            0,
            tiles,
            shift,
        );
    }
}

/// # Safety
/// Input/output are disjoint F-order matrices. Histogram/local prefixes contain
/// cols*tiles*256 u32 entries; segment metadata, when present, is validated.
/// Launch 256 threads per block. Output contains stable locally ordered keys.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_segments_hist_values(
    values: *const u32,
    output: *mut u32,
    histogram: *mut u32,
    local_prefix: *mut u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        histogram_body::<true, true>(
            values,
            histogram,
            local_prefix,
            begins,
            ends,
            tile_offsets,
            rows,
            cols,
            segments,
            tiles,
            shift,
            output,
        );
    }
}
/// # Safety
/// Inputs contain locally ordered keys and matching per-tile digit prefixes;
/// output is disjoint and contains rows*cols entries. Launch 256 threads/block.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_radix_segments_scatter_values(
    values: *const u32,
    output: *mut u32,
    histogram: *const u32,
    local_prefix: *const u32,
    begins: *const u64,
    ends: *const u64,
    tile_offsets: *const u64,
    rows: u64,
    cols: u64,
    segments: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        scatter_values_body::<true>(
            values,
            output,
            histogram,
            local_prefix,
            begins,
            ends,
            tile_offsets,
            rows,
            cols,
            segments,
            tiles,
            shift,
        );
    }
}

// Constant indices keep each thread's 32 keys in registers. Ordinary indexed
// loops spill these arrays in the current CUDA compiler even with unroll hints.
macro_rules! packed_items {
    ($index:ident, $body:block) => {
        packed_items!(@items $index, $body,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    };
    (@items $index:ident, $body:block, $($value:literal),*) => {
        $({ const $index: usize = $value; $body })*
    };
}

#[inline(always)]
fn packed_variation_reduce(mut any: u32, mut all: u32) -> (u32, u32) {
    macro_rules! step {
        ($offset:literal) => {
            any |= warp::shuffle_down(any, $offset);
            all &= warp::shuffle_down(all, $offset);
        };
    }
    step!(16);
    step!(8);
    step!(4);
    step!(2);
    step!(1);
    (any, all)
}

/// Stable block radix sort for the bounded reference-column tier. Six-bit
/// passes pack two 16-bit counters into each word; 8192 items cannot carry
/// between the halves. Shared storage alternates between counters and keys.
///
/// # Safety
/// Launch exactly 256 threads and one block per column. Both buffers contain
/// `rows * cols` column-major u32 values, with `1 <= rows <= 8192`.
#[kernel]
#[cuda_device::launch_bounds(256)]
pub unsafe fn rank_sort_packed(
    input: *const u32,
    output: *mut u32,
    rows: u64,
    cols: u64,
    raw: u32,
) {
    unsafe {
        static mut STORAGE: SharedArray<u32, 8448> = SharedArray::UNINIT;
        static mut WARPS: SharedArray<u32, 9> = SharedArray::UNINIT;
        let storage = SharedArray::as_raw_mut_ptr(&raw mut STORAGE);
        let sums = SharedArray::as_raw_mut_ptr(&raw mut WARPS);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let wid = tid / 32;
        let col = thread::blockIdx_x() as u64;
        if col >= cols {
            return;
        }
        // Coalesced input, then blocked register ownership preserves stable
        // ordering when each thread processes its 32 consecutive input keys.
        packed_items!(I, {
            let row = I * 256 + tid;
            *storage.add(row) = if (row as u64) < rows {
                *input.add((col * rows + row as u64) as usize)
            } else if raw != 0 {
                u32::MAX
            } else {
                0x7fff_ffff // Largest positive NaN key; padding remains last.
            };
        });
        thread::sync_threads();
        let mut keys = [0u32; 32];
        packed_items!(I, {
            keys[I] = *storage.add(tid * 32 + I);
        });
        thread::sync_threads();

        // Skip digits constant within each sign partition. The final digit
        // always runs to separate signs, including signed-zero equivalence.
        // Padding is excluded from this reduction and has maximal digits.
        let mut any = 0u32;
        let mut all = u32::MAX;
        packed_items!(I, {
            if ((tid * 32 + I) as u64) < rows {
                let bits = keys[I] & if raw != 0 { u32::MAX } else { 0x7fff_ffff };
                any |= bits;
                all &= bits;
            }
        });
        let (any, all) = packed_variation_reduce(any, all);
        if lane == 0 {
            *storage.add(wid) = any;
            *storage.add(8 + wid) = all;
        }
        thread::sync_threads();
        if wid == 0 {
            let any = if lane < 8 { *storage.add(lane) } else { 0 };
            let all = if lane < 8 {
                *storage.add(8 + lane)
            } else {
                u32::MAX
            };
            let (any, all) = packed_variation_reduce(any, all);
            if lane == 0 {
                *storage.add(16) = (any ^ all) | 0xc000_0000;
            }
        }
        thread::sync_threads();
        let varied = *storage.add(16);
        thread::sync_threads();

        let mut shift = 0;
        while shift < 32 {
            if (varied >> shift) & 63 == 0 {
                shift += 6;
                continue;
            }
            packed_items!(I, {
                *storage.add(I * 256 + tid) = 0;
            });
            *storage.add(32 * 256 + tid) = 0;
            let mut ranks = [0u32; 32];
            let mut locations = [0u32; 32];
            packed_items!(I, {
                let key = if raw != 0 {
                    keys[I]
                } else {
                    ordered_key(f32::from_bits(keys[I]))
                };
                let digit = (key >> shift) & 63;
                let location = (digit as usize & 31) * 256 + tid;
                let half = (digit >> 5) * 16;
                let counter = *storage.add(location);
                ranks[I] = (counter >> half) & 65535;
                locations[I] = digit;
                *storage.add(location) = counter + (1 << half);
            });
            thread::sync_threads();

            // Scan the packed histogram in contiguous chunks. The upper half
            // receives the complete lower-half count before digit extraction.
            let mut cached = [0u32; 33];
            let mut total = 0u32;
            packed_items!(I, {
                let value = *storage.add(tid * 33 + I);
                cached[I] = value;
                total += value;
            });
            cached[32] = *storage.add(tid * 33 + 32);
            total += cached[32];
            let mut inclusive = total;
            let mut distance = 1;
            while distance < 32 {
                let previous = warp::shuffle_up(inclusive, distance);
                if lane >= distance as usize {
                    inclusive += previous;
                }
                distance *= 2;
            }
            if lane == 31 {
                *sums.add(wid) = inclusive;
            }
            thread::sync_threads();
            let mut prefix = inclusive - total;
            for w in 0..wid {
                prefix += *sums.add(w);
            }
            if tid == 255 {
                *sums.add(8) = (prefix + total) << 16;
            }
            thread::sync_threads();
            prefix += *sums.add(8);
            packed_items!(I, {
                *storage.add(tid * 33 + I) = prefix;
                prefix += cached[I];
            });
            *storage.add(tid * 33 + 32) = prefix;
            thread::sync_threads();
            packed_items!(I, {
                let digit = locations[I];
                let location = (digit as usize & 31) * 256 + tid;
                ranks[I] += (*storage.add(location) >> ((digit >> 5) * 16)) & 65535;
            });
            thread::sync_threads();
            packed_items!(I, {
                *storage.add(ranks[I] as usize) = keys[I];
            });
            thread::sync_threads();
            packed_items!(I, {
                keys[I] = *storage.add(tid * 32 + I);
            });
            thread::sync_threads();
            shift += 6;
        }
        packed_items!(I, {
            *storage.add(tid * 32 + I) = keys[I];
        });
        thread::sync_threads();
        packed_items!(I, {
            let row = I * 256 + tid;
            if (row as u64) < rows {
                *output.add((col * rows + row as u64) as usize) = *storage.add(row);
            }
        });
    }
}
