//! Stable segmented radix sorting for contiguous float32 or uint32 columns.
#![allow(clippy::too_many_arguments)]
use cuda_device::{SharedArray, kernel, thread, warp};

#[inline(always)]
fn ordered_key(value: f32) -> u32 {
    let bits = value.to_bits();
    if value.is_nan() {
        u32::MAX
    } else if value == 0.0 {
        0x8000_0000
    } else if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
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

/// # Safety
/// Input/local_rank have rows*cols elements. Histogram has cols*tiles*256
/// elements, tiles=ceil(rows/256). Launch with 256 threads per block.
#[kernel]
pub unsafe fn rank_radix_hist(
    values: *const u32,
    histogram: *mut u32,
    local_rank: *mut u8,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    unsafe {
        static mut COUNTS: SharedArray<u32, 2048> = SharedArray::UNINIT;
        let counts = SharedArray::as_raw_mut_ptr(&raw mut COUNTS);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let wid = tid / 32;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < cols * tiles {
            for w in 0..8 {
                *counts.add(w * 256 + tid) = 0;
            }
            thread::sync_threads();
            let col = tile / tiles;
            let row = (tile % tiles) * 256 + tid as u64;
            let valid = row < rows;
            let pos = col * rows + row;
            let digit = if valid {
                radix_digit(*values.add(pos as usize), shift)
            } else {
                256
            };
            let peers = warp::match_any_sync(u32::MAX, digit);
            if valid && lane as u32 == peers.trailing_zeros() {
                *counts.add(wid * 256 + digit as usize) = peers.count_ones();
            }
            thread::sync_threads();
            if valid {
                let mut rank = (peers & warp::lanemask_lt()).count_ones();
                for w in 0..wid {
                    rank += *counts.add(w * 256 + digit as usize);
                }
                *local_rank.add(pos as usize) = rank as u8;
            }
            let mut total = 0;
            for w in 0..8 {
                total += *counts.add(w * 256 + tid);
            }
            *histogram.add((tile * 256 + tid as u64) as usize) = total;
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Histogram has cols*tiles*256 counts. Launch one 256-thread block per column.
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
/// Value buffers and enabled index buffers have rows*cols elements and are
/// disjoint. Histogram/local_rank were produced for this input and radix digit.
#[kernel]
pub unsafe fn rank_radix_scatter(
    values: *const u32,
    indices: *const i64,
    output: *mut u32,
    output_indices: *mut i64,
    histogram: *const u32,
    local_rank: *const u8,
    rows: u64,
    cols: u64,
    tiles: u64,
    shift: u32,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < rows * cols {
        unsafe {
            let col = i / rows;
            let row = i % rows;
            let value = *values.add(i as usize);
            let digit = radix_digit(value, shift);
            let prefix = *histogram.add(((col * tiles + row / 256) * 256 + digit as u64) as usize);
            let pos = col * rows + prefix as u64 + *local_rank.add(i as usize) as u64;
            *output.add(pos as usize) = value;
            if !output_indices.is_null() {
                *output_indices.add(pos as usize) = if indices.is_null() {
                    row as i64
                } else {
                    *indices.add(i as usize)
                };
            }
        }
        i += thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
    }
}
