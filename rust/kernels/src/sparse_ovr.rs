//! Exact sparse Wilcoxon ranks with an analytic contribution from implicit zeros.
#![allow(clippy::too_many_arguments)]
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU64, DeviceAtomicF64};
use cuda_device::{DynamicSharedArray, kernel, thread};

#[inline(always)]
unsafe fn index(pointer: *const u8, at: u64, wide: u32) -> i64 {
    unsafe {
        if wide != 0 {
            *pointer.cast::<i64>().add(at as usize)
        } else {
            *pointer.cast::<i32>().add(at as usize) as i64
        }
    }
}
#[inline(always)]
unsafe fn value(pointer: *const u8, at: u64, wide: u32) -> f64 {
    unsafe {
        if wide != 0 {
            *pointer.cast::<f64>().add(at as usize)
        } else {
            *pointer.cast::<f32>().add(at as usize) as f64
        }
    }
}
#[inline(always)]
unsafe fn add(pointer: *mut f64, value: f64) {
    unsafe { DeviceAtomicF64::from_ptr(pointer) }.fetch_add(value, AtomicOrdering::Relaxed);
}
#[inline(always)]
fn stride() -> u64 {
    thread::blockDim_x() as u64 * thread::gridDim_x() as u64
}
#[inline(always)]
unsafe fn segment(pointer: *const u8, col: u64, wide: u32, nnz: u64) -> (u64, u64) {
    let start = unsafe { index(pointer, col, wide) };
    let stop = unsafe { index(pointer, col + 1, wide) };
    if start >= 0 && stop >= start && stop as u64 <= nnz {
        (start as u64, stop as u64)
    } else {
        (0, 0)
    }
}
#[inline(always)]
fn ordered(value: f32) -> u32 {
    // CUB's numeric ordering treats signed zero as a single rank tie.
    if value == 0.0 {
        0x8000_0000
    } else {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }
}
#[inline(always)]
unsafe fn sorted(keys: *const u32, order: *const u32, base: u64, at: u64) -> u32 {
    let original = unsafe { *order.add((base + at) as usize) } as u64;
    unsafe { *keys.add((base + original) as usize) }
}
#[inline(always)]
fn decoded(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    })
}
#[inline(always)]
unsafe fn bounds(
    keys: *const u32,
    order: *const u32,
    base: u64,
    count: u64,
    key: u32,
) -> (u64, u64) {
    let mut low = 0;
    let mut high = count;
    while low < high {
        let mid = low + (high - low) / 2;
        let current = unsafe { sorted(keys, order, base, mid) };
        if if key == 0x8000_0000 {
            decoded(current) < 0.0
        } else {
            current < key
        } {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let first = low;
    low = if key == 0x8000_0000 { 0 } else { low };
    high = count;
    while low < high {
        let mid = low + (high - low) / 2;
        let current = unsafe { sorted(keys, order, base, mid) };
        if if key == 0x8000_0000 {
            decoded(current) <= 0.0
        } else {
            current <= key
        } {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    (first, low)
}

/// Pack float32 sort keys and accumulate statistics in original precision.
/// # Safety
/// Valid CSC buffers span nnz entries and col_begin+cols+1 pointers. Keys has
/// cols*width entries; width covers the largest stored column. Codes has rows
/// entries. Non-null outputs have groups*out_cols or out_cols elements.
#[kernel]
pub unsafe fn sparse_ovr_pack(
    data: *const u8,
    indices: *const u8,
    indptr: *const u8,
    codes: *const i32,
    keys: *mut u32,
    sums: *mut f64,
    counts: *mut f64,
    totals: *mut f64,
    total_counts: *mut f64,
    rows: u64,
    cols: u64,
    width: u64,
    nnz: u64,
    col_begin: u64,
    output_begin: u64,
    out_cols: u64,
    groups: u64,
    data_wide: u32,
    index_wide: u32,
    pointer_wide: u32,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < cols * width {
        let col = i / width;
        let offset = i % width;
        let (start, stop) = unsafe { segment(indptr, col_begin + col, pointer_wide, nnz) };
        let p = start + offset;
        let mut key = u32::MAX;
        if p < stop {
            let v = unsafe { value(data, p, data_wide) };
            key = ordered(if data_wide != 0 {
                v as f32
            } else {
                // A float64 round trip quiets f32 signaling NaNs and changes
                // their CUB key ordering. Statistics still use v above.
                unsafe { *data.cast::<f32>().add(p as usize) }
            });
            let row = unsafe { index(indices, p, index_wide) };
            if row >= 0 && (row as u64) < rows {
                let group = unsafe { *codes.add(row as usize) };
                if group >= 0 && (group as u64) < groups {
                    let output = (group as u64 * out_cols + output_begin + col) as usize;
                    unsafe {
                        if !sums.is_null() {
                            add(sums.add(output), v);
                        }
                        if !counts.is_null() && v != 0.0 {
                            add(counts.add(output), 1.0);
                        }
                    }
                }
                let output = (output_begin + col) as usize;
                unsafe {
                    if !totals.is_null() {
                        add(totals.add(output), v);
                    }
                    if !total_counts.is_null() && v != 0.0 {
                        add(total_counts.add(output), 1.0);
                    }
                }
            }
        }
        unsafe { *keys.add(i as usize) = key };
        i += stride();
    }
}

/// Fused per-column sparse rank accumulation and tie reduction.
/// # Safety
/// Validated CSC/sort buffers and outputs follow sparse_ovr_base's contract.
/// Launch 256 threads with (2*groups+32)*8 dynamic shared bytes. The host caps
/// that payload at 48 KiB and rows at i32::MAX, bounding doubled rank sums.
#[kernel]
pub unsafe fn sparse_ovr_fused(
    keys: *const u32,
    order: *const u32,
    indices: *const u8,
    indptr: *const u8,
    codes: *const i32,
    sizes: *const f64,
    ranks: *mut f64,
    tie: *mut f64,
    rows: u64,
    cols: u64,
    width: u64,
    nnz: u64,
    col_begin: u64,
    output_begin: u64,
    out_cols: u64,
    groups: u64,
    index_wide: u32,
    pointer_wide: u32,
) {
    unsafe {
        // Ranks are half-integers. Exact doubled ranks use native shared u64
        // atomics, retaining the original shared path without FP64 CAS loops.
        let sums = DynamicSharedArray::<u64>::get();
        let counts = sums.add(groups as usize);
        let zero_bounds = counts.add(groups as usize);
        let partial = zero_bounds.add(2).cast::<f64>();
        let tid = thread::threadIdx_x() as u64;
        let lane = tid % 32;
        let warp = tid / 32;
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            let (stored_begin, stored_end) = segment(indptr, col_begin + col, pointer_wide, nnz);
            let count = stored_end - stored_begin;
            let base = col * width;
            let mut group = tid;
            while group < groups {
                *sums.add(group as usize) = 0;
                *counts.add(group as usize) = 0;
                group += 256;
            }
            if tid == 0 {
                let (negative, positive) = bounds(keys, order, base, count, 0x8000_0000);
                *zero_bounds = negative;
                *zero_bounds.add(1) = positive;
            }
            thread::sync_threads();
            let negative = *zero_bounds;
            let positive = *zero_bounds.add(1);
            let implicit = rows.saturating_sub(count);
            let zeros = implicit + positive - negative;
            let zero_rank = if zeros != 0 {
                negative as f64 + (zeros as f64 + 1.0) * 0.5
            } else {
                0.0
            };
            let mut tie_sum = 0.0;
            // Walk negative and positive runs separately. The analytic zero
            // block occupies its exact position between these two regions.
            for region in 0..2 {
                let floor = if region == 0 { 0 } else { positive };
                let ceiling = if region == 0 { negative } else { count };
                let chunk = (ceiling - floor).div_ceil(256);
                let start = (floor + tid * chunk).min(ceiling);
                let end = (start + chunk).min(ceiling);
                let mut position = start;
                while position < end {
                    let key = sorted(keys, order, base, position);
                    let value = decoded(key);
                    let mut local_end = position + 1;
                    while local_end < end && decoded(sorted(keys, order, base, local_end)) == value
                    {
                        local_end += 1;
                    }
                    let mut first = position;
                    let mut last = local_end;
                    // Only ties crossing a thread's chunk boundary need a
                    // binary search; interior runs are traversed once.
                    if position == start
                        && position > floor
                        && decoded(sorted(keys, order, base, position - 1)) == value
                    {
                        first = bounds(keys, order, base, count, key).0;
                    }
                    if local_end == end
                        && local_end < ceiling
                        && decoded(sorted(keys, order, base, local_end)) == value
                    {
                        last = bounds(keys, order, base, count, key).1;
                    }
                    let doubled = first + last + 1 + if region == 0 { 0 } else { 2 * implicit };
                    for position in position..local_end {
                        let original = *order.add((base + position) as usize) as u64;
                        if original < count {
                            let row = index(indices, stored_begin + original, index_wide);
                            if row >= 0 && (row as u64) < rows {
                                let group = *codes.add(row as usize);
                                if group >= 0 && (group as u64) < groups {
                                    BlockAtomicU64::from_ptr(sums.add(group as usize))
                                        .fetch_add(doubled, AtomicOrdering::Relaxed);
                                    BlockAtomicU64::from_ptr(counts.add(group as usize))
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                    }
                    if !tie.is_null() && first >= start {
                        let length = (last - first) as f64;
                        tie_sum += length * length * length - length;
                    }
                    position = local_end;
                }
            }
            thread::sync_threads();
            let mut group = tid;
            while group < groups {
                *ranks.add((group * out_cols + output_begin + col) as usize) =
                    (*sizes.add(group as usize) - *counts.add(group as usize) as f64) * zero_rank
                        + *sums.add(group as usize) as f64 * 0.5;
                group += 256;
            }
            if !tie.is_null() {
                if tid == 0 {
                    let z = zeros as f64;
                    tie_sum += z * z * z - z;
                }
                let local = crate::harmony::sum_f64(tie_sum);
                if lane == 0 {
                    *partial.add(warp as usize) = local;
                }
                thread::sync_threads();
                let value = if lane < 8 {
                    *partial.add(lane as usize)
                } else {
                    0.0
                };
                let total = crate::harmony::sum_f64(value);
                if tid == 0 {
                    *tie.add((output_begin + col) as usize) = total;
                }
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}

/// Initialize each group's analytic zero ranks and the zero tie correction.
/// # Safety
/// The validated argsort indexes cols*width keys. Each count is <= rows and
/// width. Sizes has groups entries; output dimensions and pointers are valid.
#[kernel]
pub unsafe fn sparse_ovr_base(
    keys: *const u32,
    order: *const u32,
    indptr: *const u8,
    sizes: *const f64,
    ranks: *mut f64,
    tie: *mut f64,
    zero_ranks: *mut f64,
    zero_bounds: *mut u64,
    rows: u64,
    cols: u64,
    width: u64,
    nnz: u64,
    col_begin: u64,
    output_begin: u64,
    out_cols: u64,
    groups: u64,
    pointer_wide: u32,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < cols * groups.max(1) {
        let col = i % cols;
        let group = i / cols;
        let (start, stop) = unsafe { segment(indptr, col_begin + col, pointer_wide, nnz) };
        let count = stop - start;
        let (negative, positive) = unsafe { bounds(keys, order, col * width, count, 0x8000_0000) };
        let zeros = rows.saturating_sub(count) + positive - negative;
        let zero_rank = if zeros != 0 {
            negative as f64 + (zeros as f64 + 1.0) / 2.0
        } else {
            0.0
        };
        if group < groups {
            unsafe {
                *ranks.add((group * out_cols + output_begin + col) as usize) =
                    *sizes.add(group as usize) * zero_rank;
            }
        }
        if group == 0 {
            unsafe {
                *zero_ranks.add(col as usize) = zero_rank;
                *zero_bounds.add((2 * col) as usize) = negative;
                *zero_bounds.add((2 * col + 1) as usize) = positive;
                if !tie.is_null() {
                    let z = zeros as f64;
                    *tie.add((output_begin + col) as usize) = z * z * z - z;
                }
            }
        }
        i += stride();
    }
}

/// Add stored nonzero ranks relative to the analytic zero base.
/// # Safety
/// Same CSC, sorting, output and scratch contracts as sparse_ovr_base. Row
/// indices and group codes are checked before output access.
#[kernel]
pub unsafe fn sparse_ovr_delta(
    keys: *const u32,
    order: *const u32,
    indices: *const u8,
    indptr: *const u8,
    codes: *const i32,
    zero_ranks: *const f64,
    zero_bounds: *const u64,
    ranks: *mut f64,
    tie: *mut f64,
    rows: u64,
    cols: u64,
    width: u64,
    nnz: u64,
    col_begin: u64,
    output_begin: u64,
    out_cols: u64,
    groups: u64,
    index_wide: u32,
    pointer_wide: u32,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < cols * width {
        let col = i / width;
        let position = i % width;
        let base = col * width;
        let (start, stop) = unsafe { segment(indptr, col_begin + col, pointer_wide, nnz) };
        let count = stop - start;
        if position < count {
            let original = unsafe { *order.add(i as usize) } as u64;
            let key = unsafe { *keys.add((base + original) as usize) };
            let negative = unsafe { *zero_bounds.add((2 * col) as usize) };
            let positive = unsafe { *zero_bounds.add((2 * col + 1) as usize) };
            if (position < negative || position >= positive) && original < count {
                // NaNs compare unequal even to themselves in the original
                // sparse tie walker, so every stored NaN has a singleton rank.
                let (first, end) = if decoded(key).is_nan() {
                    (position, position + 1)
                } else {
                    unsafe { bounds(keys, order, base, count, key) }
                };
                let rank = (first as f64 + end as f64 + 1.0) / 2.0
                    + if position >= positive {
                        rows.saturating_sub(count) as f64
                    } else {
                        0.0
                    };
                let row = unsafe { index(indices, start + original, index_wide) };
                if row >= 0 && (row as u64) < rows {
                    let group = unsafe { *codes.add(row as usize) };
                    if group >= 0 && (group as u64) < groups {
                        unsafe {
                            add(
                                ranks.add((group as u64 * out_cols + output_begin + col) as usize),
                                rank - *zero_ranks.add(col as usize),
                            );
                        }
                    }
                }
                if !tie.is_null() && position == first {
                    let ties = (end - first) as f64;
                    unsafe {
                        add(
                            tie.add((output_begin + col) as usize),
                            ties * ties * ties - ties,
                        )
                    };
                }
            }
        }
        i += stride();
    }
}
