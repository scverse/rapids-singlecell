//! Exact OVO tiers: unsorted U identity, shared scans/sorts, and sorted huge groups.
#![allow(clippy::too_many_arguments)]
use cuda_device::{SharedArray, kernel, thread};

#[inline(always)]
unsafe fn lower(values: *const f32, mut lo: u64, mut hi: u64, value: f32) -> u64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *values.add(mid as usize) } < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
#[inline(always)]
unsafe fn upper(values: *const f32, mut lo: u64, mut hi: u64, value: f32) -> u64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *values.add(mid as usize) } <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
#[inline(always)]
fn tie_term(n: f64) -> f64 {
    n * n * n - n
}
#[inline(always)]
fn tie_delta(group_count: u64, reference_count: u64) -> f64 {
    let combined = group_count + reference_count;
    // The original kernels skip singleton group terms and absent overlaps.
    // Preserve the cubic arithmetic for real ties without charging the common
    // continuous-data path for float64 operations whose result is exactly zero.
    if combined <= 1 {
        0.0
    } else {
        tie_term(combined as f64) - tie_term(reference_count as f64)
    }
}
#[inline(always)]
fn tie_factor(n: u64, ties: f64) -> f64 {
    if n < 2 {
        1.0
    } else {
        1.0 - ties / tie_term(n as f64)
    }
}
#[inline(always)]
unsafe fn reduce_pair(a: f64, b: f64, scratch: *mut f64) -> (f64, f64) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let warp = tid / 32;
    let warps = thread::blockDim_x() as usize / 32;
    let a = crate::harmony::sum_f64(a);
    let b = crate::harmony::sum_f64(b);
    if lane == 0 {
        unsafe {
            *scratch.add(warp) = a;
            *scratch.add(warps + warp) = b;
        }
    }
    thread::sync_threads();
    let a = if lane < warps {
        unsafe { *scratch.add(lane) }
    } else {
        0.0
    };
    let b = if lane < warps {
        unsafe { *scratch.add(warps + lane) }
    } else {
        0.0
    };
    (crate::harmony::sum_f64(a), crate::harmony::sum_f64(b))
}

/// # Safety
/// Reference is sorted F-order, ties has cols entries, and block size is 256.
#[kernel]
pub unsafe fn rank_ovo_ref_ties(reference: *const f32, ties: *mut f64, rows: u64, cols: u64) {
    unsafe {
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            let values = reference.add((col * rows) as usize);
            let mut row = tid;
            let mut sum = 0.0;
            while row < rows {
                let v = *values.add(row as usize);
                if row == 0 || v != *values.add((row - 1) as usize) {
                    let end = upper(values, row + 1, rows, v);
                    let count = end - row;
                    if count > 1 {
                        sum += tie_term(count as f64);
                    }
                }
                row += 256;
            }
            let (sum, _) = reduce_pair(sum, 0.0, scratch);
            if tid == 0 {
                *ties.add(col as usize) = sum;
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Input matrices have rows*cols elements, offsets partitions group rows, and
/// outputs are groups*out_cols. Reference is sorted. Launch 256 threads/block.
#[kernel]
pub unsafe fn rank_ovo_unsorted(
    reference: *const f32,
    groups: *const f32,
    offsets: *const u64,
    ranks: *mut f64,
    nref: u64,
    nrows: u64,
    cols: u64,
    ngroups: u64,
    out_cols: u64,
    col_offset: u64,
) {
    unsafe {
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * ngroups {
            let col = segment % cols;
            let group = segment / cols;
            let start = *offsets.add(group as usize);
            let n = *offsets.add(group as usize + 1) - start;
            let r = reference.add((col * nref) as usize);
            let g = groups.add((col * nrows + start) as usize);
            let mut row = tid;
            let mut sum = 0.0;
            while row < n {
                let v = *g.add(row as usize);
                let a = lower(r, 0, nref, v);
                let b = upper(r, a, nref, v);
                sum += a as f64 + 0.5 * (b - a) as f64;
                row += 256;
            }
            let (sum, _) = reduce_pair(sum, 0.0, scratch);
            if tid == 0 {
                *ranks.add((group * out_cols + col_offset + col) as usize) =
                    sum + n as f64 * (n as f64 + 1.0) * 0.5;
            }
            thread::sync_threads();
            segment += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Same layout as rank_ovo_unsorted; tie_base has cols entries, tie output matches
/// ranks. Only groups of at most 512 are processed, using 256 threads per block.
#[kernel]
pub unsafe fn rank_ovo_medium(
    reference: *const f32,
    groups: *const f32,
    offsets: *const u64,
    tie_base: *const f64,
    ranks: *mut f64,
    ties: *mut f64,
    nref: u64,
    nrows: u64,
    cols: u64,
    ngroups: u64,
    out_cols: u64,
    col_offset: u64,
) {
    unsafe {
        static mut VALUES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let values = SharedArray::as_raw_mut_ptr(&raw mut VALUES);
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * ngroups {
            let col = segment % cols;
            let group = segment / cols;
            let start = *offsets.add(group as usize);
            let n = *offsets.add(group as usize + 1) - start;
            if n <= 512 {
                let r = reference.add((col * nref) as usize);
                let g = groups.add((col * nrows + start) as usize);
                let mut row = tid;
                while row < n {
                    *values.add(row as usize) = *g.add(row as usize);
                    row += 256;
                }
                thread::sync_threads();
                let mut rank = 0.0;
                let mut delta = 0.0;
                row = tid;
                while row < n {
                    let v = *values.add(row as usize);
                    let a = lower(r, 0, nref, v);
                    let b = upper(r, a, nref, v);
                    let mut count = 0;
                    let mut less = 0;
                    let mut first = true;
                    for j in 0..n {
                        less += (*values.add(j as usize) < v) as u64;
                        if *values.add(j as usize) == v {
                            count += 1;
                            if j < row {
                                first = false;
                            }
                        }
                    }
                    rank += (a + less) as f64 + 0.5 * (b - a + count + 1) as f64;
                    if first {
                        delta += tie_delta(count, b - a);
                    }
                    row += 256;
                }
                let (rank, delta) = reduce_pair(rank, delta, scratch);
                if tid == 0 {
                    let pos = (group * out_cols + col_offset + col) as usize;
                    *ranks.add(pos) = rank;
                    *ties.add(pos) = tie_factor(nref + n, *tie_base.add(col as usize) + delta);
                }
                thread::sync_threads();
            }
            segment += thread::gridDim_x() as u64;
        }
    }
}

#[inline(always)]
unsafe fn contains_sorted_nan(values: *const f32, n: u64) -> bool {
    n > 0 && unsafe { (*values).is_nan() || (*values.add(n as usize - 1)).is_nan() }
}

/// Preserve CUB's deterministic floating comparison behavior for NaNs; the
/// ordinary finite path can use the equivalent, cheaper U identity.
#[inline(always)]
unsafe fn legacy_sorted_group(r: *const f32, g: *const f32, nref: u64, n: u64) -> (f64, f64) {
    let mut logical = thread::threadIdx_x() as u64;
    let mut rank = 0.0;
    let mut delta = 0.0;
    // CUB's original tier uses 512 logical threads. NaNs make the monotonic
    // search cursors depend on that stride, even when physical blocks are smaller.
    while logical < 512 {
        let mut row = logical;
        let (mut rl, mut ru, mut gl, mut gu) = (0, 0, 0, 0);
        while row < n {
            let v = unsafe { *g.add(row as usize) };
            rl = unsafe { lower(r, rl, nref, v) };
            ru = unsafe { upper(r, ru.max(rl), nref, v) };
            gl = unsafe { lower(g, gl, n, v) };
            gu = unsafe { upper(g, gu.max(gl), n, v) };
            rank += (rl + gl) as f64 + (ru - rl + gu - gl + 1) as f64 * 0.5;
            if row == 0 || v != unsafe { *g.add(row as usize - 1) } {
                let end = unsafe { upper(g, row + 1, n, v) };
                let a = unsafe { lower(r, 0, nref, v) };
                let b = unsafe { upper(r, a, nref, v) };
                delta += tie_delta(end - row, b - a);
            }
            row += 512;
        }
        logical += thread::blockDim_x() as u64;
    }
    if thread::threadIdx_x() == 0 {
        rank -= n as f64 * (n as f64 + 1.0) * 0.5;
    }
    (rank, delta)
}

#[inline(always)]
unsafe fn sorted_group(r: *const f32, g: *const f32, nref: u64, n: u64) -> (f64, f64) {
    let mut row = thread::threadIdx_x() as u64;
    let mut rank = 0.0;
    let mut delta = 0.0;
    // Each unique group value contributes once. Reference ties are amortized
    // across groups; no group rescans reference-only ties.
    while row < n {
        let v = unsafe { *g.add(row as usize) };
        if row == 0 || v != unsafe { *g.add((row - 1) as usize) } {
            let end = unsafe { upper(g, row + 1, n, v) };
            let a = unsafe { lower(r, 0, nref, v) };
            let b = unsafe { upper(r, a, nref, v) };
            rank += (end - row) as f64 * (a as f64 + 0.5 * (b - a) as f64);
            delta += tie_delta(end - row, b - a);
        }
        row += thread::blockDim_x() as u64;
    }
    (rank, delta)
}

/// # Safety
/// Same contract as rank_ovo_medium. Only 513..=2500-row groups are processed.
/// Shared bitonic scratch pads each group to its own power of two, at most 4096.
/// Launch with 1024 threads per block.
#[kernel]
#[cuda_device::launch_bounds(1024)]
pub unsafe fn rank_ovo_large(
    reference: *const f32,
    groups: *const f32,
    offsets: *const u64,
    tie_base: *const f64,
    ranks: *mut f64,
    ties: *mut f64,
    nref: u64,
    nrows: u64,
    cols: u64,
    ngroups: u64,
    out_cols: u64,
    col_offset: u64,
) {
    unsafe {
        static mut VALUES: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f64, 64> = SharedArray::UNINIT;
        let values = SharedArray::as_raw_mut_ptr(&raw mut VALUES);
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * ngroups {
            let col = segment % cols;
            let group = segment / cols;
            let start = *offsets.add(group as usize);
            let n = *offsets.add(group as usize + 1) - start;
            if n > 512 && n <= 2500 {
                let r = reference.add((col * nref) as usize);
                let g = groups.add((col * nrows + start) as usize);
                let padded = (n as u32).next_power_of_two();
                let mut row = tid as u32;
                let mut has_nan = false;
                while row < padded {
                    *values.add(row as usize) = if (row as u64) < n {
                        let v = *g.add(row as usize);
                        has_nan |= v.is_nan();
                        v
                    } else {
                        f32::INFINITY
                    };
                    row += thread::blockDim_x();
                }
                let (nan_count, _) = reduce_pair(has_nan as u32 as f64, 0.0, scratch);
                thread::sync_threads();
                let mut size = 2u32;
                while size <= padded {
                    let mut distance = size / 2;
                    while distance != 0 {
                        row = tid as u32;
                        while row < padded {
                            let other = row ^ distance;
                            if other > row {
                                let a = *values.add(row as usize);
                                let b = *values.add(other as usize);
                                let asc = row & size == 0;
                                if if asc { a > b } else { a < b } {
                                    *values.add(row as usize) = b;
                                    *values.add(other as usize) = a;
                                }
                            }
                            row += thread::blockDim_x();
                        }
                        thread::sync_threads();
                        distance /= 2;
                    }
                    size *= 2;
                }
                let (rank, delta) = if nan_count > 0.0 || contains_sorted_nan(r, nref) {
                    legacy_sorted_group(r, values, nref, n)
                } else {
                    sorted_group(r, values, nref, n)
                };
                let (rank, delta) = reduce_pair(rank, delta, scratch);
                if tid == 0 {
                    let pos = (group * out_cols + col_offset + col) as usize;
                    *ranks.add(pos) = rank + n as f64 * (n as f64 + 1.0) * 0.5;
                    *ties.add(pos) = tie_factor(nref + n, *tie_base.add(col as usize) + delta);
                }
                thread::sync_threads();
            }
            segment += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Float64 inputs are F-order; offsets partitions group rows. Outputs contain
/// (ngroups+1)*out_cols elements, with the reference in the final output row.
#[kernel]
pub unsafe fn rank_ovo_stats(
    reference: *const f64,
    groups: *const f64,
    offsets: *const u64,
    sums: *mut f64,
    nnz: *mut f64,
    nref: u64,
    nrows: u64,
    cols: u64,
    ngroups: u64,
    out_cols: u64,
    col_offset: u64,
) {
    unsafe {
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * (ngroups + 1) {
            let col = segment % cols;
            let group = segment / cols;
            let (values, n) = if group == ngroups {
                (reference.add((col * nref) as usize), nref)
            } else {
                let start = *offsets.add(group as usize);
                (
                    groups.add((col * nrows + start) as usize),
                    *offsets.add(group as usize + 1) - start,
                )
            };
            let mut row = tid;
            let mut sum = 0.0;
            let mut count = 0.0;
            while row < n {
                let v = *values.add(row as usize);
                sum += v;
                count += (v != 0.0) as u32 as f64;
                row += 256;
            }
            let (sum, count) = reduce_pair(sum, count, scratch);
            if tid == 0 {
                let pos = (group * out_cols + col_offset + col) as usize;
                *sums.add(pos) = sum;
                if !nnz.is_null() {
                    *nnz.add(pos) = count;
                }
            }
            thread::sync_threads();
            segment += thread::gridDim_x() as u64;
        }
    }
}

/// # Safety
/// Active (>2500-row) segments in group are sorted independently in F-order.
/// Offsets partitions all group rows, tie_base has cols entries, outputs contain
/// ngroups*out_cols. Launch 256 threads per block.
#[kernel]
pub unsafe fn rank_ovo_huge_segments(
    reference: *const f32,
    group: *const f32,
    offsets: *const u64,
    tie_base: *const f64,
    ranks: *mut f64,
    ties: *mut f64,
    nref: u64,
    nrows: u64,
    cols: u64,
    ngroups: u64,
    out_cols: u64,
    col_offset: u64,
) {
    unsafe {
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x();
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * ngroups {
            let col = segment % cols;
            let g = segment / cols;
            let start = *offsets.add(g as usize);
            let n = *offsets.add(g as usize + 1) - start;
            if n > 2500 {
                let r = reference.add((col * nref) as usize);
                let gp = group.add((col * nrows + start) as usize);
                let (rank, delta) = if contains_sorted_nan(r, nref) || contains_sorted_nan(gp, n) {
                    legacy_sorted_group(r, gp, nref, n)
                } else {
                    sorted_group(r, gp, nref, n)
                };
                let (rank, delta) = reduce_pair(rank, delta, scratch);
                if tid == 0 {
                    let pos = (g * out_cols + col_offset + col) as usize;
                    *ranks.add(pos) = rank + n as f64 * (n as f64 + 1.0) * 0.5;
                    *ties.add(pos) = tie_factor(nref + n, *tie_base.add(col as usize) + delta);
                }
                thread::sync_threads();
            }
            segment += thread::gridDim_x() as u64;
        }
    }
}
