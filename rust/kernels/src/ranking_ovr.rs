//! OVR ranks from contiguous sorted tie runs with per-column group reduction.
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU64, DeviceAtomicF64};
use cuda_device::{DynamicSharedArray, SharedArray, kernel, thread};

// Reserve the original 32-warp reduction allowance inside the portable 48 KiB
// per-block budget. The actual static reduction array below occupies 128 bytes.
const SHARED_GROUP_LIMIT: u64 = (48 * 1024 - 32 * 8) / 8;

/// # Safety
/// Values/u32 indices are sorted F-order matrices with rows*cols elements. Codes has
/// rows entries. Rank output is groups*out_cols and initially zero. Ties has
/// out_cols entries or is null. Launch32..512 threads per block in complete warps;
/// rows <= u32::MAX. Dynamic shared storage has groups*8 bytes for groups<=6112,
/// otherwise zero bytes and the initially zero global output is accumulated.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub unsafe fn rank_ovr_runs(
    values: *const f32,
    order: *const u32,
    codes: *const i32,
    ranks: *mut f64,
    ties: *mut f64,
    rows: u64,
    cols: u64,
    groups: u64,
    out_cols: u64,
    offset: u64,
) {
    unsafe {
        // All averaged ranks are half-integers. Exact doubled ranks permit
        // integer shared atomics, avoiding FP64 CAS loops and rounding drift.
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let sums = DynamicSharedArray::<u64>::get();
        let partial = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let threads = thread::blockDim_x() as u64;
        let lane = tid % 32;
        let warp = tid / 32;
        let shared = groups <= SHARED_GROUP_LIMIT;
        let mut col = thread::blockIdx_x() as u64;
        while col < cols {
            if shared {
                let mut g = tid;
                while g < groups {
                    *sums.add(g as usize) = 0;
                    g += threads;
                }
            }
            thread::sync_threads();
            let sv = values.add((col * rows) as usize);
            let si = order.add((col * rows) as usize);
            let chunk = rows.div_ceil(threads);
            let start = (tid * chunk).min(rows);
            let end = (start + chunk).min(rows);
            let mut row = start;
            let mut tie_sum = 0.0;
            while row < end {
                let value = *sv.add(row as usize);
                let mut local_end = row + 1;
                while local_end < end && *sv.add(local_end as usize) == value {
                    local_end += 1;
                }
                let mut first = row;
                if row == start && row > 0 && *sv.add((row - 1) as usize) == value {
                    let mut lo = 0;
                    let mut hi = row;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if *sv.add(mid as usize) < value {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    first = lo;
                }
                let mut last = local_end;
                if local_end == end && local_end < rows && *sv.add(local_end as usize) == value {
                    let mut lo = local_end;
                    let mut hi = rows;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if *sv.add(mid as usize) <= value {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    last = lo;
                }
                let doubled = first + last + 1;
                for j in row..local_end {
                    let original = *si.add(j as usize);
                    if (original as u64) < rows {
                        let group = *codes.add(original as usize);
                        if group >= 0 && (group as u64) < groups {
                            if shared {
                                BlockAtomicU64::from_ptr(sums.add(group as usize))
                                    .fetch_add(doubled, AtomicOrdering::Relaxed);
                            } else {
                                DeviceAtomicF64::from_ptr(
                                    ranks.add((group as u64 * out_cols + offset + col) as usize),
                                )
                                .fetch_add(doubled as f64 * 0.5, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                }
                if !ties.is_null() && first >= start && last - first > 1 {
                    let count = (last - first) as f64;
                    tie_sum += count * count * count - count;
                }
                row = local_end;
            }
            thread::sync_threads();
            if shared {
                let mut g = tid;
                while g < groups {
                    *ranks.add((g * out_cols + offset + col) as usize) =
                        *sums.add(g as usize) as f64 * 0.5;
                    g += threads;
                }
            }
            if !ties.is_null() {
                let local = crate::harmony::sum_f64(tie_sum);
                if lane == 0 {
                    *partial.add(warp as usize) = local;
                }
                thread::sync_threads();
                let v = if lane < threads / 32 {
                    *partial.add(lane as usize)
                } else {
                    0.0
                };
                let total = crate::harmony::sum_f64(v);
                if tid == 0 {
                    let n = rows as f64;
                    *ties.add((offset + col) as usize) = if rows < 2 {
                        1.0
                    } else {
                        1.0 - total / (n * n * n - n)
                    };
                }
            }
            thread::sync_threads();
            col += thread::gridDim_x() as u64;
        }
    }
}
