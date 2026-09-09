//! Sparse streaming aggregation into full-width grouped output planes.
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{kernel, thread};
#[inline(always)]
unsafe fn index(p: *const u8, i: u64, wide: u32) -> i64 {
    unsafe {
        if wide != 0 {
            *p.cast::<i64>().add(i as usize)
        } else {
            *p.cast::<i32>().add(i as usize) as i64
        }
    }
}
#[inline(always)]
unsafe fn value(p: *const u8, i: u64, wide: u32) -> f64 {
    unsafe {
        if wide != 0 {
            *p.cast::<f64>().add(i as usize)
        } else {
            *p.cast::<f32>().add(i as usize) as f64
        }
    }
}
#[inline(always)]
unsafe fn add(p: *mut f64, i: u64, v: f64) {
    if !p.is_null() {
        unsafe { DeviceAtomicF64::from_ptr(p.add(i as usize)) }
            .fetch_add(v, AtomicOrdering::Relaxed);
    }
}
/// # Safety
/// All pointers borrow validated device allocations: indptr has major+1 entries,
/// indices/data have nnz, codes/mask have rows, and each nonnull output contains
/// groups*out_cols doubles. Outputs are disjoint from inputs and each other.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rank_stream_aggr(
    indptr: *const u8,
    indices: *const u8,
    data: *const u8,
    codes: *const i32,
    mask: *const u8,
    sums: *mut f64,
    counts: *mut f64,
    squares: *mut f64,
    major: u64,
    rows: u64,
    out_cols: u64,
    groups: u64,
    col_offset: u64,
    nnz: u64,
    pwide: u32,
    iwide: u32,
    wide: u32,
    csc: u32,
) {
    let mut seg = thread::blockIdx_x() as u64;
    let lane = thread::threadIdx_x() as u64;
    while seg < major {
        let start = unsafe { index(indptr, seg, pwide) };
        let stop = unsafe { index(indptr, seg + 1, pwide) };
        if start >= 0 && stop >= start && stop as u64 <= nnz {
            let mut p = start as u64 + lane;
            while p < stop as u64 {
                let ix = unsafe { index(indices, p, iwide) };
                let row = if csc != 0 { ix } else { seg as i64 };
                let col = if csc != 0 {
                    seg as i64 + col_offset as i64
                } else {
                    ix
                };
                if row >= 0
                    && (row as u64) < rows
                    && col >= 0
                    && (col as u64) < out_cols
                    && (mask.is_null() || unsafe { *mask.add(row as usize) } != 0)
                {
                    let group = unsafe { *codes.add(row as usize) };
                    if group >= 0 && (group as u64) < groups {
                        let o = group as u64 * out_cols + col as u64;
                        let v = unsafe { value(data, p, wide) };
                        unsafe {
                            add(sums, o, v);
                            add(counts, o, 1.0);
                            add(squares, o, v * v);
                        }
                    }
                }
                p += thread::blockDim_x() as u64;
            }
        }
        seg += thread::gridDim_x() as u64;
    }
}
