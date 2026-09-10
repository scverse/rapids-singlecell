//! Compact CSR uploads feeding the original bounded dense-reference OVO tier.
#![allow(clippy::too_many_arguments)]
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicF64},
    kernel, thread,
};

#[inline(always)]
unsafe fn column(columns: *const u8, position: u64, narrow: u32) -> u64 {
    unsafe {
        if narrow != 0 {
            *columns.cast::<u16>().add(position as usize) as u64
        } else {
            *columns.cast::<u32>().add(position as usize) as u64
        }
    }
}
#[inline(always)]
unsafe fn value(data: *const u8, position: u64, wide: u32) -> f64 {
    unsafe {
        if wide != 0 {
            *data.cast::<f64>().add(position as usize)
        } else {
            *data.cast::<f32>().add(position as usize) as f64
        }
    }
}

/// Extract unique CSR coordinates into a zero-filled F-order float32 window.
/// # Safety
/// The validated CSR arrays cover row_first+rows and nnz; values and columns
/// have the selected widths. Output contains rows*cols initialized zeros.
#[kernel]
pub unsafe fn sparse_ovo_csr_dense(
    data: *const u8,
    columns: *const u8,
    pointers: *const u64,
    output: *mut f32,
    row_first: u64,
    rows: u64,
    first: u64,
    cols: u64,
    nnz: u64,
    wide: u32,
    narrow: u32,
) {
    let mut row =
        thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
    let stride = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
    while row < rows {
        let begin = unsafe { *pointers.add((row_first + row) as usize) };
        let end = unsafe { *pointers.add((row_first + row + 1) as usize) }.min(nnz);
        for position in begin..end {
            let col = unsafe { column(columns, position, narrow) };
            if col >= first && col - first < cols {
                unsafe {
                    *output.add(((col - first) * rows + row) as usize) = if wide != 0 {
                        *data.cast::<f64>().add(position as usize) as f32
                    } else {
                        // Preserve f32 signaling/quiet NaN payload ordering.
                        *data.cast::<f32>().add(position as usize)
                    };
                }
            }
        }
        row += stride;
    }
}

/// Compute statistics once from original-precision selected sparse values.
/// # Safety
/// CSR rows are reference first, then test groups described by offsets. Sums
/// and optional counts are zeroed (groups+1)*cols outputs, reference last.
#[kernel]
pub unsafe fn sparse_ovo_csr_stats(
    data: *const u8,
    columns: *const u8,
    pointers: *const u64,
    offsets: *const u64,
    sums: *mut f64,
    counts: *mut f64,
    nref: u64,
    rows: u64,
    cols: u64,
    groups: u64,
    nnz: u64,
    wide: u32,
    narrow: u32,
) {
    let lane = thread::threadIdx_x() as u64;
    let mut row = thread::blockIdx_x() as u64;
    while row < rows {
        let mut group = groups;
        if row >= nref {
            let local = row - nref;
            group = 0;
            while group < groups && unsafe { *offsets.add((group + 1) as usize) } <= local {
                group += 1;
            }
        }
        if group <= groups {
            let begin = unsafe { *pointers.add(row as usize) };
            let end = unsafe { *pointers.add((row + 1) as usize) }.min(nnz);
            let mut position = begin + lane;
            while position < end {
                let col = unsafe { column(columns, position, narrow) };
                if col < cols {
                    let value = unsafe { value(data, position, wide) };
                    let at = (group * cols + col) as usize;
                    unsafe {
                        DeviceAtomicF64::from_ptr(sums.add(at))
                            .fetch_add(value, AtomicOrdering::Relaxed);
                        if !counts.is_null() && value != 0.0 {
                            DeviceAtomicF64::from_ptr(counts.add(at))
                                .fetch_add(1.0, AtomicOrdering::Relaxed);
                        }
                    }
                }
                position += thread::blockDim_x() as u64;
            }
        }
        row += thread::gridDim_x() as u64;
    }
}

#[inline(always)]
unsafe fn bound<const UPPER: bool>(values: *const f32, mut lo: u64, mut hi: u64, v: f32) -> u64 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let item = unsafe { *values.add(mid as usize) };
        if if UPPER { item <= v } else { item < v } {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(always)]
fn cubic(n: u64) -> f64 {
    let n = n as f64;
    n * n * n - n
}

/// Original analytic-zero shared tier, with nonnegative finite inputs verified
/// by the native CSR snapshot. Sort only positive values and count group zeros.
/// # Safety
/// F-order reference is sorted; groups contains nrows*cols nonnegative,
/// non-NaN ranking values (finite original f64 values can cast to +infinity). Offsets partition groups, each at most 2500 rows. All output
/// and tie-cache dimensions are validated; launch uses 256 threads per block.
#[kernel]
pub unsafe fn sparse_ovo_csr_analytic(
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
    use cuda_device::{SharedArray, atomic::DeviceAtomicU32};
    unsafe {
        static mut VALUES: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut COUNTS: SharedArray<u32, 2> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f64, 16> = SharedArray::UNINIT;
        let values = SharedArray::as_raw_mut_ptr(&raw mut VALUES);
        let counts = SharedArray::as_raw_mut_ptr(&raw mut COUNTS);
        let scratch = SharedArray::as_raw_mut_ptr(&raw mut REDUCE);
        let tid = thread::threadIdx_x() as u64;
        let lane = tid as usize % 32;
        let warp = tid as usize / 32;
        let mut segment = thread::blockIdx_x() as u64;
        while segment < cols * ngroups {
            let col = segment % cols;
            let group = segment / cols;
            let start = *offsets.add(group as usize);
            let n = *offsets.add(group as usize + 1) - start;
            let r = reference.add((col * nref) as usize);
            let g = groups.add((col * nrows + start) as usize);
            if tid == 0 {
                *counts = 0;
                *counts.add(1) = bound::<true>(r, 0, nref, 0.0) as u32;
            }
            thread::sync_threads();
            let mut row = tid;
            while row < n {
                let v = *g.add(row as usize);
                if v > 0.0 {
                    let position =
                        DeviceAtomicU32::from_ptr(counts).fetch_add(1, AtomicOrdering::Relaxed);
                    *values.add(position as usize) = v;
                }
                row += 256;
            }
            thread::sync_threads();
            let nnz = *counts as u64;
            let ref_zeros = *counts.add(1) as u64;
            let group_zeros = n - nnz;
            let total_zeros = ref_zeros + group_zeros;
            let padded = (nnz as u32).next_power_of_two();
            row = nnz + tid;
            while row < padded as u64 {
                *values.add(row as usize) = f32::INFINITY;
                row += 256;
            }
            thread::sync_threads();
            let mut size = 2;
            while size <= padded {
                let mut distance = size / 2;
                while distance != 0 {
                    let mut index = tid as u32;
                    while index < padded {
                        let other = index ^ distance;
                        if other > index {
                            let a = *values.add(index as usize);
                            let b = *values.add(other as usize);
                            let ascending = index & size == 0;
                            if if ascending { a > b } else { a < b } {
                                *values.add(index as usize) = b;
                                *values.add(other as usize) = a;
                            }
                        }
                        index += 256;
                    }
                    thread::sync_threads();
                    distance /= 2;
                }
                size *= 2;
            }
            let mut rank = if tid == 0 {
                group_zeros as f64 * (total_zeros as f64 + 1.0) * 0.5
            } else {
                0.0
            };
            let mut delta = 0.0;
            row = tid;
            while row < nnz {
                let v = *values.add(row as usize);
                let rlo = bound::<false>(r, 0, nref, v);
                let rhi = bound::<true>(r, rlo, nref, v);
                let glo = bound::<false>(values, 0, nnz, v);
                let ghi = bound::<true>(values, glo, nnz, v);
                rank += (rlo + glo + group_zeros) as f64 + 0.5 * (rhi - rlo + ghi - glo + 1) as f64;
                if row == 0 || v != *values.add(row as usize - 1) {
                    let cg = ghi - row;
                    let cr = rhi - rlo;
                    delta += cubic(cg + cr) - cubic(cr);
                }
                row += 256;
            }
            let rank = crate::harmony::sum_f64(rank);
            let delta = crate::harmony::sum_f64(delta);
            if lane == 0 {
                *scratch.add(warp) = rank;
                *scratch.add(8 + warp) = delta;
            }
            thread::sync_threads();
            let rank = crate::harmony::sum_f64(if lane < 8 { *scratch.add(lane) } else { 0.0 });
            let delta = crate::harmony::sum_f64(if lane < 8 {
                *scratch.add(8 + lane)
            } else {
                0.0
            });
            if tid == 0 {
                let pos = (group * out_cols + col_offset + col) as usize;
                *ranks.add(pos) = rank;
                let total = nref + n;
                *ties.add(pos) = if total < 2 {
                    1.0
                } else {
                    1.0 - (*tie_base.add(col as usize) + delta + cubic(total_zeros)
                        - cubic(ref_zeros))
                        / cubic(total)
                };
            }
            thread::sync_threads();
            segment += thread::gridDim_x() as u64;
        }
    }
}
