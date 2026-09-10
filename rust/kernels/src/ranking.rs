//! Ranking, group reductions, histograms and sparse window extraction.
//!
//! Every data-dependent sparse offset is checked on the device before use.
#![allow(clippy::too_many_arguments)]
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64, DeviceAtomicU32};
use cuda_device::{kernel, thread};

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
unsafe fn put_index(p: *mut u8, i: u64, v: i64, wide: u32) {
    unsafe {
        if wide != 0 {
            *p.cast::<i64>().add(i as usize) = v;
        } else {
            *p.cast::<i32>().add(i as usize) = v as i32;
        }
    }
}

#[inline(always)]
unsafe fn lower_column(
    indices: *const u8,
    mut low: u64,
    mut high: u64,
    column: i64,
    wide: u32,
) -> u64 {
    while low < high {
        let middle = low + (high - low) / 2;
        if unsafe { index(indices, middle, wide) } < column {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}
#[inline(always)]
unsafe fn add(p: *mut f64, v: f64) {
    unsafe { DeviceAtomicF64::from_ptr(p) }.fetch_add(v, AtomicOrdering::Relaxed);
}
#[cuda_device::device]
fn rank_stride() -> u64 {
    thread::blockDim_x() as u64 * thread::gridDim_x() as u64
}

/// # Safety
/// `values` contains rows*cols f64 elements and is exclusively writable.
#[kernel]
pub unsafe fn rank_bh(values: *mut f64, rows: u64, cols: u64) {
    let mut row = thread::index_1d().get() as u64;
    while row < rows {
        let mut running = 1.0_f64;
        let mut col = cols;
        while col > 0 {
            col -= 1;
            let p = unsafe { values.add((row * cols + col) as usize) };
            let v = unsafe { *p };
            if v < running {
                running = v;
            }
            unsafe { *p = running };
        }
        row += rank_stride();
    }
}

/// # Safety
/// All optional non-null outputs have groups*out_cols elements (totals out_cols).
/// Input is contiguous with rows*cols elements and codes/mask have rows elements.
#[inline(always)]
unsafe fn rank_stats_impl<const COLUMNS: bool>(
    x: *const u8,
    codes: *const i32,
    mask: *const u8,
    sums: *mut f64,
    squares: *mut f64,
    nnz: *mut f64,
    total: *mut f64,
    total_nnz: *mut f64,
    rows: u64,
    cols: u64,
    groups: u64,
    out_cols: u64,
    col_offset: u64,
    wide: u32,
    c_order: u32,
) {
    let mut work = if COLUMNS {
        thread::blockIdx_x() as u64
    } else {
        thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64
    };
    let bound = if COLUMNS { cols } else { rows * cols };
    while work < bound {
        let mut row_in_column = thread::threadIdx_x() as u64;
        loop {
            if COLUMNS && row_in_column >= rows {
                break;
            }
            let i = if COLUMNS && c_order & 1 != 0 {
                row_in_column * cols + work
            } else if COLUMNS {
                work * rows + row_in_column
            } else {
                work
            };
            let (row, col) = if COLUMNS {
                (row_in_column, work)
            } else if c_order & 1 != 0 {
                (i / cols, i % cols)
            } else {
                (i % rows, i / rows)
            };
            if mask.is_null() || unsafe { *mask.add(row as usize) } != 0 {
                let v = unsafe { value(x, i, wide) };
                // Bit one is used only by host aggregation; rank statistics
                // retain their original treatment of zero-valued inputs.
                if c_order & 2 == 0 || v != 0.0 {
                    let g = unsafe { *codes.add(row as usize) };
                    if g >= 0 && (g as u64) < groups {
                        let o = (g as u64 * out_cols + col_offset + col) as usize;
                        unsafe {
                            if !sums.is_null() {
                                add(sums.add(o), v);
                            }
                            if !squares.is_null() {
                                add(squares.add(o), v * v);
                            }
                            if !nnz.is_null() && v != 0.0 {
                                add(nnz.add(o), 1.0);
                            }
                        }
                    }
                    let o = (col_offset + col) as usize;
                    unsafe {
                        if !total.is_null() {
                            add(total.add(o), v);
                        }
                        if !total_nnz.is_null() && v != 0.0 {
                            add(total_nnz.add(o), 1.0);
                        }
                    }
                }
            }
            if !COLUMNS {
                break;
            }
            row_in_column += thread::blockDim_x() as u64;
        }
        work += if COLUMNS {
            thread::gridDim_x() as u64
        } else {
            rank_stride()
        };
    }
}

macro_rules! stats_entry {
    ($name:ident, $columns:expr) => {
        /// Accumulate validated per-group statistics in float64.
        /// # Safety
        /// Same pointer, extent and aliasing requirements as rank_stats_impl.
        #[kernel]
        pub unsafe fn $name(
            x: *const u8,
            codes: *const i32,
            mask: *const u8,
            sums: *mut f64,
            squares: *mut f64,
            nnz: *mut f64,
            total: *mut f64,
            total_nnz: *mut f64,
            rows: u64,
            cols: u64,
            groups: u64,
            out_cols: u64,
            col_offset: u64,
            wide: u32,
            c_order: u32,
        ) {
            unsafe {
                rank_stats_impl::<$columns>(
                    x, codes, mask, sums, squares, nnz, total, total_nnz, rows, cols, groups,
                    out_cols, col_offset, wide, c_order,
                )
            };
        }
    };
}
stats_entry!(rank_stats, false);
stats_entry!(rank_stats_columns, true);

/// # Safety
/// Valid compressed sparse arrays; indptr has major+1 entries, index/data nnz.
/// Out is pre-zeroed F-order rows*window. Type flags match pointer types.
#[kernel]
pub unsafe fn rank_tile(
    indptr: *const u8,
    indices: *const u8,
    data: *const u8,
    out: *mut f64,
    rows: u64,
    window: u64,
    lb: u64,
    major: u64,
    nnz: u64,
    pwide: u32,
    iwide: u32,
    wide: u32,
    csc: u32,
) {
    let width = if csc != 0 { 128 } else { 1 };
    let id = thread::index_1d().get() as u64;
    let lane = id % width;
    let mut seg = id / width;
    let segments = if csc != 0 { window } else { rows };
    while seg < segments {
        let major_i = if csc != 0 { seg + lb } else { seg };
        if major_i < major {
            let a = unsafe { index(indptr, major_i, pwide) };
            let b = unsafe { index(indptr, major_i + 1, pwide) };
            if a >= 0 && b >= a && b as u64 <= nnz {
                let mut p = a as u64 + lane;
                while p < b as u64 {
                    let ix = unsafe { index(indices, p, iwide) };
                    let (row, col) = if csc != 0 {
                        (ix, seg as i64)
                    } else {
                        (seg as i64, ix - lb as i64)
                    };
                    if row >= 0 && (row as u64) < rows && col >= 0 && (col as u64) < window {
                        unsafe {
                            add(
                                out.add((col as u64 * rows + row as u64) as usize),
                                value(data, p, wide),
                            );
                        }
                    }
                    p += width;
                }
            }
        }
        seg += rank_stride() / width;
    }
}

/// # Safety
/// tie contains cols initialized tie-sum values.
#[kernel]
pub unsafe fn rank_tie_finish(tie: *mut f64, rows: u64, cols: u64) {
    let mut i = thread::index_1d().get() as u64;
    let n = rows as f64;
    let d = n * n * n - n;
    while i < cols {
        let p = unsafe { tie.add(i as usize) };
        unsafe {
            *p = if rows < 2 { 1.0 } else { 1.0 - *p / d };
        }
        i += rank_stride();
    }
}

/// # Safety
/// Dense F-order input rows*cols; histogram has cols*groups*(bins+1) entries.
#[kernel]
pub unsafe fn rank_hist_dense(
    x: *const u8,
    codes: *const i32,
    hist: *mut u32,
    rows: u64,
    cols: u64,
    groups: u64,
    bins: u64,
    low: f64,
    inverse: f64,
    wide: u32,
    skip_zero: u32,
) {
    // Keep each column's contended histogram on one block, as in the original
    // kernel. Flattening the matrix launches many competing blocks per gene.
    let mut col = thread::blockIdx_x() as u64;
    while col < cols {
        let mut row = thread::threadIdx_x() as u64;
        while row < rows {
            let g = unsafe { *codes.add(row as usize) };
            let v = unsafe { value(x, col * rows + row, wide) };
            if g >= 0 && (g as u64) < groups && (skip_zero == 0 || v != 0.0) {
                let bin = (((v - low) * inverse) as i64).max(0).min(bins as i64 - 1) as u64 + 1;
                unsafe {
                    DeviceAtomicU32::from_ptr(
                        hist.add(((col * groups + g as u64) * (bins + 1) + bin) as usize),
                    )
                }
                .fetch_add(1, AtomicOrdering::Relaxed);
            }
            row += thread::blockDim_x() as u64;
        }
        col += thread::gridDim_x() as u64;
    }
}

/// # Safety
/// Validated compressed arrays as rank_tile; histogram has cols*groups*(bins+1).
#[kernel]
pub unsafe fn rank_hist_sparse(
    data: *const u8,
    indices: *const u8,
    indptr: *const u8,
    codes: *const i32,
    hist: *mut u32,
    rows: u64,
    cols: u64,
    groups: u64,
    bins: u64,
    low: f64,
    inverse: f64,
    start: u64,
    nnz: u64,
    wide: u32,
    iwide: u32,
    csc: u32,
    full_block: u32,
) {
    let id = thread::index_1d().get() as u64;
    let width = if full_block != 0 {
        thread::blockDim_x() as u64
    } else {
        32
    };
    let (lane, mut seg, stride) = if full_block != 0 {
        (
            thread::threadIdx_x() as u64,
            thread::blockIdx_x() as u64,
            thread::gridDim_x() as u64,
        )
    } else {
        (id % 32, id / 32, rank_stride() / 32)
    };
    let major = if csc != 0 { cols } else { rows };
    while seg < major {
        let source = if csc != 0 { seg + start } else { seg };
        let a = unsafe { index(indptr, source, iwide) };
        let b = unsafe { index(indptr, source + 1, iwide) };
        if a >= 0 && b >= a && b as u64 <= nnz {
            let mut p = a as u64 + lane;
            while p < b as u64 {
                let ix = unsafe { index(indices, p, iwide) };
                let (row, col) = if csc != 0 {
                    (ix, seg as i64)
                } else {
                    (seg as i64, ix - start as i64)
                };
                if row >= 0 && (row as u64) < rows && col >= 0 && (col as u64) < cols {
                    let g = unsafe { *codes.add(row as usize) };
                    let v = unsafe { value(data, p, wide) };
                    if g >= 0 && (g as u64) < groups && v != 0.0 {
                        let bin =
                            (((v - low) * inverse) as i64).max(0).min(bins as i64 - 1) as u64 + 1;
                        unsafe {
                            DeviceAtomicU32::from_ptr(hist.add(
                                ((col as u64 * groups + g as u64) * (bins + 1) + bin) as usize,
                            ))
                        }
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                }
                p += width;
            }
        }
        seg += stride;
    }
}

/// # Safety
/// indptr/local contain rows+1 indices; indices/data nnz; output capacity local_nnz.
/// Counting mode writes row counts at local[row+1]; gather uses scanned offsets.
/// Row indices must be sorted, as in the canonical CSR inputs of the public API.
#[kernel]
pub unsafe fn rank_csr_range(
    indices: *const u8,
    indptr: *const u8,
    data: *const u8,
    local: *mut u8,
    out_data: *mut u8,
    out_indices: *mut u8,
    rows: u64,
    nnz: u64,
    local_nnz: u64,
    start: u64,
    stop: u64,
    iwide: u32,
    wide: u32,
    gather: u32,
) {
    let id = thread::index_1d().get() as u64;
    let width = if gather != 0 { 32 } else { 1 };
    let lane = id % width;
    let mut row = id / width;
    if id == 0 && gather == 0 {
        unsafe {
            put_index(local, 0, 0, iwide);
        }
    }
    while row < rows {
        let a = unsafe { index(indptr, row, iwide) };
        let b = unsafe { index(indptr, row + 1, iwide) };
        let mut count = 0i64;
        let offset = if gather != 0 {
            unsafe { index(local, row, iwide) }
        } else {
            0
        };
        if a >= 0 && b >= a && b as u64 <= nnz {
            let first = unsafe { lower_column(indices, a as u64, b as u64, start as i64, iwide) };
            let last = unsafe { lower_column(indices, first, b as u64, stop as i64, iwide) };
            count = (last - first) as i64;
            if gather != 0 {
                let mut p = first + lane;
                while p < last {
                    let col = unsafe { index(indices, p, iwide) };
                    let dest = offset + (p - first) as i64;
                    if gather != 0 && dest >= 0 && (dest as u64) < local_nnz {
                        unsafe {
                            put_index(out_indices, dest as u64, col - start as i64, iwide);
                            if wide != 0 {
                                *out_data.cast::<f64>().add(dest as usize) =
                                    *data.cast::<f64>().add(p as usize);
                            } else {
                                *out_data.cast::<f32>().add(dest as usize) =
                                    *data.cast::<f32>().add(p as usize);
                            }
                        }
                    }
                    p += width;
                }
            }
        }
        if gather == 0 {
            unsafe {
                put_index(local, row + 1, count, iwide);
            }
        }
        row += rank_stride() / width;
    }
}
