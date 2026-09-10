//! Native bounded CSR-to-CSC windows for device sparse ranking.
#![allow(clippy::too_many_arguments)]
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU64},
    kernel, thread,
};

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

/// # Safety
/// Indptr spans rows+1 entries, indices spans nnz, counts spans cols and is zero.
/// Invalid row spans and column indices are skipped before pointer access.
#[kernel]
pub unsafe fn sparse_csr_histogram(
    indices: *const u8,
    indptr: *const u8,
    counts: *mut u64,
    rows: u64,
    cols: u64,
    nnz: u64,
    index_wide: u32,
    pointer_wide: u32,
) {
    let mut row = thread::index_1d().get() as u64;
    let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
    while row < rows {
        let first = unsafe { index(indptr, row, pointer_wide) };
        let stop = unsafe { index(indptr, row + 1, pointer_wide) };
        if first >= 0 && stop >= first && stop as u64 <= nnz {
            for position in first as u64..stop as u64 {
                let col = unsafe { index(indices, position, index_wide) };
                if col >= 0 && (col as u64) < cols {
                    unsafe { DeviceAtomicU64::from_ptr(counts.add(col as usize)) }
                        .fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
        }
        row += stride;
    }
}

/// # Safety
/// Input buffers follow the histogram contract. Positions contains one mutable
/// prefix counter per requested column; outputs each have capacity entries.
#[kernel]
pub unsafe fn sparse_csr_scatter(
    data: *const u8,
    indices: *const u8,
    indptr: *const u8,
    positions: *mut u64,
    output_data: *mut u8,
    output_indices: *mut u8,
    rows: u64,
    first_col: u64,
    stop_col: u64,
    nnz: u64,
    capacity: u64,
    data_wide: u32,
    index_wide: u32,
    pointer_wide: u32,
    row_wide: u32,
) {
    let mut row = thread::index_1d().get() as u64;
    let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
    while row < rows {
        let first = unsafe { index(indptr, row, pointer_wide) };
        let stop = unsafe { index(indptr, row + 1, pointer_wide) };
        if first >= 0 && stop >= first && stop as u64 <= nnz {
            for position in first as u64..stop as u64 {
                let col = unsafe { index(indices, position, index_wide) };
                if col >= 0 && (col as u64) >= first_col && (col as u64) < stop_col {
                    let destination = unsafe {
                        DeviceAtomicU64::from_ptr(positions.add((col as u64 - first_col) as usize))
                    }
                    .fetch_add(1, AtomicOrdering::Relaxed);
                    if destination < capacity {
                        unsafe {
                            if data_wide != 0 {
                                *output_data.cast::<f64>().add(destination as usize) =
                                    *data.cast::<f64>().add(position as usize);
                            } else {
                                *output_data.cast::<f32>().add(destination as usize) =
                                    *data.cast::<f32>().add(position as usize);
                            }
                            if row_wide != 0 {
                                *output_indices.cast::<i64>().add(destination as usize) =
                                    row as i64;
                            } else {
                                *output_indices.cast::<i32>().add(destination as usize) =
                                    row as i32;
                            }
                        }
                    }
                }
            }
        }
        row += stride;
    }
}
