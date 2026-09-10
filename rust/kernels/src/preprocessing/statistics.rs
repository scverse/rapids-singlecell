//! Typed sparse statistics with the original floating-point reduction tree.
#![allow(clippy::too_many_arguments, static_mut_refs)]

use cuda_device::{SharedArray, kernel, thread, warp};

macro_rules! major_statistics {
    ($name:ident, $nan:literal, $value:ty, $index:ty) => {
        /// Per-row moments or masked finite sums and integer NaN counts.
        /// # Safety
        /// Launch exactly 64 threads. Inputs and outputs have the validated
        /// extents and are disjoint; unused mode-specific pointers may be null.
        #[kernel]
        pub unsafe fn $name(
            indptr: *const $index,
            index: *const $index,
            data: *const $value,
            sums: *mut f64,
            squares: *mut f64,
            nans: *mut i32,
            mask: *const u8,
            major: u64,
            minor: u64,
            nnz: u64,
        ) {
            static mut UPPER_SUM: SharedArray<f64, 32> = SharedArray::UNINIT;
            static mut UPPER_SQUARES: SharedArray<f64, 32> = SharedArray::UNINIT;
            static mut UPPER_NANS: SharedArray<u32, 32> = SharedArray::UNINIT;
            let lane = thread::threadIdx_x() as usize;
            let mut row = thread::blockIdx_x() as u64;
            while row < major {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                let mut sum = 0.0_f64;
                let mut square = 0.0_f64;
                let mut count = 0_u32;
                if start >= 0 && stop >= start && stop as u64 <= nnz {
                    let mut p = start as u64 + lane as u64;
                    while p < stop as u64 {
                        let selected = if $nan {
                            let col = unsafe { *index.add(p as usize) } as i64;
                            col >= 0
                                && (col as u64) < minor
                                && unsafe { *mask.add(col as usize) != 0 }
                        } else {
                            true
                        };
                        if selected {
                            let value = unsafe { *data.add(p as usize) } as f64;
                            if $nan && value.is_nan() {
                                count = count.wrapping_add(1);
                            } else {
                                sum += value;
                                if !$nan {
                                    square = value.mul_add(value, square);
                                }
                            }
                        }
                        p += 64;
                    }
                }
                if lane >= 32 {
                    unsafe {
                        UPPER_SUM[lane - 32] = sum;
                        if $nan {
                            UPPER_NANS[lane - 32] = count;
                        } else {
                            UPPER_SQUARES[lane - 32] = square;
                        }
                    }
                }
                thread::sync_threads();
                if lane < 32 {
                    // Combine lanes separated by 32 first, then 16,8,4,2,1.
                    // Reducing each warp first would change cancellation and
                    // lose fidelity with the original 64-thread shared tree.
                    unsafe {
                        sum += UPPER_SUM[lane];
                        if $nan {
                            count = count.wrapping_add(UPPER_NANS[lane]);
                        } else {
                            square += UPPER_SQUARES[lane];
                        }
                    }
                    let mut offset = 16;
                    while offset > 0 {
                        // Every lane participates in the shuffle, but only
                        // the original lower half performs floating-point
                        // additions. Unused upper-lane work is costly on GPUs
                        // with fewer double-precision execution units.
                        let other_sum = warp::shuffle_down_f64_sync(u32::MAX, sum, offset);
                        if $nan {
                            let other_count = warp::shuffle_down_sync(u32::MAX, count, offset);
                            if lane < offset as usize {
                                sum += other_sum;
                                count = count.wrapping_add(other_count);
                            }
                        } else {
                            let other_square =
                                warp::shuffle_down_f64_sync(u32::MAX, square, offset);
                            if lane < offset as usize {
                                sum += other_sum;
                                square += other_square;
                            }
                        }
                        offset /= 2;
                    }
                    if lane == 0 {
                        unsafe {
                            *sums.add(row as usize) = sum;
                            if $nan {
                                *nans.add(row as usize) = count as i32;
                            } else {
                                *squares.add(row as usize) = square;
                            }
                        }
                    }
                }
                row += thread::gridDim_x() as u64;
                // Only a subsequent grid-stride iteration can overwrite the
                // upper-warp storage. This condition is block-uniform.
                if row < major {
                    thread::sync_threads();
                }
            }
        }
    };
}

major_statistics!(prep_stats_major_f32_i32, false, f32, i32);
major_statistics!(prep_stats_major_f32_i64, false, f32, i64);
major_statistics!(prep_stats_major_f64_i32, false, f64, i32);
major_statistics!(prep_stats_major_f64_i64, false, f64, i64);
major_statistics!(prep_nan_major_f32_i32, true, f32, i32);
major_statistics!(prep_nan_major_f32_i64, true, f32, i64);
major_statistics!(prep_nan_major_f64_i32, true, f64, i32);
major_statistics!(prep_nan_major_f64_i64, true, f64, i64);

macro_rules! cell_statistics {
    ($name:ident, $value:ty, $index:ty) => {
        /// Sum every stored value and count entries, including explicit zeros.
        /// # Safety
        /// Arrays have validated extents and outputs are disjoint from inputs.
        #[kernel]
        pub unsafe fn $name(
            indptr: *const $index,
            data: *const $value,
            sums: *mut $value,
            counts: *mut i32,
            rows: u64,
            nnz: u64,
        ) {
            let mut row = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
                + thread::threadIdx_x() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while row < rows {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                let mut sum = 0.0 as $value;
                let mut count = 0;
                if start >= 0 && stop >= start && stop as u64 <= nnz {
                    let mut p = start as u64;
                    #[unroll(4)]
                    while p < stop as u64 {
                        sum += unsafe { *data.add(p as usize) };
                        p += 1;
                    }
                    count = (stop - start) as i32;
                }
                unsafe {
                    *sums.add(row as usize) = sum;
                    *counts.add(row as usize) = count;
                }
                row += stride;
            }
        }
    };
}

cell_statistics!(prep_qc_cells_f32_i32, f32, i32);
cell_statistics!(prep_qc_cells_f32_i64, f32, i64);
cell_statistics!(prep_qc_cells_f64_i32, f64, i32);
cell_statistics!(prep_qc_cells_f64_i64, f64, i64);
