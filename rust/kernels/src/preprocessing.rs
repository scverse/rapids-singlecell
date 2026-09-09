//! GPU reductions and transformations used by preprocessing.
//!
//! Compressed sparse metadata is checked on the device before it is dereferenced.
//! Host wrappers validate allocation bounds, dtype, shape, and device ownership.
#![allow(clippy::too_many_arguments)]
#![allow(static_mut_refs)] // Block-local SharedArray access is synchronized below.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32};
use cuda_device::{SharedArray, kernel, thread, warp};

#[inline(always)]
unsafe fn add_i32(out: *mut i32, value: i32) {
    unsafe { DeviceAtomicI32::from_ptr(out) }.fetch_add(value, AtomicOrdering::Relaxed);
}
#[inline(always)]
unsafe fn add_f32(out: *mut f32, value: f32) {
    unsafe { DeviceAtomicF32::from_ptr(out) }.fetch_add(value, AtomicOrdering::Relaxed);
}
#[inline(always)]
unsafe fn add_f64(out: *mut f64, value: f64) {
    unsafe { DeviceAtomicF64::from_ptr(out) }.fetch_add(value, AtomicOrdering::Relaxed);
}

// These baseline PTX instructions are available on Turing. The pinned
// cuda-oxide generated wrappers conservatively require sm_80, so use their
// identical instructions directly to retain the sm_75 deployment target.
#[inline(always)]
fn rsqrt_approx_f32(value: f32) -> f32 {
    let result: f32;
    unsafe {
        cuda_device::ptx_asm!("rsqrt.approx.f32 %0, %1;", out("=f") result, in("f") value);
    }
    result
}

#[inline(always)]
fn rsqrt_approx_f64(value: f64) -> f64 {
    let result: f64;
    unsafe {
        cuda_device::ptx_asm!("rsqrt.approx.f64 %0, %1;", out("=d") result, in("d") value);
    }
    result
}

macro_rules! sparse_kernels {
    ($stats_major:ident, $stats_minor:ident, $norm:ident, $scale:ident, $qc:ident,
     $residual:ident, $value:ty, $index:ty, $add:ident, $reduce:ident, $rsqrt:ident, $fma:ident) => {
        /// Sum and sum of squares, or masked NaN-aware sum and NaN count.
        /// # Safety
        /// Host-validated arrays; launch exactly 64 threads per block. `mode`
        /// selects moments (0) or masked NaN statistics (1).
        #[kernel]
        pub unsafe fn $stats_major(
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
            mode: u32,
        ) {
            static mut SUM: SharedArray<f64, 64> = SharedArray::UNINIT;
            static mut SECOND: SharedArray<f64, 64> = SharedArray::UNINIT;
            let lane = thread::threadIdx_x() as usize;
            let mut row = thread::blockIdx_x() as u64;
            while row < major {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                let mut sum = 0.0_f64;
                let mut second = 0.0_f64;
                if start >= 0 && stop >= start && stop as u64 <= nnz {
                    let mut p = start as u64 + lane as u64;
                    while p < stop as u64 {
                        let mut selected = true;
                        if mode == 1 {
                            let col = unsafe { *index.add(p as usize) } as i64;
                            selected = col >= 0
                                && (col as u64) < minor
                                && unsafe { *mask.add(col as usize) != 0 };
                        }
                        if selected {
                            let v = unsafe { *data.add(p as usize) } as f64;
                            if mode == 0 {
                                sum += v;
                                second += v * v;
                            } else if v.is_nan() {
                                second += 1.0;
                            } else {
                                sum += v;
                            }
                        }
                        p += 64;
                    }
                }
                unsafe {
                    SUM[lane] = sum;
                    SECOND[lane] = second;
                }
                thread::sync_threads();
                let mut offset = 32;
                while offset > 0 {
                    if lane < offset {
                        unsafe {
                            SUM[lane] += SUM[lane + offset];
                            SECOND[lane] += SECOND[lane + offset];
                        }
                    }
                    thread::sync_threads();
                    offset /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *sums.add(row as usize) = SUM[0];
                        if mode == 0 {
                            *squares.add(row as usize) = SECOND[0];
                        } else {
                            *nans.add(row as usize) = SECOND[0] as i32;
                        }
                    }
                }
                thread::sync_threads();
                row += thread::gridDim_x() as u64;
            }
        }

        /// Scatter moments, NaN statistics, or sparse QC counts by minor index.
        /// # Safety
        /// Arrays have the lengths validated by the host; updates are atomic.
        #[kernel]
        pub unsafe fn $stats_minor(
            index: *const $index,
            data: *const $value,
            sums: *mut f64,
            squares: *mut f64,
            nans: *mut i32,
            mask: *const u8,
            qc_sums: *mut $value,
            nnz: u64,
            minor: u64,
            mode: u32,
        ) {
            let mut p = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while p < nnz {
                let col = unsafe { *index.add(p as usize) } as i64;
                if col >= 0 && (col as u64) < minor {
                    let c = col as usize;
                    let v = unsafe { *data.add(p as usize) };
                    unsafe {
                        if mode == 0 {
                            add_f64(sums.add(c), v as f64);
                            add_f64(squares.add(c), (v as f64) * (v as f64));
                        } else if mode == 1 {
                            if *mask.add(c) != 0 {
                                if v.is_nan() {
                                    add_i32(nans.add(c), 1);
                                } else {
                                    add_f64(sums.add(c), v as f64);
                                }
                            }
                        } else {
                            $add(qc_sums.add(c), v);
                            add_i32(nans.add(c), 1);
                        }
                    }
                }
                p += stride;
            }
        }

        /// Sparse normalization and row sums, using one warp per row.
        /// # Safety
        /// Host validates compressed storage. Launch 32 threads per block.
        /// Modes: normalize=0, sum=1, masked normalize=2, masked sum=3,
        /// identify high genes=4, precomputed scaling=5. Mode 4 receives an
        /// aligned i32 flag array through `mask`; other masks contain bytes.
        #[kernel]
        pub unsafe fn $norm(
            indptr: *const $index,
            index: *const $index,
            data: *mut $value,
            mask: *mut u8,
            sums: *mut $value,
            scales: *const $value,
            major: u64,
            minor: u64,
            nnz: u64,
            scalar: f64,
            mode: u32,
        ) {
            let lane = thread::threadIdx_x() as u64;
            let mut row = thread::blockIdx_x() as u64;
            while row < major {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                if start >= 0 && stop >= start && stop as u64 <= nnz {
                    let mut total = 0.0 as $value;
                    let mut p = start as u64 + lane;
                    if mode != 5 {
                        while p < stop as u64 {
                            let mut selected = true;
                            if mode == 2 || mode == 3 {
                                let col = unsafe { *index.add(p as usize) } as i64;
                                selected = col >= 0
                                    && (col as u64) < minor
                                    && unsafe { *mask.add(col as usize) == 0 };
                            }
                            if selected {
                                total += unsafe { *data.add(p as usize) };
                            }
                            p += 32;
                        }
                        total = warp::$reduce(total);
                    }
                    if mode == 1 || mode == 3 {
                        if lane == 0 {
                            unsafe {
                                *sums.add(row as usize) = total;
                            }
                        }
                    } else {
                        let factor = if mode == 5 {
                            unsafe { *scales.add(row as usize) }
                        } else if mode == 4 {
                            (scalar as $value) * total
                        } else if total > 0.0 {
                            (scalar as $value) / total
                        } else {
                            0.0
                        };
                        p = start as u64 + lane;
                        while p < stop as u64 {
                            unsafe {
                                if mode == 4 {
                                    let col = *index.add(p as usize) as i64;
                                    if col >= 0
                                        && (col as u64) < minor
                                        && *data.add(p as usize) > factor
                                    {
                                        DeviceAtomicI32::from_ptr(
                                            mask.cast::<i32>().add(col as usize),
                                        )
                                        .fetch_or(1, AtomicOrdering::Relaxed);
                                    }
                                } else if mode == 5 || factor > 0.0 {
                                    *data.add(p as usize) *= factor;
                                }
                            }
                            p += 32;
                        }
                    }
                }
                row += thread::gridDim_x() as u64;
            }
        }

        /// Sparse scaling (CSC columns or masked CSR rows).
        /// # Safety
        /// Host validates dtype/shape and launch lengths; invalid spans/indices are skipped.
        #[kernel]
        pub unsafe fn $scale(
            indptr: *const $index,
            index: *const $index,
            data: *mut $value,
            std: *const $value,
            mask: *const i32,
            major: u64,
            minor: u64,
            nnz: u64,
            clip: f64,
            mode: u32,
        ) {
            let mut row = thread::blockIdx_x() as u64;
            while row < major {
                if mode == 0 || unsafe { *mask.add(row as usize) != 0 } {
                    let start = unsafe { *indptr.add(row as usize) } as i64;
                    let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                    if start >= 0 && stop >= start && stop as u64 <= nnz {
                        let mut p = start as u64 + thread::threadIdx_x() as u64;
                        while p < stop as u64 {
                            let col = if mode == 0 {
                                row as i64
                            } else {
                                (unsafe { *index.add(p as usize) }) as i64
                            };
                            if col >= 0 && (col as u64) < minor {
                                unsafe {
                                    let value = if mode == 0 {
                                        *data.add(p as usize)
                                            * ((1.0 as $value) / *std.add(col as usize))
                                    } else {
                                        *data.add(p as usize) / *std.add(col as usize)
                                    };
                                    *data.add(p as usize) = if mode == 0 || value < clip as $value {
                                        value
                                    } else {
                                        clip as $value
                                    };
                                }
                            }
                            p += thread::blockDim_x() as u64;
                        }
                    }
                }
                row += thread::gridDim_x() as u64;
            }
        }

        /// Compressed sparse QC, subset sums, and Pearson marginal sums.
        /// # Safety
        /// Outputs are disjoint from inputs. Modes: full=0, CSC subset=1,
        /// CSR subset=2, major-only QC=3, marginal sums=4.
        #[kernel]
        pub unsafe fn $qc(
            indptr: *const $index,
            index: *const $index,
            data: *const $value,
            major_sums: *mut $value,
            minor_sums: *mut $value,
            major_ex: *mut i32,
            minor_ex: *mut i32,
            mask: *const u8,
            major: u64,
            minor: u64,
            nnz: u64,
            mode: u32,
        ) {
            let mut row = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while row < major {
                if mode != 1 || unsafe { *mask.add(row as usize) != 0 } {
                    let start = unsafe { *indptr.add(row as usize) } as i64;
                    let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                    let mut sum = 0.0 as $value;
                    let mut count = 0_i32;
                    if start >= 0 && stop >= start && stop as u64 <= nnz {
                        let mut p = start as u64;
                        while p < stop as u64 {
                            let col = unsafe { *index.add(p as usize) } as i64;
                            if mode == 3 || (col >= 0 && (col as u64) < minor) {
                                unsafe {
                                    let value = *data.add(p as usize);
                                    if mode != 2 || *mask.add(col as usize) != 0 {
                                        sum += value;
                                    }
                                    count += 1;
                                    if mode == 0 || mode == 1 || mode == 4 {
                                        $add(minor_sums.add(col as usize), value);
                                    }
                                    if mode == 0 {
                                        add_i32(minor_ex.add(col as usize), 1);
                                    }
                                }
                            }
                            p += 1;
                        }
                    }
                    unsafe {
                        if mode != 1 {
                            *major_sums.add(row as usize) = sum;
                        }
                        if mode == 0 || mode == 3 {
                            *major_ex.add(row as usize) = count;
                        }
                    }
                }
                row += stride;
            }
        }

        /// Sparse clipped Pearson residuals and their per-gene variance.
        /// # Safety
        /// Host validates output size and compressed storage; rows must be sorted.
        /// Modes: CSR=0, CSC=1, CSC residual variance=2.
        #[kernel]
        pub unsafe fn $residual(
            indptr: *const $index,
            index: *const $index,
            data: *const $value,
            cells: *const $value,
            genes: *const $value,
            output: *mut $value,
            n_cells: u64,
            n_genes: u64,
            nnz: u64,
            inv_sum: f64,
            clip: f64,
            inv_theta: f64,
            mode: u32,
        ) {
            let major = if mode == 0 { n_cells } else { n_genes };
            let minor = if mode == 0 { n_genes } else { n_cells };
            let mut row = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while row < major {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                let valid = start >= 0 && stop >= start && stop as u64 <= nnz;
                let mut p = if valid { start as u64 } else { 0 };
                let end = if valid { stop as u64 } else { 0 };
                let mut mean = 0.0 as $value;
                let mut m2 = 0.0 as $value;
                let mut col = 0_u64;
                while col < minor {
                    let (cell, gene) = if mode == 0 { (row, col) } else { (col, row) };
                    let offset = (cell * n_genes + gene) as usize;
                    let mut value = if mode == 2 {
                        0.0
                    } else {
                        unsafe { *output.add(offset) }
                    };
                    if p < end && unsafe { *index.add(p as usize) } as i64 == col as i64 {
                        value += unsafe { *data.add(p as usize) };
                        p += 1;
                    }
                    let mu = unsafe { *genes.add(gene as usize) * *cells.add(cell as usize) }
                        * inv_sum as $value;
                    let mut x = (value - mu) * $rsqrt(mu + mu * mu * inv_theta as $value);
                    if mode == 2 {
                        x = x.max(-(clip as $value)).min(clip as $value);
                    } else {
                        if x < -(clip as $value) {
                            x = -(clip as $value);
                        }
                        if x > clip as $value {
                            x = clip as $value;
                        }
                    }
                    if mode == 2 {
                        let delta = x - mean;
                        mean += delta / (cell + 1) as $value;
                        m2 = delta.mul_add(x - mean, m2);
                    } else {
                        unsafe {
                            *output.add(offset) = x;
                        }
                    }
                    col += 1;
                }
                if mode == 2 {
                    unsafe {
                        *output.add(row as usize) = m2 / n_cells as $value;
                    }
                }
                row += stride;
            }
        }
    };
}

macro_rules! dense_kernels {
    ($norm:ident, $scale:ident, $qc:ident, $expected:ident, $residual:ident,
     $value:ty, $add:ident, $reduce:ident, $rsqrt:ident, $fma:ident) => {
        /// Dense row normalization or multiplication by supplied row factors.
        /// # Safety
        /// Host validates `rows * cols` and scales; launch 32 threads per block.
        #[kernel]
        pub unsafe fn $norm(
            data: *mut $value,
            scales: *const $value,
            rows: u64,
            cols: u64,
            scalar: f64,
            mode: u32,
        ) {
            let lane = thread::threadIdx_x() as u64;
            let mut row = thread::blockIdx_x() as u64;
            while row < rows {
                let mut sum = 0.0 as $value;
                let mut col = lane;
                if mode == 0 {
                    while col < cols {
                        sum += unsafe { *data.add((row * cols + col) as usize) };
                        col += 32;
                    }
                    sum = warp::$reduce(sum);
                }
                let factor = if mode == 1 {
                    unsafe { *scales.add(row as usize) }
                } else if sum > 0.0 {
                    scalar as $value / sum
                } else {
                    0.0
                };
                if mode == 1 || factor > 0.0 {
                    col = lane;
                    while col < cols {
                        unsafe {
                            *data.add((row * cols + col) as usize) *= factor;
                        }
                        col += 32;
                    }
                }
                row += thread::gridDim_x() as u64;
            }
        }

        /// Masked dense scaling, optionally centering and symmetrically clipping.
        /// # Safety
        /// Host validates all arrays and dimensions. Every output has one writer.
        #[kernel]
        pub unsafe fn $scale(
            data: *mut $value,
            mean: *const $value,
            std: *const $value,
            mask: *const i32,
            rows: u64,
            cols: u64,
            clip: f64,
            center: u32,
        ) {
            let mut p = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while p < rows * cols {
                let row = p / cols;
                let col = p % cols;
                if unsafe { *mask.add(row as usize) != 0 } {
                    unsafe {
                        let mut value = *data.add(p as usize);
                        if center != 0 {
                            value -= *mean.add(col as usize);
                        }
                        value /= *std.add(col as usize);
                        if center != 0 {
                            if value > clip as $value {
                                value = clip as $value;
                            }
                            if value < -(clip as $value) {
                                value = -(clip as $value);
                            }
                        } else {
                            // Keep the legacy conditional's NaN behavior.
                            value = if value < clip as $value {
                                value
                            } else {
                                clip as $value
                            };
                        }
                        *data.add(p as usize) = value;
                    }
                }
                p += stride;
            }
        }

        /// Dense QC or masked cell sums. Dense QC counts strictly positive values.
        /// # Safety
        /// Host validates all lengths. Modes: full=0, subset=1, cells=2, genes=3.
        #[kernel]
        pub unsafe fn $qc(
            data: *const $value,
            cells: *mut $value,
            genes: *mut $value,
            cell_ex: *mut i32,
            gene_ex: *mut i32,
            mask: *const u8,
            rows: u64,
            cols: u64,
            mode: u32,
        ) {
            let mut p = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while p < rows * cols {
                let row = (p / cols) as usize;
                let col = (p % cols) as usize;
                let v = unsafe { *data.add(p as usize) };
                unsafe {
                    if mode == 1 {
                        if *mask.add(col) != 0 {
                            $add(cells.add(row), v);
                        }
                    } else if v > 0.0 {
                        if mode != 3 {
                            $add(cells.add(row), v);
                            add_i32(cell_ex.add(row), 1);
                        }
                        if mode != 2 {
                            $add(genes.add(col), v);
                            add_i32(gene_ex.add(col), 1);
                        }
                    }
                }
                p += stride;
            }
        }

        /// Per-gene Poisson zero probability averaged over cell library sizes.
        /// # Safety
        /// Host validates vector lengths; each expected probability has one writer.
        #[kernel]
        pub unsafe fn $expected(
            means: *const $value,
            counts: *const $value,
            out: *mut $value,
            genes: u64,
            cells: u64,
        ) {
            let mut gene = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while gene < genes {
                let sm = unsafe { *means.add(gene as usize) };
                let mut sum = 0.0 as $value;
                let mut cell = 0;
                while cell < cells {
                    sum += (-sm * unsafe { *counts.add(cell as usize) }).exp();
                    cell += 1;
                }
                unsafe {
                    *out.add(gene as usize) = sum / cells as $value;
                }
                gene += stride;
            }
        }

        /// Dense Pearson residuals (C layout) or per-gene variance (F layout).
        /// # Safety
        /// Host validates the mode-dependent output length and storage layout.
        #[kernel]
        pub unsafe fn $residual(
            data: *const $value,
            cells: *const $value,
            genes: *const $value,
            out: *mut $value,
            rows: u64,
            cols: u64,
            inv_sum: f64,
            clip: f64,
            inv_theta: f64,
            mode: u32,
        ) {
            let mut p = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            let size = if mode == 0 { rows * cols } else { cols };
            while p < size {
                if mode == 0 {
                    let row = p / cols;
                    let col = p % cols;
                    let mu = unsafe { *genes.add(col as usize) * *cells.add(row as usize) }
                        * inv_sum as $value;
                    let mut x = (unsafe { *data.add(p as usize) } - mu)
                        * $rsqrt(mu + mu * mu * inv_theta as $value);
                    if x < -(clip as $value) {
                        x = -(clip as $value);
                    }
                    if x > clip as $value {
                        x = clip as $value;
                    }
                    unsafe {
                        *out.add(p as usize) = x;
                    }
                } else {
                    let gene_sum = unsafe { *genes.add(p as usize) };
                    let mut mean = 0.0 as $value;
                    let mut m2 = 0.0 as $value;
                    let mut cell = 0_u64;
                    while cell < rows {
                        let mu =
                            gene_sum * unsafe { *cells.add(cell as usize) } * inv_sum as $value;
                        let value = unsafe { *data.add((p * rows + cell) as usize) };
                        let x = ((value - mu) * $rsqrt(mu + mu * mu * inv_theta as $value))
                            .max(-(clip as $value))
                            .min(clip as $value);
                        let delta = x - mean;
                        mean += delta / (cell + 1) as $value;
                        m2 = delta.mul_add(x - mean, m2);
                        cell += 1;
                    }
                    unsafe {
                        *out.add(p as usize) = m2 / rows as $value;
                    }
                }
                p += stride;
            }
        }
    };
}

sparse_kernels!(
    prep_stats_major_f32_i32,
    prep_stats_minor_f32_i32,
    prep_norm_f32_i32,
    prep_scale_f32_i32,
    prep_qc_f32_i32,
    prep_residual_f32_i32,
    f32,
    i32,
    add_f32,
    reduce_sum_f32,
    rsqrt_approx_f32,
    fma_rn_f32
);
sparse_kernels!(
    prep_stats_major_f32_i64,
    prep_stats_minor_f32_i64,
    prep_norm_f32_i64,
    prep_scale_f32_i64,
    prep_qc_f32_i64,
    prep_residual_f32_i64,
    f32,
    i64,
    add_f32,
    reduce_sum_f32,
    rsqrt_approx_f32,
    fma_rn_f32
);
sparse_kernels!(
    prep_stats_major_f64_i32,
    prep_stats_minor_f64_i32,
    prep_norm_f64_i32,
    prep_scale_f64_i32,
    prep_qc_f64_i32,
    prep_residual_f64_i32,
    f64,
    i32,
    add_f64,
    reduce_sum_f64,
    rsqrt_approx_f64,
    fma_rn_f64
);
sparse_kernels!(
    prep_stats_major_f64_i64,
    prep_stats_minor_f64_i64,
    prep_norm_f64_i64,
    prep_scale_f64_i64,
    prep_qc_f64_i64,
    prep_residual_f64_i64,
    f64,
    i64,
    add_f64,
    reduce_sum_f64,
    rsqrt_approx_f64,
    fma_rn_f64
);
dense_kernels!(
    prep_norm_dense_f32,
    prep_scale_dense_f32,
    prep_qc_dense_f32,
    prep_expected_f32,
    prep_residual_dense_f32,
    f32,
    add_f32,
    reduce_sum_f32,
    rsqrt_approx_f32,
    fma_rn_f32
);
dense_kernels!(
    prep_norm_dense_f64,
    prep_scale_dense_f64,
    prep_qc_dense_f64,
    prep_expected_f64,
    prep_residual_dense_f64,
    f64,
    add_f64,
    reduce_sum_f64,
    rsqrt_approx_f64,
    fma_rn_f64
);

/// Merge atomic integer flags into the caller's existing bool mask.
/// # Safety
/// Both arrays have at least `genes` elements and are disjoint. Prior updates
/// to `flags` must be complete in this stream before this kernel is launched.
#[kernel]
pub unsafe fn prep_high_flags_f32(flags: *const i32, mask: *mut u8, genes: u64) {
    let mut i = thread::index_1d().get() as u64;
    let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
    while i < genes {
        if unsafe { *flags.add(i as usize) != 0 } {
            unsafe {
                *mask.add(i as usize) = 1;
            }
        }
        i += stride;
    }
}
