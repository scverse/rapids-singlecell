//! Layout-specialized Pearson residuals with the original sequential Welford
//! update and four-way loop unrolling. Inputs are disjoint from the output, so
//! invariant sums and read-only data can use CUDA's noncoherent read-only cache.
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};

macro_rules! arithmetic {
    ($module:ident, $value:ty, $out:literal, $input:literal,
     $load:literal, $subtract:literal, $inverse:literal, $clamp:literal) => {
        mod $module {
            #[inline(always)]
            pub unsafe fn load(pointer: *const $value) -> $value {
                let result: $value;
                unsafe {
                    cuda_device::ptx_asm!($load, out($out) result, in("l") pointer);
                }
                result
            }

            #[inline(always)]
            pub fn residual(value: $value, mu: $value, inv_theta: $value) -> $value {
                let difference: $value;
                // Materialize mu before subtraction, matching CUDA's rounded
                // intermediate instead of contracting value - gene*cell*scale.
                unsafe {
                    cuda_device::ptx_asm!($subtract, out($out) difference,
                        in($input) value, in($input) mu);
                }
                let variance = (mu * mu).mul_add(inv_theta, mu);
                let inverse: $value;
                unsafe {
                    cuda_device::ptx_asm!($inverse, out($out) inverse, in($input) variance);
                }
                difference * inverse
            }

            #[inline(always)]
            pub fn clamp(value: $value, clip: $value) -> $value {
                let result: $value;
                unsafe {
                    cuda_device::ptx_asm!($clamp, out($out) result,
                        in($input) value, in($input) -clip, in($input) clip);
                }
                result
            }
        }
    };
}

arithmetic!(
    float,
    f32,
    "=f",
    "f",
    "ld.global.nc.f32 %0, [%1];",
    "sub.rn.f32 %0, %1, %2;",
    "rsqrt.approx.f32 %0, %1;",
    "{ .reg .f32 lower; max.f32 lower, %1, %2; min.f32 %0, lower, %3; }"
);
arithmetic!(
    double,
    f64,
    "=d",
    "d",
    "ld.global.nc.f64 %0, [%1];",
    "sub.rn.f64 %0, %1, %2;",
    "rsqrt.approx.f64 %0, %1;",
    "{ .reg .f64 lower; max.f64 lower, %1, %2; min.f64 %0, lower, %3; }"
);

macro_rules! sparse_residual {
    ($name:ident, $mode:literal, $value:ty, $index:ty, $math:ident) => {
        /// Clipped sparse Pearson residuals, or their per-gene variance.
        /// # Safety
        /// Compressed rows are sorted; arrays have the host-validated lengths,
        /// are on this device, and input allocations are disjoint from output.
        #[kernel]
        pub unsafe fn $name(
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
        ) {
            // Compile-time modes retain separate CSR, CSC and HVG loops; no
            // per-element layout/variance dispatch survives in generated PTX.
            let major = if $mode == 0 { n_cells } else { n_genes };
            let minor = if $mode == 0 { n_genes } else { n_cells };
            let major_sums = if $mode == 0 { cells } else { genes };
            let minor_sums = if $mode == 0 { genes } else { cells };
            let mut row = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
                + thread::threadIdx_x() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while row < major {
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add(row as usize + 1) } as i64;
                let valid = start >= 0 && stop >= start && stop as u64 <= nnz;
                let mut cursor = if valid { start as u64 } else { 0 };
                let end = if valid { stop as u64 } else { 0 };
                let major_sum = unsafe { $math::load(major_sums.add(row as usize)) };
                let mut mean = 0.0 as $value;
                let mut m2 = 0.0 as $value;
                let mut col = 0_u64;
                #[unroll(4)]
                while col < minor {
                    let offset = if $mode == 0 {
                        row * n_genes + col
                    } else {
                        col * n_genes + row
                    } as usize;
                    let mut value = if $mode == 2 {
                        0.0
                    } else {
                        unsafe { *output.add(offset) }
                    };
                    if cursor < end && unsafe { *index.add(cursor as usize) } as i64 == col as i64 {
                        value += unsafe { $math::load(data.add(cursor as usize)) };
                        cursor += 1;
                    }
                    let mu = major_sum
                        * unsafe { $math::load(minor_sums.add(col as usize)) }
                        * inv_sum as $value;
                    let mut x = $math::residual(value, mu, inv_theta as $value);
                    if $mode == 2 {
                        x = $math::clamp(x, clip as $value);
                        let delta = x - mean;
                        mean += delta / (col + 1) as $value;
                        m2 = delta.mul_add(x - mean, m2);
                    } else {
                        // Ordinary residuals preserve NaNs; HVG deliberately
                        // follows CUDA fmin/fmax's non-NaN operand selection.
                        if x < -(clip as $value) {
                            x = -(clip as $value);
                        }
                        if x > clip as $value {
                            x = clip as $value;
                        }
                        unsafe { *output.add(offset) = x };
                    }
                    col += 1;
                }
                if $mode == 2 {
                    unsafe { *output.add(row as usize) = m2 / n_cells as $value };
                }
                row += stride;
            }
        }
    };
}

sparse_residual!(prep_residual_csr_f32_i32, 0, f32, i32, float);
sparse_residual!(prep_residual_csr_f32_i64, 0, f32, i64, float);
sparse_residual!(prep_residual_csr_f64_i32, 0, f64, i32, double);
sparse_residual!(prep_residual_csr_f64_i64, 0, f64, i64, double);
sparse_residual!(prep_residual_csc_f32_i32, 1, f32, i32, float);
sparse_residual!(prep_residual_csc_f32_i64, 1, f32, i64, float);
sparse_residual!(prep_residual_csc_f64_i32, 1, f64, i32, double);
sparse_residual!(prep_residual_csc_f64_i64, 1, f64, i64, double);
sparse_residual!(prep_residual_hvg_f32_i32, 2, f32, i32, float);
sparse_residual!(prep_residual_hvg_f32_i64, 2, f32, i64, float);
sparse_residual!(prep_residual_hvg_f64_i32, 2, f64, i32, double);
sparse_residual!(prep_residual_hvg_f64_i64, 2, f64, i64, double);

macro_rules! dense_residual {
    ($name:ident, $variance:literal, $value:ty, $math:ident) => {
        /// Dense C-layout Pearson residuals or F-layout per-gene variance.
        /// # Safety
        /// Arrays have the host-validated lengths and layout, are on this
        /// device, and input allocations are disjoint from output.
        #[kernel]
        pub unsafe fn $name(
            data: *const $value,
            cells: *const $value,
            genes: *const $value,
            output: *mut $value,
            rows: u64,
            cols: u64,
            inv_sum: f64,
            clip: f64,
            inv_theta: f64,
        ) {
            let mut p = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
                + thread::threadIdx_x() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            let size = if $variance { cols } else { rows * cols };
            while p < size {
                if $variance {
                    let gene_sum = unsafe { $math::load(genes.add(p as usize)) };
                    let mut mean = 0.0 as $value;
                    let mut m2 = 0.0 as $value;
                    let mut cell = 0_u64;
                    #[unroll(4)]
                    while cell < rows {
                        let mu = gene_sum
                            * unsafe { $math::load(cells.add(cell as usize)) }
                            * inv_sum as $value;
                        let value = unsafe { $math::load(data.add((p * rows + cell) as usize)) };
                        let x = $math::clamp(
                            $math::residual(value, mu, inv_theta as $value),
                            clip as $value,
                        );
                        let delta = x - mean;
                        mean += delta / (cell + 1) as $value;
                        m2 = delta.mul_add(x - mean, m2);
                        cell += 1;
                    }
                    unsafe { *output.add(p as usize) = m2 / rows as $value };
                } else {
                    let row = p / cols;
                    let col = p % cols;
                    let mu = unsafe {
                        $math::load(genes.add(col as usize)) * $math::load(cells.add(row as usize))
                    } * inv_sum as $value;
                    let value = unsafe { $math::load(data.add(p as usize)) };
                    let mut x = $math::residual(value, mu, inv_theta as $value);
                    if x < -(clip as $value) {
                        x = -(clip as $value);
                    }
                    if x > clip as $value {
                        x = clip as $value;
                    }
                    unsafe { *output.add(p as usize) = x };
                }
                p += stride;
            }
        }
    };
}

dense_residual!(prep_residual_dense_f32, false, f32, float);
dense_residual!(prep_residual_dense_f64, false, f64, double);
dense_residual!(prep_residual_dense_hvg_f32, true, f32, float);
dense_residual!(prep_residual_dense_hvg_f64, true, f64, double);
