//! Native replacements for formerly embedded elementwise and reduction kernels.
#![allow(clippy::too_many_arguments)]
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32};
use cuda_device::{kernel, thread, warp};

macro_rules! float_kernels {
    ($axpy:ident, $sum:ident, $value:ty, $fma:ident) => {
        /// Fused in-place Lanczos AXPY with a device scalar.
        /// # Safety
        /// Host validates input lengths, dtype, layout, and disjoint allocations.
        #[kernel]
        pub unsafe fn $axpy(alpha: *const $value, y: *const $value, x: *mut $value, len: u64) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            let a = unsafe { *alpha };
            while i < len {
                unsafe {
                    *x.add(i as usize) = (-a).mul_add(*y.add(i as usize), *x.add(i as usize));
                }
                i += stride;
            }
        }
        /// Float64 dense reduction, retaining input-precision multiplication.
        /// # Safety
        /// Contiguous C/F storage has validated dimensions/strides. Launch
        /// 32 threads per block; mode 0 sums values and mode 1 sums their squares.
        /// Strided reductions write `parts * major` outputs. Contiguous
        /// reductions require one part; the host always passes a positive count.
        #[kernel]
        pub unsafe fn $sum(
            data: *const $value,
            output: *mut f64,
            rows: u64,
            cols: u64,
            row_stride: u64,
            col_stride: u64,
            axis: u32,
            square: u32,
            parts: u64,
        ) {
            let major = if axis == 0 { cols } else { rows };
            let minor = if axis == 0 { rows } else { cols };
            let major_stride = if axis == 0 { col_stride } else { row_stride };
            let minor_stride = if axis == 0 { row_stride } else { col_stride };
            // Warp reductions are used along physically contiguous dimensions;
            // otherwise neighboring lanes process neighboring output elements.
            if minor_stride == 1 {
                let mut major_idx = thread::blockIdx_x() as u64;
                while major_idx < major {
                    let mut sum = 0.0_f64;
                    let mut minor_idx = thread::threadIdx_x() as u64;
                    while minor_idx < minor {
                        let x =
                            unsafe { *data.add((major_idx * major_stride + minor_idx) as usize) };
                        sum += if square != 0 {
                            (x * x) as f64
                        } else {
                            x as f64
                        };
                        minor_idx += 32;
                    }
                    sum = warp::reduce_sum_f64(sum);
                    if thread::threadIdx_x() == 0 {
                        unsafe {
                            *output.add(major_idx as usize) = sum;
                        }
                    }
                    major_idx += thread::gridDim_x() as u64;
                }
            } else {
                let mut output_idx = thread::index_1d().get() as u64;
                let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
                while output_idx < major * parts {
                    let major_idx = output_idx % major;
                    let part = output_idx / major;
                    let width = minor.div_ceil(parts);
                    let mut sum = 0.0_f64;
                    let mut minor_idx = part * width;
                    let end = ((part + 1) * width).min(minor);
                    while minor_idx < end {
                        let x = unsafe {
                            *data
                                .add((major_idx * major_stride + minor_idx * minor_stride) as usize)
                        };
                        sum += if square != 0 {
                            (x * x) as f64
                        } else {
                            x as f64
                        };
                        minor_idx += 1;
                    }
                    unsafe {
                        *output.add(output_idx as usize) = sum;
                    }
                    output_idx += stride;
                }
            }
        }
    };
}

macro_rules! clip_kernel {
    ($name:ident, $value:ty, $index:ty) => {
        /// Clipped sparse sums and squares for Seurat v3 HVG.
        /// # Safety
        /// Host validates all arrays; invalid indices are skipped before access.
        #[kernel]
        pub unsafe fn $name(
            data: *const $value,
            indices: *const $index,
            clip: *const f64,
            squares: *mut f64,
            sums: *mut f64,
            len: u64,
            genes: u64,
        ) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while i < len {
                let g = unsafe { *indices.add(i as usize) } as i64;
                if g >= 0 && (g as u64) < genes {
                    let v = (unsafe { *data.add(i as usize) } as f64)
                        .min(unsafe { *clip.add(g as usize) });
                    unsafe { DeviceAtomicF64::from_ptr(sums.add(g as usize)) }
                        .fetch_add(v, AtomicOrdering::Relaxed);
                    unsafe { DeviceAtomicF64::from_ptr(squares.add(g as usize)) }
                        .fetch_add(v * v, AtomicOrdering::Relaxed);
                }
                i += stride;
            }
        }
    };
}

macro_rules! index_kernels {
    ($count:ident, $diff:ident, $assign:ident, $scatter:ident, $index:ty) => {
        /// Count stored entries per minor index.
        /// # Safety
        /// Host validates lengths; destination indices are checked on device.
        #[kernel]
        pub unsafe fn $count(indices: *const $index, counts: *mut i32, len: u64, genes: u64) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while i < len {
                let g = unsafe { *indices.add(i as usize) } as i64;
                if g >= 0 && (g as u64) < genes {
                    unsafe { DeviceAtomicI32::from_ptr(counts.add(g as usize)) }
                        .fetch_add(1, AtomicOrdering::Relaxed);
                }
                i += stride;
            }
        }
        /// Mark boundaries in sorted coordinate arrays, retaining initial zero.
        /// # Safety
        /// Host validates equally sized, disjoint arrays.
        #[kernel]
        pub unsafe fn $diff(rows: *const $index, cols: *const $index, out: *mut $index, len: u64) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while i < len {
                let changed = i != 0
                    && unsafe {
                        *rows.add(i as usize) != *rows.add(i as usize - 1)
                            || *cols.add(i as usize) != *cols.add(i as usize - 1)
                    };
                unsafe {
                    *out.add(i as usize) = changed as $index;
                }
                i += stride;
            }
        }
        /// Assign one representative per duplicate coordinate pair.
        /// # Safety
        /// Host validates array lengths. Indices are a nondecreasing run mapping;
        /// only the first entry of each run writes, avoiding concurrent stores.
        #[kernel]
        pub unsafe fn $assign(
            src_row: *const $index,
            src_col: *const $index,
            indices: *const $index,
            rows: *mut $index,
            cols: *mut $index,
            len: u64,
            groups: u64,
        ) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while i < len {
                let g = unsafe { *indices.add(i as usize) } as i64;
                if g >= 0
                    && (g as u64) < groups
                    && (i == 0 || unsafe { *indices.add(i as usize - 1) } as i64 != g)
                {
                    unsafe {
                        *rows.add(g as usize) = *src_row.add(i as usize);
                        *cols.add(g as usize) = *src_col.add(i as usize);
                    }
                }
                i += stride;
            }
        }
        /// Grouped float64 sum/squared sum and float32 nonzero counts.
        /// # Safety
        /// Host validates arrays; all scatter indices are checked on device.
        #[kernel]
        pub unsafe fn $scatter(
            data: *const f64,
            indices: *const $index,
            sums: *mut f64,
            squares: *mut f64,
            counts: *mut f32,
            len: u64,
            groups: u64,
            mode: u32,
        ) {
            let mut i = thread::index_1d().get() as u64;
            let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            while i < len {
                let g = unsafe { *indices.add(i as usize) } as i64;
                if g >= 0 && (g as u64) < groups {
                    let value = unsafe { *data.add(i as usize) };
                    if mode == 2 {
                        if value != 0.0 {
                            unsafe { DeviceAtomicF32::from_ptr(counts.add(g as usize)) }
                                .fetch_add(1.0, AtomicOrdering::Relaxed);
                        }
                    } else {
                        unsafe { DeviceAtomicF64::from_ptr(sums.add(g as usize)) }
                            .fetch_add(value, AtomicOrdering::Relaxed);
                        if mode == 1 {
                            unsafe { DeviceAtomicF64::from_ptr(squares.add(g as usize)) }
                                .fetch_add(value * value, AtomicOrdering::Relaxed);
                        }
                    }
                }
                i += stride;
            }
        }
    };
}

float_kernels!(prep_axpy_f32, prep_dense_sum_f32, f32, fma_rn_f32);
float_kernels!(prep_axpy_f64, prep_dense_sum_f64, f64, fma_rn_f64);
clip_kernel!(prep_clip_sums_f32_i32, f32, i32);
clip_kernel!(prep_clip_sums_f32_i64, f32, i64);
clip_kernel!(prep_clip_sums_f64_i32, f64, i32);
clip_kernel!(prep_clip_sums_f64_i64, f64, i64);
index_kernels!(
    prep_count_f32_i32,
    prep_duplicates_diff_f32_i32,
    prep_duplicates_assign_f32_i32,
    prep_scatter_f64_i32,
    i32
);
index_kernels!(
    prep_count_f32_i64,
    prep_duplicates_diff_f32_i64,
    prep_duplicates_assign_f32_i64,
    prep_scatter_f64_i64,
    i64
);
