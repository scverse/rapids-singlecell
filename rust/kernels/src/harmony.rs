//! Harmony device kernels with bounded reductions and adaptive row blocks.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub(crate) use crate::atomics::add_f32;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{kernel, ptx_asm, thread, warp};
#[inline(always)]
pub(crate) fn sum_f32(mut v: f32) -> f32 {
    let mut d = 16;
    while d > 0 {
        v += warp::shuffle_down_f32(v, d);
        d /= 2;
    }
    warp::shuffle_f32(v, 0)
}
#[inline(always)]
pub(crate) fn max_f32(mut v: f32) -> f32 {
    let mut d = 16;
    while d > 0 {
        v = v.max(warp::shuffle_down_f32(v, d));
        d /= 2;
    }
    warp::shuffle_f32(v, 0)
}
#[inline(always)]
pub(crate) unsafe fn add_f64(p: *mut f64, v: f64) {
    unsafe { DeviceAtomicF64::from_ptr(p) }.fetch_add(v, AtomicOrdering::Relaxed);
}
#[inline(always)]
pub(crate) fn sum_f64(mut v: f64) -> f64 {
    let mut d = 16;
    while d > 0 {
        v += warp::shuffle_down_f64(v, d);
        d /= 2;
    }
    warp::shuffle_f64(v, 0)
}
#[inline(always)]
pub(crate) fn max_f64(mut v: f64) -> f64 {
    let mut d = 16;
    while d > 0 {
        v = v.max(warp::shuffle_down_f64(v, d));
        d /= 2;
    }
    warp::shuffle_f64(v, 0)
}
#[kernel]
pub unsafe fn harmony_scatter_f32(
    v: *const f32,
    cats: *const i32,
    a: *mut f32,
    rows: u64,
    cols: u64,
    covs: u64,
    batches: u64,
    sign: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (rows, cols, covs, batches) = (
            rows as usize,
            cols as usize,
            covs as usize,
            batches as usize,
        );
        while i < rows * cols {
            let row = i / cols;
            let val = if sign == 0 { -*v.add(i) } else { *v.add(i) };
            for c in 0..covs {
                let cat = *cats.add(row * covs + c);
                if cat >= 0 && (cat as usize) < batches {
                    add_f32(a.add(cat as usize * cols + i % cols), val);
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_scatter_f64(
    v: *const f64,
    cats: *const i32,
    a: *mut f64,
    rows: u64,
    cols: u64,
    covs: u64,
    batches: u64,
    sign: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (rows, cols, covs, batches) = (
            rows as usize,
            cols as usize,
            covs as usize,
            batches as usize,
        );
        while i < rows * cols {
            let row = i / cols;
            let val = if sign == 0 { -*v.add(i) } else { *v.add(i) };
            for c in 0..covs {
                let cat = *cats.add(row * covs + c);
                if cat >= 0 && (cat as usize) < batches {
                    add_f64(a.add(cat as usize * cols + i % cols), val);
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_rows_f32(
    src: *const f32,
    idx: *const i32,
    dst: *mut f32,
    rows: u64,
    cols: u64,
    other_rows: u64,
    scatter: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (rows, cols) = (rows as usize, cols as usize);
        while i < rows * cols {
            let row = i / cols;
            let r = *idx.add(row);
            if r >= 0 && (r as u64) < other_rows {
                let j = r as usize * cols + i % cols;
                if scatter != 0 {
                    *dst.add(j) = *src.add(i);
                } else {
                    *dst.add(i) = *src.add(j);
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_rows_f64(
    src: *const f64,
    idx: *const i32,
    dst: *mut f64,
    rows: u64,
    cols: u64,
    other_rows: u64,
    scatter: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (rows, cols) = (rows as usize, cols as usize);
        while i < rows * cols {
            let row = i / cols;
            let r = *idx.add(row);
            if r >= 0 && (r as u64) < other_rows {
                let j = r as usize * cols + i % cols;
                if scatter != 0 {
                    *dst.add(j) = *src.add(i);
                } else {
                    *dst.add(i) = *src.add(j);
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_rows_i32(
    src: *const i32,
    idx: *const i32,
    dst: *mut i32,
    rows: u64,
    cols: u64,
    other_rows: u64,
    scatter: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (rows, cols) = (rows as usize, cols as usize);
        while i < rows * cols {
            let row = i / cols;
            let r = *idx.add(row);
            if r >= 0 && (r as u64) < other_rows {
                let j = r as usize * cols + i % cols;
                if scatter != 0 {
                    *dst.add(j) = *src.add(i);
                } else {
                    *dst.add(i) = *src.add(j);
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_outer_f32(
    e: *mut f32,
    p: *const f32,
    r: *const f32,
    rows: u64,
    cols: u64,
    sign: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let v = *p.add(i / cols as usize) * *r.add(i % cols as usize);
            *e.add(i) += if sign == 0 { -v } else { v };
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_outer_f64(
    e: *mut f64,
    p: *const f64,
    r: *const f64,
    rows: u64,
    cols: u64,
    sign: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let v = *p.add(i / cols as usize) * *r.add(i % cols as usize);
            *e.add(i) += if sign == 0 { -v } else { v };
            i += stride;
        }
    }
}
// Column sums need a full block for tall matrices. The accumulating variant
// partitions rows and coalesces adjacent columns before one atomic per tile.
#[inline(always)]
fn colsum_plus_f32(a: f32, b: f32) -> f32 {
    a + b
}
#[inline(always)]
fn colsum_plus_f64(a: f64, b: f64) -> f64 {
    a + b
}
#[inline(always)]
fn colsum_plus_i32(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}
#[inline(always)]
fn colsum_shuffle_i32(mask: u32, value: i32, delta: u32) -> i32 {
    warp::shuffle_down_sync(mask, value as u32, delta) as i32
}
#[inline(always)]
unsafe fn colsum_add_i32(pointer: *mut i32, value: i32) {
    unsafe { cuda_device::atomic::DeviceAtomicI32::from_ptr(pointer) }
        .fetch_add(value, AtomicOrdering::Relaxed);
}
macro_rules! column_sum {
    ($name:ident, $value:ty, $plus:ident, $shuffle:path, $atomic:ident) => {
        #[kernel]
        #[cuda_device::launch_bounds(1024)]
        pub unsafe fn $name(
            a: *const $value,
            out: *mut $value,
            rows: u64,
            cols: u64,
            accumulate: u32,
            rows_per_tile: u64,
        ) {
            unsafe {
                static mut SHARED: cuda_device::SharedArray<$value, 256> =
                    cuda_device::SharedArray::UNINIT;
                let tid = thread::threadIdx_x() as u64;
                let block = thread::blockDim_x() as u64;
                let lane = tid % 32;
                let warp_id = tid / 32;
                if accumulate != 0 {
                    let col_tiles = cols.div_ceil(32);
                    let row_tiles = rows.div_ceil(rows_per_tile);
                    let mut tile = thread::blockIdx_x() as u64;
                    while tile < col_tiles * row_tiles {
                        let col = tile % col_tiles * 32 + lane;
                        let begin = tile / col_tiles * rows_per_tile;
                        let end = (begin + rows_per_tile).min(rows);
                        let mut value = 0 as $value;
                        let mut row = begin + warp_id;
                        if col < cols {
                            while row < end {
                                value = $plus(value, *a.add((row * cols + col) as usize));
                                row += 8;
                            }
                        }
                        SHARED[tid as usize] = value;
                        thread::sync_threads();
                        if warp_id == 0 {
                            let mut total = SHARED[lane as usize];
                            let mut w = 1;
                            while w < 8 {
                                total = $plus(total, SHARED[(w * 32 + lane) as usize]);
                                w += 1;
                            }
                            if col < cols && total != 0 as $value {
                                $atomic(out.add(col as usize), total);
                            }
                        }
                        thread::sync_threads();
                        tile += thread::gridDim_x() as u64;
                    }
                } else {
                    let mut col = thread::blockIdx_x() as u64;
                    while col < cols {
                        let mut value = 0 as $value;
                        let mut row = tid;
                        while row < rows {
                            value = $plus(value, *a.add((row * cols + col) as usize));
                            row += block;
                        }
                        let mut delta = 16;
                        while delta > 0 {
                            value = $plus(value, $shuffle(u32::MAX, value, delta));
                            delta /= 2;
                        }
                        if lane == 0 {
                            SHARED[warp_id as usize] = value;
                        }
                        thread::sync_threads();
                        if warp_id == 0 {
                            let mut total = if lane < block / 32 {
                                SHARED[lane as usize]
                            } else {
                                0 as $value
                            };
                            let mut delta = 16;
                            while delta > 0 {
                                total = $plus(total, $shuffle(u32::MAX, total, delta));
                                delta /= 2;
                            }
                            if lane == 0 {
                                *out.add(col as usize) = total;
                            }
                        }
                        thread::sync_threads();
                        col += thread::gridDim_x() as u64;
                    }
                }
            }
        }
    };
}
column_sum!(
    harmony_colsum_f32,
    f32,
    colsum_plus_f32,
    warp::shuffle_down_f32_sync,
    add_f32
);
column_sum!(
    harmony_colsum_f64,
    f64,
    colsum_plus_f64,
    warp::shuffle_down_f64_sync,
    add_f64
);
column_sum!(
    harmony_colsum_i32,
    i32,
    colsum_plus_i32,
    colsum_shuffle_i32,
    colsum_add_i32
);
// Match the original CUDA rsqrt(double) PTX, including its subnormal and
// nonfinite handling. Inline assembly avoids the intrinsic catalog's newer
// target guard while preserving the supported sm75 release architecture.
#[inline(always)]
fn inverse_norm_f64(value: f64, l2: bool) -> f64 {
    if !l2 {
        return 1.0 / if value < 1e-12 { 1e-12 } else { value };
    }
    let inverse: f64;
    unsafe {
        ptx_asm!("rsqrt.approx.f64 %0, %1;", out("=d") inverse, in("d") value);
    }
    if inverse > 1e12 { 1e12 } else { inverse }
}
#[inline(always)]
fn inverse_norm_f32(value: f32, l2: bool) -> f32 {
    if !l2 {
        return 1.0 / if value < 1e-12 { 1e-12 } else { value };
    }
    let inverse = 1.0 / value.sqrt();
    if inverse > 1e12 { 1e12 } else { inverse }
}
#[inline(always)]
fn row_reduce_f32(mut value: f32, mode: u32) -> f32 {
    let maximum = mode == 1;
    static mut PARTIAL: cuda_device::SharedArray<f32, 9> = cuda_device::SharedArray::UNINIT;
    let lane = thread::threadIdx_x() as usize;
    value = if maximum {
        max_f32(value)
    } else {
        sum_f32(value)
    };
    if thread::blockDim_x() == 32 {
        if mode >= 2 {
            if lane == 0 {
                value = inverse_norm_f32(value, mode == 3);
            }
            return warp::shuffle_f32(value, 0);
        }
        return value;
    }
    if lane.is_multiple_of(32) {
        unsafe {
            PARTIAL[lane / 32] = value;
        }
    }
    thread::sync_threads();
    if lane < 32 {
        value = if lane < thread::blockDim_x() as usize / 32 {
            unsafe { PARTIAL[lane] }
        } else if maximum {
            f32::NEG_INFINITY
        } else {
            0.0
        };
        value = if maximum {
            max_f32(value)
        } else {
            sum_f32(value)
        };
        if lane == 0 {
            unsafe {
                PARTIAL[8] = if mode >= 2 {
                    inverse_norm_f32(value, mode == 3)
                } else {
                    value
                };
            }
        }
    }
    thread::sync_threads();
    // A separate broadcast slot survives the next reduction's partial
    // writes until every thread reaches its first barrier.
    unsafe { PARTIAL[8] }
}
#[inline(always)]
fn row_reduce_f64(mut value: f64, mode: u32) -> f64 {
    let maximum = mode == 1;
    static mut PARTIAL: cuda_device::SharedArray<f64, 9> = cuda_device::SharedArray::UNINIT;
    let lane = thread::threadIdx_x() as usize;
    value = if maximum {
        max_f64(value)
    } else {
        sum_f64(value)
    };
    if thread::blockDim_x() == 32 {
        if mode >= 2 {
            if lane == 0 {
                value = inverse_norm_f64(value, mode == 3);
            }
            return warp::shuffle_f64(value, 0);
        }
        return value;
    }
    if lane.is_multiple_of(32) {
        unsafe {
            PARTIAL[lane / 32] = value;
        }
    }
    thread::sync_threads();
    if lane < 32 {
        value = if lane < thread::blockDim_x() as usize / 32 {
            unsafe { PARTIAL[lane] }
        } else if maximum {
            f64::NEG_INFINITY
        } else {
            0.0
        };
        value = if maximum {
            max_f64(value)
        } else {
            sum_f64(value)
        };
        if lane == 0 {
            unsafe {
                PARTIAL[8] = if mode >= 2 {
                    inverse_norm_f64(value, mode == 3)
                } else {
                    value
                };
            }
        }
    }
    thread::sync_threads();
    // A separate broadcast slot survives the next reduction's partial
    // writes until every thread reaches its first barrier.
    unsafe { PARTIAL[8] }
}
#[kernel]
pub unsafe fn harmony_normalize_f32(src: *const f32, dst: *mut f32, rows: u64, cols: u64, l2: u32) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        let cols = cols as usize;
        while row < rows as usize {
            let mut v = 0_f32;
            let mut c = lane;
            while c < cols {
                let x = *src.add(row * cols + c);
                v = if l2 != 0 {
                    x.mul_add(x, v)
                } else {
                    v + x.abs()
                };
                c += thread::blockDim_x() as usize;
            }
            // The inverse norm is uniform: compute it once per row,
            // then broadcast it with the block reduction.
            let scale = row_reduce_f32(v, if l2 != 0 { 3 } else { 2 });
            c = lane;
            while c < cols {
                *dst.add(row * cols + c) = *src.add(row * cols + c) * scale;
                c += thread::blockDim_x() as usize;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_normalize_f64(src: *const f64, dst: *mut f64, rows: u64, cols: u64, l2: u32) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        let cols = cols as usize;
        while row < rows as usize {
            let mut v = 0_f64;
            let mut c = lane;
            while c < cols {
                let x = *src.add(row * cols + c);
                v = if l2 != 0 {
                    x.mul_add(x, v)
                } else {
                    v + x.abs()
                };
                c += thread::blockDim_x() as usize;
            }
            // The inverse norm is uniform: compute it once per row,
            // then broadcast it with the block reduction.
            let scale = row_reduce_f64(v, if l2 != 0 { 3 } else { 2 });
            c = lane;
            while c < cols {
                *dst.add(row * cols + c) = *src.add(row * cols + c) * scale;
                c += thread::blockDim_x() as usize;
            }
            row += stride;
        }
    }
}
#[inline(always)]
unsafe fn kmeans_load4(pointer: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    unsafe {
        ptx_asm!("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];",
            out("=f") a, out("=f") b, out("=f") c, out("=f") d,
            in("l") pointer as u64, clobber("memory"));
    }
    [a, b, c, d]
}
#[inline(always)]
unsafe fn kmeans_load2(pointer: *const f64) -> [f64; 2] {
    let (a, b): (f64, f64);
    unsafe {
        ptx_asm!("ld.global.v2.f64 {%0, %1}, [%2];",
            out("=d") a, out("=d") b,
            in("l") pointer as u64, clobber("memory"));
    }
    [a, b]
}
macro_rules! kmeans_error {
    ($name:ident, $value:ty, $width:literal, $load:ident, $sum:ident, $atomic:ident) => {
        #[kernel]
        pub unsafe fn $name(r: *const $value, dot: *const $value, out: *mut $value, n: u64) {
            unsafe {
                static mut PARTIAL: cuda_device::SharedArray<$value, 8> =
                    cuda_device::SharedArray::UNINIT;
                let tid = thread::index_1d().get() as usize;
                let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
                let n = n as usize;
                let mut value = 0 as $value;
                let aligned = ((r as usize ^ dot as usize) & 15) == 0;
                if aligned {
                    let prefix = ((16 - (r as usize & 15)) & 15) / core::mem::size_of::<$value>();
                    let prefix = prefix.min(n);
                    let mut i = tid;
                    while i < prefix {
                        value += *r.add(i) * 2 as $value * (1 as $value - *dot.add(i));
                        i += stride;
                    }
                    let chunks = (n - prefix) / $width;
                    let mut chunk = tid;
                    while chunk < chunks {
                        let position = prefix + chunk * $width;
                        let rv = $load(r.add(position));
                        let dv = $load(dot.add(position));
                        let mut v = 0 as $value;
                        let mut j = 0;
                        while j < $width {
                            v += rv[j] * 2 as $value * (1 as $value - dv[j]);
                            j += 1;
                        }
                        value += v;
                        chunk += stride;
                    }
                    i = prefix + chunks * $width + tid;
                    while i < n {
                        value += *r.add(i) * 2 as $value * (1 as $value - *dot.add(i));
                        i += stride;
                    }
                } else {
                    let mut i = tid;
                    while i < n {
                        value += *r.add(i) * 2 as $value * (1 as $value - *dot.add(i));
                        i += stride;
                    }
                }
                value = $sum(value);
                let lane = thread::threadIdx_x() as usize;
                if lane.is_multiple_of(32) {
                    PARTIAL[lane / 32] = value;
                }
                thread::sync_threads();
                if lane < 32 {
                    let value = if lane < 8 { PARTIAL[lane] } else { 0 as $value };
                    let total = $sum(value);
                    if lane == 0 {
                        $atomic(out, total);
                    }
                }
            }
        }
    };
}
kmeans_error!(harmony_kmeans_f32, f32, 4, kmeans_load4, sum_f32, add_f32);
kmeans_error!(harmony_kmeans_f64, f64, 2, kmeans_load2, sum_f64, add_f64);
#[kernel]
pub unsafe fn harmony_penalty_f32(
    e: *const f32,
    o: *const f32,
    theta: *const f32,
    out: *mut f32,
    rows: u64,
    cols: u64,
    stabilized: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let ev = *e.add(i);
            let ov = *o.add(i);
            let denominator = ov + if stabilized != 0 { ev } else { 0_f32 } + 1_f32;
            *out.add(i) = ((ev + 1_f32) / denominator).powf(*theta.add(i / cols as usize));
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_penalty_f64(
    e: *const f64,
    o: *const f64,
    theta: *const f64,
    out: *mut f64,
    rows: u64,
    cols: u64,
    stabilized: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let ev = *e.add(i);
            let ov = *o.add(i);
            let denominator = ov + if stabilized != 0 { ev } else { 0_f64 } + 1_f64;
            *out.add(i) = ((ev + 1_f64) / denominator).powf(*theta.add(i / cols as usize));
            i += stride;
        }
    }
}
#[inline(always)]
unsafe fn harmony_pen_norm_impl_f32<const COVARIATES: usize>(
    sim: *const f32,
    pen: *const f32,
    cats: *const i32,
    idx: *const i32,
    out: *mut f32,
    rows: u64,
    cols: u64,
    covs: u64,
    sim_rows: u64,
    batches: u64,
    term: f32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        let cols = cols as usize;
        let covs = if COVARIATES == 0 {
            covs as usize
        } else {
            COVARIATES
        };
        while row < rows as usize {
            let sr = *idx.add(row);
            if sr >= 0 && (sr as u64) < sim_rows {
                // Marginal categories are invariant across this row's columns.
                let cat0 = *cats.add(row * covs);
                let cat1 = if covs >= 2 {
                    *cats.add(row * covs + 1)
                } else {
                    0
                };
                let cat2 = if covs >= 3 {
                    *cats.add(row * covs + 2)
                } else {
                    0
                };
                let cat3 = if covs >= 4 {
                    *cats.add(row * covs + 3)
                } else {
                    0
                };
                let valid01 =
                    cat0 >= 0 && (cat0 as u64) < batches && cat1 >= 0 && (cat1 as u64) < batches;
                let valid2 = cat2 >= 0 && (cat2 as u64) < batches;
                let valid3 = cat3 >= 0 && (cat3 as u64) < batches;
                // The single-covariate path avoids a logarithm per element.
                // Fall back to log-sum-exp if finite inputs under/overflow.
                if covs == 1 {
                    let cat = *cats.add(row);
                    if cat >= 0 && (cat as u64) < batches {
                        let mut sum = 0.0;
                        let mut c = lane;
                        while c < cols {
                            let v = (term * (1.0 - *sim.add(sr as usize * cols + c))).exp()
                                * *pen.add(cat as usize * cols + c);
                            *out.add(row * cols + c) = v;
                            sum += v;
                            c += thread::blockDim_x() as usize;
                        }
                        sum = row_reduce_f32(sum, 0);
                        if sum > 0.0 && sum.is_finite() {
                            let inv = 1.0 / sum;
                            c = lane;
                            while c < cols {
                                *out.add(row * cols + c) *= inv;
                                c += thread::blockDim_x() as usize;
                            }
                            row += stride;
                            continue;
                        }
                    }
                }
                let mut maximum = f32::NEG_INFINITY;
                let mut c = lane;
                while c < cols {
                    let mut v = term * (1_f32 - *sim.add(sr as usize * cols + c));
                    if (2..=4).contains(&covs) {
                        if valid01 {
                            v += (*pen.add(cat0 as usize * cols + c)).ln();
                            v += (*pen.add(cat1 as usize * cols + c)).ln();
                            if covs >= 3 {
                                if valid2 {
                                    v += (*pen.add(cat2 as usize * cols + c)).ln();
                                } else {
                                    v = f32::NEG_INFINITY;
                                }
                            }
                            if covs == 4 {
                                if valid3 {
                                    v += (*pen.add(cat3 as usize * cols + c)).ln();
                                } else {
                                    v = f32::NEG_INFINITY;
                                }
                            }
                        } else {
                            v = f32::NEG_INFINITY;
                        }
                    } else {
                        for cov in 0..covs {
                            let cat = *cats.add(row * covs + cov);
                            if cat >= 0 && (cat as u64) < batches {
                                v += (*pen.add(cat as usize * cols + c)).ln();
                            } else {
                                v = f32::NEG_INFINITY;
                            }
                        }
                    }

                    *out.add(row * cols + c) = v;
                    maximum = maximum.max(v);
                    c += thread::blockDim_x() as usize;
                }
                maximum = row_reduce_f32(maximum, 1);
                let mut sum = 0_f32;
                c = lane;
                while c < cols {
                    let v = (*out.add(row * cols + c) - maximum).exp();
                    *out.add(row * cols + c) = v;
                    sum += v;
                    c += thread::blockDim_x() as usize;
                }
                sum = row_reduce_f32(sum, 0);
                let inv = 1.0 / sum;
                c = lane;
                while c < cols {
                    *out.add(row * cols + c) *= inv;
                    c += thread::blockDim_x() as usize;
                }
            }
            row += stride;
        }
    }
}
#[inline(always)]
unsafe fn harmony_pen_norm_impl_f64<const COVARIATES: usize>(
    sim: *const f64,
    pen: *const f64,
    cats: *const i32,
    idx: *const i32,
    out: *mut f64,
    rows: u64,
    cols: u64,
    covs: u64,
    sim_rows: u64,
    batches: u64,
    term: f64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        let cols = cols as usize;
        let covs = if COVARIATES == 0 {
            covs as usize
        } else {
            COVARIATES
        };
        while row < rows as usize {
            let sr = *idx.add(row);
            if sr >= 0 && (sr as u64) < sim_rows {
                // Marginal categories are invariant across this row's columns.
                let cat0 = *cats.add(row * covs);
                let cat1 = if covs >= 2 {
                    *cats.add(row * covs + 1)
                } else {
                    0
                };
                let cat2 = if covs >= 3 {
                    *cats.add(row * covs + 2)
                } else {
                    0
                };
                let cat3 = if covs >= 4 {
                    *cats.add(row * covs + 3)
                } else {
                    0
                };
                let valid01 =
                    cat0 >= 0 && (cat0 as u64) < batches && cat1 >= 0 && (cat1 as u64) < batches;
                let valid2 = cat2 >= 0 && (cat2 as u64) < batches;
                let valid3 = cat3 >= 0 && (cat3 as u64) < batches;
                // The single-covariate path avoids a logarithm per element.
                // Fall back to log-sum-exp if finite inputs under/overflow.
                if covs == 1 {
                    let cat = *cats.add(row);
                    if cat >= 0 && (cat as u64) < batches {
                        let mut sum = 0.0;
                        let mut c = lane;
                        while c < cols {
                            let v = (term * (1.0 - *sim.add(sr as usize * cols + c))).exp()
                                * *pen.add(cat as usize * cols + c);
                            *out.add(row * cols + c) = v;
                            sum += v;
                            c += thread::blockDim_x() as usize;
                        }
                        sum = row_reduce_f64(sum, 0);
                        if sum > 0.0 && sum.is_finite() {
                            let inv = 1.0 / sum;
                            c = lane;
                            while c < cols {
                                *out.add(row * cols + c) *= inv;
                                c += thread::blockDim_x() as usize;
                            }
                            row += stride;
                            continue;
                        }
                    }
                }
                let mut maximum = f64::NEG_INFINITY;
                let mut c = lane;
                while c < cols {
                    let mut v = term * (1_f64 - *sim.add(sr as usize * cols + c));
                    if (2..=4).contains(&covs) {
                        if valid01 {
                            v += (*pen.add(cat0 as usize * cols + c)).ln();
                            v += (*pen.add(cat1 as usize * cols + c)).ln();
                            if covs >= 3 {
                                if valid2 {
                                    v += (*pen.add(cat2 as usize * cols + c)).ln();
                                } else {
                                    v = f64::NEG_INFINITY;
                                }
                            }
                            if covs == 4 {
                                if valid3 {
                                    v += (*pen.add(cat3 as usize * cols + c)).ln();
                                } else {
                                    v = f64::NEG_INFINITY;
                                }
                            }
                        } else {
                            v = f64::NEG_INFINITY;
                        }
                    } else {
                        for cov in 0..covs {
                            let cat = *cats.add(row * covs + cov);
                            if cat >= 0 && (cat as u64) < batches {
                                v += (*pen.add(cat as usize * cols + c)).ln();
                            } else {
                                v = f64::NEG_INFINITY;
                            }
                        }
                    }

                    *out.add(row * cols + c) = v;
                    maximum = maximum.max(v);
                    c += thread::blockDim_x() as usize;
                }
                maximum = row_reduce_f64(maximum, 1);
                let mut sum = 0_f64;
                c = lane;
                while c < cols {
                    let v = (*out.add(row * cols + c) - maximum).exp();
                    *out.add(row * cols + c) = v;
                    sum += v;
                    c += thread::blockDim_x() as usize;
                }
                sum = row_reduce_f64(sum, 0);
                let inv = 1.0 / sum;
                c = lane;
                while c < cols {
                    *out.add(row * cols + c) *= inv;
                    c += thread::blockDim_x() as usize;
                }
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_entropy_f32(r: *const f32, out: *mut f32, rows: u64, cols: u64, sigma: f32) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let cols = cols as usize;
        while row < rows as usize {
            let mut s = 0_f32;
            let mut c = lane;
            while c < cols {
                s += *r.add(row * cols + c);
                c += 32;
            }
            s = sum_f32(s);
            let mut v = 0_f32;
            c = lane;
            while c < cols {
                let x = *r.add(row * cols + c) / s;
                v += x * (x + 1e-12_f32).ln();
                c += 32;
            }
            v = sum_f32(v);
            if lane == 0 {
                add_f32(out, sigma * v);
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_entropy_f64(r: *const f64, out: *mut f64, rows: u64, cols: u64, sigma: f64) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let cols = cols as usize;
        while row < rows as usize {
            let mut s = 0_f64;
            let mut c = lane;
            while c < cols {
                s += *r.add(row * cols + c);
                c += 32;
            }
            s = sum_f64(s);
            let mut v = 0_f64;
            c = lane;
            while c < cols {
                let x = *r.add(row * cols + c) / s;
                v += x * (x + 1e-12_f64).ln();
                c += 32;
            }
            v = sum_f64(v);
            if lane == 0 {
                add_f64(out, sigma * v);
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_diversity_f32(
    o: *const f32,
    e: *const f32,
    theta: *const f32,
    out: *mut f32,
    rows: u64,
    cols: u64,
    stabilized: u32,
    sigma: f32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let mut v = 0_f32;
        while i < (rows * cols) as usize {
            let ov = *o.add(i);
            let ev = *e.add(i);
            let numerator = ov + if stabilized != 0 { ev } else { 0_f32 } + 1_f32;
            v += *theta.add(i / cols as usize) * ov * (numerator / (ev + 1_f32)).ln();
            i += stride;
        }
        v = sum_f32(v);
        if thread::threadIdx_x().is_multiple_of(32) {
            add_f32(out, sigma * v);
        }
    }
}
#[kernel]
pub unsafe fn harmony_diversity_f64(
    o: *const f64,
    e: *const f64,
    theta: *const f64,
    out: *mut f64,
    rows: u64,
    cols: u64,
    stabilized: u32,
    sigma: f64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let mut v = 0_f64;
        while i < (rows * cols) as usize {
            let ov = *o.add(i);
            let ev = *e.add(i);
            let numerator = ov + if stabilized != 0 { ev } else { 0_f64 } + 1_f64;
            v += *theta.add(i / cols as usize) * ov * (numerator / (ev + 1_f64)).ln();
            i += stride;
        }
        v = sum_f64(v);
        if thread::threadIdx_x().is_multiple_of(32) {
            add_f64(out, sigma * v);
        }
    }
}
#[kernel]
pub unsafe fn harmony_marginal_f32(
    joint: *const f32,
    offsets: *const i32,
    indices: *const i32,
    out: *mut f32,
    rows: u64,
    cols: u64,
    joints: u64,
    nnz: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let b = i / cols as usize;
            let a = *offsets.add(b);
            let z = *offsets.add(b + 1);
            let mut v = 0_f32;
            if a >= 0 && z >= a && (z as u64) <= nnz {
                for p in a..z {
                    let j = *indices.add(p as usize);
                    if j >= 0 && (j as u64) < joints {
                        v += *joint.add(j as usize * cols as usize + i % cols as usize);
                    }
                }
            }
            *out.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_marginal_f64(
    joint: *const f64,
    offsets: *const i32,
    indices: *const i32,
    out: *mut f64,
    rows: u64,
    cols: u64,
    joints: u64,
    nnz: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (rows * cols) as usize {
            let b = i / cols as usize;
            let a = *offsets.add(b);
            let z = *offsets.add(b + 1);
            let mut v = 0_f64;
            if a >= 0 && z >= a && (z as u64) <= nnz {
                for p in a..z {
                    let j = *indices.add(p as usize);
                    if j >= 0 && (j as u64) < joints {
                        v += *joint.add(j as usize * cols as usize + i % cols as usize);
                    }
                }
            }
            *out.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_shuffle(keys: *mut u32, indices: *mut i32, n: u64, seed: u32) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < n as usize {
            let state = (i as u32 ^ seed)
                .wrapping_mul(747796405)
                .wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            *keys.add(i) = (word >> 22) ^ word;
            *indices.add(i) = i as i32;
            i += stride;
        }
    }
}

// Match the CUDA specializations for common covariate counts while retaining
// the general log-domain fallback for arbitrary marginal categories.
macro_rules! pen_norm_entry {
    ($name:ident, $implementation:ident, $value:ty, $specialization:literal) => {
        #[kernel]
        pub unsafe fn $name(
            sim: *const $value,
            pen: *const $value,
            cats: *const i32,
            idx: *const i32,
            out: *mut $value,
            rows: u64,
            cols: u64,
            covs: u64,
            sim_rows: u64,
            batches: u64,
            term: $value,
        ) {
            unsafe {
                $implementation::<$specialization>(
                    sim, pen, cats, idx, out, rows, cols, covs, sim_rows, batches, term,
                )
            }
        }
    };
}
pen_norm_entry!(harmony_pen_norm_f32, harmony_pen_norm_impl_f32, f32, 0);
pen_norm_entry!(harmony_pen_norm_cov1_f32, harmony_pen_norm_impl_f32, f32, 1);
pen_norm_entry!(harmony_pen_norm_cov2_f32, harmony_pen_norm_impl_f32, f32, 2);
pen_norm_entry!(harmony_pen_norm_cov3_f32, harmony_pen_norm_impl_f32, f32, 3);
pen_norm_entry!(harmony_pen_norm_cov4_f32, harmony_pen_norm_impl_f32, f32, 4);
pen_norm_entry!(harmony_pen_norm_f64, harmony_pen_norm_impl_f64, f64, 0);
pen_norm_entry!(harmony_pen_norm_cov1_f64, harmony_pen_norm_impl_f64, f64, 1);
pen_norm_entry!(harmony_pen_norm_cov2_f64, harmony_pen_norm_impl_f64, f64, 2);
pen_norm_entry!(harmony_pen_norm_cov3_f64, harmony_pen_norm_impl_f64, f64, 3);
pen_norm_entry!(harmony_pen_norm_cov4_f64, harmony_pen_norm_impl_f64, f64, 4);
