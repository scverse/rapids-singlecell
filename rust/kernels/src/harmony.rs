//! Harmony device kernels. Warp-owned rows avoid cross-block synchronization.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{kernel, ptx_asm, thread, warp};
#[inline(always)]
pub(crate) unsafe fn add_f32(p: *mut f32, v: f32) {
    // Native floating-point atomics preserve CUDA atomicAdd rounding and
    // avoid the CAS retry loop under highly contended category updates.
    let previous: f32;
    unsafe {
        ptx_asm!("atom.global.add.f32 %0, [%1], %2;",
            out("=f") previous, in("l") p as u64, in("f") v,
            clobber("memory"));
    }
    let _ = previous;
}
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
#[kernel]
pub unsafe fn harmony_colsum_f32(
    a: *const f32,
    out: *mut f32,
    rows: u64,
    cols: u64,
    accumulate: u32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < cols as usize {
            let mut v = 0_f32;
            let mut r = lane;
            while r < rows as usize {
                v += *a.add(r * cols as usize + row);
                r += 32;
            }
            v = sum_f32(v);
            if lane == 0 {
                if accumulate != 0 {
                    *out.add(row) += v;
                } else {
                    *out.add(row) = v;
                }
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_colsum_f64(
    a: *const f64,
    out: *mut f64,
    rows: u64,
    cols: u64,
    accumulate: u32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < cols as usize {
            let mut v = 0_f64;
            let mut r = lane;
            while r < rows as usize {
                v += *a.add(r * cols as usize + row);
                r += 32;
            }
            v = sum_f64(v);
            if lane == 0 {
                if accumulate != 0 {
                    *out.add(row) += v;
                } else {
                    *out.add(row) = v;
                }
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_colsum_i32(
    a: *const i32,
    out: *mut i32,
    rows: u64,
    cols: u64,
    accumulate: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < cols as usize {
            let mut v = 0i32;
            for r in 0..rows as usize {
                v = v.wrapping_add(*a.add(r * cols as usize + i));
            }
            if accumulate != 0 {
                *out.add(i) = (*out.add(i)).wrapping_add(v);
            } else {
                *out.add(i) = v;
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_normalize_f32(src: *const f32, dst: *mut f32, rows: u64, cols: u64, l2: u32) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let cols = cols as usize;
        while row < rows as usize {
            let mut v = 0_f32;
            let mut c = lane;
            while c < cols {
                let x = *src.add(row * cols + c);
                v += if l2 != 0 { x * x } else { x.abs() };
                c += 32;
            }
            v = sum_f32(v);
            let scale = if l2 != 0 {
                (1_f32 / v.sqrt()).min(1e12_f32)
            } else {
                1_f32 / v.max(1e-12_f32)
            };
            c = lane;
            while c < cols {
                *dst.add(row * cols + c) = *src.add(row * cols + c) * scale;
                c += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_normalize_f64(src: *const f64, dst: *mut f64, rows: u64, cols: u64, l2: u32) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let cols = cols as usize;
        while row < rows as usize {
            let mut v = 0_f64;
            let mut c = lane;
            while c < cols {
                let x = *src.add(row * cols + c);
                v += if l2 != 0 { x * x } else { x.abs() };
                c += 32;
            }
            v = sum_f64(v);
            let scale = if l2 != 0 {
                (1_f64 / v.sqrt()).min(1e12_f64)
            } else {
                1_f64 / v.max(1e-12_f64)
            };
            c = lane;
            while c < cols {
                *dst.add(row * cols + c) = *src.add(row * cols + c) * scale;
                c += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_kmeans_f32(r: *const f32, dot: *const f32, out: *mut f32, n: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let mut v = 0_f32;
        while i < n as usize {
            v += *r.add(i) * 2_f32 * (1_f32 - *dot.add(i));
            i += stride;
        }
        v = sum_f32(v);
        if thread::threadIdx_x().is_multiple_of(32) {
            add_f32(out, v);
        }
    }
}
#[kernel]
pub unsafe fn harmony_kmeans_f64(r: *const f64, dot: *const f64, out: *mut f64, n: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let mut v = 0_f64;
        while i < n as usize {
            v += *r.add(i) * 2_f64 * (1_f64 - *dot.add(i));
            i += stride;
        }
        v = sum_f64(v);
        if thread::threadIdx_x().is_multiple_of(32) {
            add_f64(out, v);
        }
    }
}
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
#[kernel]
pub unsafe fn harmony_pen_norm_f32(
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
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (cols, covs) = (cols as usize, covs as usize);
        while row < rows as usize {
            let sr = *idx.add(row);
            if sr >= 0 && (sr as u64) < sim_rows {
                let mut maximum = f32::NEG_INFINITY;
                let mut c = lane;
                while c < cols {
                    let mut v = term * (1_f32 - *sim.add(sr as usize * cols + c));
                    for cov in 0..covs {
                        let cat = *cats.add(row * covs + cov);
                        if cat >= 0 && (cat as u64) < batches {
                            v += (*pen.add(cat as usize * cols + c)).ln();
                        } else {
                            v = f32::NEG_INFINITY;
                        }
                    }
                    *out.add(row * cols + c) = v;
                    maximum = maximum.max(v);
                    c += 32;
                }
                maximum = max_f32(maximum);
                let mut sum = 0_f32;
                c = lane;
                while c < cols {
                    let v = (*out.add(row * cols + c) - maximum).exp();
                    *out.add(row * cols + c) = v;
                    sum += v;
                    c += 32;
                }
                sum = sum_f32(sum);
                c = lane;
                while c < cols {
                    *out.add(row * cols + c) /= sum;
                    c += 32;
                }
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_pen_norm_f64(
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
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (cols, covs) = (cols as usize, covs as usize);
        while row < rows as usize {
            let sr = *idx.add(row);
            if sr >= 0 && (sr as u64) < sim_rows {
                let mut maximum = f64::NEG_INFINITY;
                let mut c = lane;
                while c < cols {
                    let mut v = term * (1_f64 - *sim.add(sr as usize * cols + c));
                    for cov in 0..covs {
                        let cat = *cats.add(row * covs + cov);
                        if cat >= 0 && (cat as u64) < batches {
                            v += (*pen.add(cat as usize * cols + c)).ln();
                        } else {
                            v = f64::NEG_INFINITY;
                        }
                    }
                    *out.add(row * cols + c) = v;
                    maximum = maximum.max(v);
                    c += 32;
                }
                maximum = max_f64(maximum);
                let mut sum = 0_f64;
                c = lane;
                while c < cols {
                    let v = (*out.add(row * cols + c) - maximum).exp();
                    *out.add(row * cols + c) = v;
                    sum += v;
                    c += 32;
                }
                sum = sum_f64(sum);
                c = lane;
                while c < cols {
                    *out.add(row * cols + c) /= sum;
                    c += 32;
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
