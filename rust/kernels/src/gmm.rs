//! Gaussian-mixture device kernels and per-gene fused EM.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
use super::harmony::{max_f32, max_f64, sum_f32, sum_f64};
use cuda_device::{kernel, ptx_asm, thread, warp};

// Match the original 256-thread halving tree: grouping each warp first
// changes EM's convergence decision for poorly separated components. The first
// warp gathers the upper three levels in registers, then performs the remaining
// five levels with synchronized shuffles in the same addition order.
macro_rules! em_reduce4 {
    ($name:ident, $value:ty, $shuffle:ident) => {
        #[inline(always)]
        fn $name(mut values: [$value; 4]) -> [$value; 4] {
            static mut SHARED: cuda_device::SharedArray<$value, 1024> =
                cuda_device::SharedArray::UNINIT;
            let tid = thread::threadIdx_x() as usize;
            unsafe {
                SHARED[tid] = values[0];
                SHARED[256 + tid] = values[1];
                SHARED[512 + tid] = values[2];
                SHARED[768 + tid] = values[3];
            }
            thread::sync_threads();
            if tid < 32 {
                let mut component = 0;
                while component < 4 {
                    let index = component * 256 + tid;
                    unsafe {
                        let low = (SHARED[index] + SHARED[index + 128])
                            + (SHARED[index + 64] + SHARED[index + 192]);
                        let high = (SHARED[index + 32] + SHARED[index + 160])
                            + (SHARED[index + 96] + SHARED[index + 224]);
                        values[component] = low + high;
                    }
                    component += 1;
                }
                let mut offset = 16;
                while offset > 0 {
                    values[0] += warp::$shuffle(u32::MAX, values[0], offset);
                    values[1] += warp::$shuffle(u32::MAX, values[1], offset);
                    values[2] += warp::$shuffle(u32::MAX, values[2], offset);
                    values[3] += warp::$shuffle(u32::MAX, values[3], offset);
                    offset /= 2;
                }
                if tid == 0 {
                    unsafe {
                        SHARED[0] = values[0];
                        SHARED[256] = values[1];
                        SHARED[512] = values[2];
                        SHARED[768] = values[3];
                    }
                }
            }
            thread::sync_threads();
            let result = unsafe { [SHARED[0], SHARED[256], SHARED[512], SHARED[768]] };
            // Every consumer must finish reading before the next reduction
            // reuses this storage; this also protects grid-stride gene reuse.
            thread::sync_threads();
            result
        }
    };
}
em_reduce4!(em_reduce4_f32, f32, shuffle_down_f32_sync);
em_reduce4!(em_reduce4_f64, f64, shuffle_down_f64_sync);
// Inputs are validated disjoint from all projection/EM outputs. Preserve
// CUDA's read-only cache path across the serial population and dot-product scans.
#[inline(always)]
unsafe fn read_only_mask(pointer: *const u8) -> u8 {
    let value: u32;
    unsafe {
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=r") value, in("l") pointer as u64, clobber("memory"));
    }
    value as u8
}
macro_rules! projection_read {
    ($load:ident, $sum:ident, $dot:ident, $value:ty, $instruction:literal, $constraint:literal) => {
        #[inline(always)]
        unsafe fn $load(pointer: *const $value) -> $value {
            let value: $value;
            unsafe { ptx_asm!($instruction, out($constraint) value, in("l") pointer as u64, clobber("memory")); }
            value
        }
        #[inline(always)]
        unsafe fn $dot(dat: *const $value, direction: *const $value, features: usize) -> $value {
            unsafe {
                let mut sum = 0.0 as $value;
                let mut feature = 0;
                // Gather direction entries before the read-only global loads,
                // exposing the original four-feature shared-load/FMA sequence.
                // Accumulation order stays serial across every feature.
                while feature + 3 < features {
                    let v0 = *direction.add(feature);
                    let v1 = *direction.add(feature + 1);
                    let v2 = *direction.add(feature + 2);
                    let v3 = *direction.add(feature + 3);
                    sum = $load(dat.add(feature)).mul_add(v0, sum);
                    sum = $load(dat.add(feature + 1)).mul_add(v1, sum);
                    sum = $load(dat.add(feature + 2)).mul_add(v2, sum);
                    sum = $load(dat.add(feature + 3)).mul_add(v3, sum);
                    feature += 4;
                }
                while feature < features {
                    sum = $load(dat.add(feature)).mul_add(*direction.add(feature), sum);
                    feature += 1;
                }
                sum
            }
        }
        #[inline(always)]
        unsafe fn $sum(dat: *const $value, guide: *const u8, n: usize, k: usize, column: usize) -> $value {
            unsafe {
                let mut sum = 0.0 as $value;
                let mut cell = 0;
                // Keep the original sequential sum and expose four independent
                // read-only loads per loop, matching NVCC's population unroll.
                while cell + 3 < n {
                    if read_only_mask(guide.add(cell)) != 0 { sum += $load(dat.add(cell * k + column)); }
                    if read_only_mask(guide.add(cell + 1)) != 0 { sum += $load(dat.add((cell + 1) * k + column)); }
                    if read_only_mask(guide.add(cell + 2)) != 0 { sum += $load(dat.add((cell + 2) * k + column)); }
                    if read_only_mask(guide.add(cell + 3)) != 0 { sum += $load(dat.add((cell + 3) * k + column)); }
                    cell += 4;
                }
                while cell < n {
                    if read_only_mask(guide.add(cell)) != 0 { sum += $load(dat.add(cell * k + column)); }
                    cell += 1;
                }
                sum
            }
        }
    };
}
projection_read!(
    read_only_f32,
    guide_sum_f32,
    projection_dot_f32,
    f32,
    "ld.global.nc.f32 %0, [%1];",
    "=f"
);
projection_read!(
    read_only_f64,
    guide_sum_f64,
    projection_dot_f64,
    f64,
    "ld.global.nc.f64 %0, [%1];",
    "=d"
);

// Projection statistics are identical for every lane in a gene. Compute
// their divisions once and broadcast through the same warp/block split as EM.
#[inline(always)]
unsafe fn projection_parameters_f32<const THREADS: usize>(
    sn: f32,
    sn2: f32,
    sg: f32,
    sg2: f32,
    ng: f32,
    nn: f32,
    lane: usize,
) -> [f32; 4] {
    static mut PARAMETERS: cuda_device::SharedArray<f32, 4> = cuda_device::SharedArray::UNINIT;
    let mut values = [0.0_f32; 4];
    if lane == 0 {
        values[0] = sn / nn.max(1.0);
        values[1] = if nn > 1.0 {
            (sn2 - sn * sn / nn) / (nn - 1.0)
        } else {
            1e-12
        };
        values[2] = sg / ng.max(1.0);
        values[3] = if ng > 1.0 {
            (sg2 - sg * sg / ng) / (ng - 1.0)
        } else {
            1e-12
        };
    }
    if THREADS == 32 {
        [
            warp::shuffle_f32_sync(u32::MAX, values[0], 0),
            warp::shuffle_f32_sync(u32::MAX, values[1], 0),
            warp::shuffle_f32_sync(u32::MAX, values[2], 0),
            warp::shuffle_f32_sync(u32::MAX, values[3], 0),
        ]
    } else {
        unsafe {
            if lane == 0 {
                PARAMETERS[0] = values[0];
                PARAMETERS[1] = values[1];
                PARAMETERS[2] = values[2];
                PARAMETERS[3] = values[3];
            }
            thread::sync_threads();
            [PARAMETERS[0], PARAMETERS[1], PARAMETERS[2], PARAMETERS[3]]
        }
    }
}

#[inline(always)]
unsafe fn projection_parameters_f64<const THREADS: usize>(
    sn: f64,
    sn2: f64,
    sg: f64,
    sg2: f64,
    ng: f64,
    nn: f64,
    lane: usize,
) -> [f64; 4] {
    static mut PARAMETERS: cuda_device::SharedArray<f64, 4> = cuda_device::SharedArray::UNINIT;
    let mut values = [0.0_f64; 4];
    if lane == 0 {
        values[0] = sn / nn.max(1.0);
        values[1] = if nn > 1.0 {
            (sn2 - sn * sn / nn) / (nn - 1.0)
        } else {
            1e-12
        };
        values[2] = sg / ng.max(1.0);
        values[3] = if ng > 1.0 {
            (sg2 - sg * sg / ng) / (ng - 1.0)
        } else {
            1e-12
        };
    }
    if THREADS == 32 {
        [
            warp::shuffle_f64_sync(u32::MAX, values[0], 0),
            warp::shuffle_f64_sync(u32::MAX, values[1], 0),
            warp::shuffle_f64_sync(u32::MAX, values[2], 0),
            warp::shuffle_f64_sync(u32::MAX, values[3], 0),
        ]
    } else {
        unsafe {
            if lane == 0 {
                PARAMETERS[0] = values[0];
                PARAMETERS[1] = values[1];
                PARAMETERS[2] = values[2];
                PARAMETERS[3] = values[3];
            }
            thread::sync_threads();
            [PARAMETERS[0], PARAMETERS[1], PARAMETERS[2], PARAMETERS[3]]
        }
    }
}

unsafe fn em_f32<const THREADS: usize>(
    p: *const f32,
    out: *mut f32,
    n: usize,
    m0: f32,
    v0: f32,
    mut m1: f32,
    mut v1: f32,
    max_iter: u32,
    tol: f32,
    reg: f32,
    lane: usize,
) -> (f32, f32, f32) {
    static mut PARAMETERS: cuda_device::SharedArray<f32, 4> = cuda_device::SharedArray::UNINIT;
    unsafe {
        let mut w = 0.5_f32;
        let mut prev = -1e30_f32;
        let v0 = v0.max(reg);
        for _ in 0..max_iter {
            let c0 = (1_f32 - w).max(1e-10_f32).ln() - 0.5_f32 * v0.ln() - 0.918_938_5_f32;
            let c1 = w.max(1e-10_f32).ln() - 0.5_f32 * v1.ln() - 0.918_938_5_f32;
            let (mut n1, mut s1, mut s2, mut ll) = (0_f32, 0_f32, 0_f32, 0_f32);
            let mut i = lane;
            while i < n {
                let y = *p.add(i);
                let d0 = y - m0;
                let d1 = y - m1;
                let a = c0 - 0.5_f32 / v0 * d0 * d0;
                let b = c1 - 0.5_f32 / v1 * d1 * d1;
                let mx = a.max(b);
                let l = mx + ((a - mx).exp() + (b - mx).exp()).ln();
                let r = (b - l).exp();
                n1 += r;
                s1 += r * y;
                s2 += r * y * y;
                ll += l;
                i += THREADS;
            }
            [n1, s1, s2, ll] = if THREADS == 32 {
                [sum_f32(n1), sum_f32(s1), sum_f32(s2), sum_f32(ll)]
            } else {
                em_reduce4_f32([n1, s1, s2, ll])
            };
            // One lane updates mixture parameters, as in the original EM.
            // Small populations share within their warp; large populations
            // broadcast to all eight warps through block-owned storage.
            let mut done = 0.0_f32;
            if lane == 0 {
                let mean = ll / n as f32;
                if (mean - prev).abs() < tol {
                    done = 1.0;
                } else {
                    prev = mean;
                    let inv = 1_f32 / n1.max(1e-12_f32);
                    m1 = s1 * inv;
                    v1 = (s2 * inv - m1 * m1 + reg).max(reg);
                    w = n1 / n as f32;
                }
            }
            if THREADS == 32 {
                m1 = warp::shuffle_f32_sync(u32::MAX, m1, 0);
                v1 = warp::shuffle_f32_sync(u32::MAX, v1, 0);
                w = warp::shuffle_f32_sync(u32::MAX, w, 0);
                done = warp::shuffle_f32_sync(u32::MAX, done, 0);
            } else {
                if lane == 0 {
                    PARAMETERS[0] = m1;
                    PARAMETERS[1] = v1;
                    PARAMETERS[2] = w;
                    PARAMETERS[3] = done;
                }
                thread::sync_threads();
                m1 = PARAMETERS[0];
                v1 = PARAMETERS[1];
                w = PARAMETERS[2];
                done = PARAMETERS[3];
            }
            if done != 0.0 {
                break;
            }
        }
        let c0 = (1_f32 - w).max(1e-10_f32).ln() - 0.5_f32 * v0.ln() - 0.918_938_5_f32;
        let c1 = w.max(1e-10_f32).ln() - 0.5_f32 * v1.ln() - 0.918_938_5_f32;
        let mut i = lane;
        while i < n {
            let y = *p.add(i);
            let a = y - m0;
            let b = y - m1;
            let lp0 = c0 - 0.5_f32 / v0 * a * a;
            let lp1 = c1 - 0.5_f32 / v1 * b * b;
            *out.add(i) = 1_f32 / (1_f32 + (lp0 - lp1).exp());
            i += THREADS;
        }
        (m1, v1, w)
    }
}

unsafe fn em_f64<const THREADS: usize>(
    p: *const f64,
    out: *mut f64,
    n: usize,
    m0: f64,
    v0: f64,
    mut m1: f64,
    mut v1: f64,
    max_iter: u32,
    tol: f64,
    reg: f64,
    lane: usize,
) -> (f64, f64, f64) {
    static mut PARAMETERS: cuda_device::SharedArray<f64, 4> = cuda_device::SharedArray::UNINIT;
    unsafe {
        let mut w = 0.5_f64;
        let mut prev = -1e30_f64;
        let v0 = v0.max(reg);
        for _ in 0..max_iter {
            let c0 = (1_f64 - w).max(1e-10_f64).ln() - 0.5_f64 * v0.ln() - 0.9189385332046727_f64;
            let c1 = w.max(1e-10_f64).ln() - 0.5_f64 * v1.ln() - 0.9189385332046727_f64;
            let (mut n1, mut s1, mut s2, mut ll) = (0_f64, 0_f64, 0_f64, 0_f64);
            let mut i = lane;
            while i < n {
                let y = *p.add(i);
                let d0 = y - m0;
                let d1 = y - m1;
                let a = c0 - 0.5_f64 / v0 * d0 * d0;
                let b = c1 - 0.5_f64 / v1 * d1 * d1;
                let mx = a.max(b);
                let l = mx + ((a - mx).exp() + (b - mx).exp()).ln();
                let r = (b - l).exp();
                n1 += r;
                s1 += r * y;
                s2 += r * y * y;
                ll += l;
                i += THREADS;
            }
            [n1, s1, s2, ll] = if THREADS == 32 {
                [sum_f64(n1), sum_f64(s1), sum_f64(s2), sum_f64(ll)]
            } else {
                em_reduce4_f64([n1, s1, s2, ll])
            };
            // One lane updates mixture parameters, as in the original EM.
            // Small populations share within their warp; large populations
            // broadcast to all eight warps through block-owned storage.
            let mut done = 0.0_f64;
            if lane == 0 {
                let mean = ll / n as f64;
                if (mean - prev).abs() < tol {
                    done = 1.0;
                } else {
                    prev = mean;
                    let inv = 1_f64 / n1.max(1e-12_f64);
                    m1 = s1 * inv;
                    v1 = (s2 * inv - m1 * m1 + reg).max(reg);
                    w = n1 / n as f64;
                }
            }
            if THREADS == 32 {
                m1 = warp::shuffle_f64_sync(u32::MAX, m1, 0);
                v1 = warp::shuffle_f64_sync(u32::MAX, v1, 0);
                w = warp::shuffle_f64_sync(u32::MAX, w, 0);
                done = warp::shuffle_f64_sync(u32::MAX, done, 0);
            } else {
                if lane == 0 {
                    PARAMETERS[0] = m1;
                    PARAMETERS[1] = v1;
                    PARAMETERS[2] = w;
                    PARAMETERS[3] = done;
                }
                thread::sync_threads();
                m1 = PARAMETERS[0];
                v1 = PARAMETERS[1];
                w = PARAMETERS[2];
                done = PARAMETERS[3];
            }
            if done != 0.0 {
                break;
            }
        }
        let c0 = (1_f64 - w).max(1e-10_f64).ln() - 0.5_f64 * v0.ln() - 0.9189385332046727_f64;
        let c1 = w.max(1e-10_f64).ln() - 0.5_f64 * v1.ln() - 0.9189385332046727_f64;
        let mut i = lane;
        while i < n {
            let y = *p.add(i);
            let a = y - m0;
            let b = y - m1;
            let lp0 = c0 - 0.5_f64 / v0 * a * a;
            let lp1 = c1 - 0.5_f64 / v1 * b * b;
            *out.add(i) = 1_f64 / (1_f64 + (lp0 - lp1).exp());
            i += THREADS;
        }
        (m1, v1, w)
    }
}
#[kernel]
pub unsafe fn gmm_logprob_f32(
    x: *const f32,
    w: *const f32,
    means: *const f32,
    prec: *const f32,
    det: *const f32,
    out: *mut f32,
    n: u64,
    d: u64,
    k: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (n, d, k) = (n as usize, d as usize, k as usize);
        while row < n * k {
            let cell = row / k;
            let cl = row % k;
            let mut mahal = 0_f32;
            let mut j = lane;
            while j < d {
                let mut y = 0_f32;
                for dd in 0..=j {
                    y += (*x.add(cell * d + dd) - *means.add(cl * d + dd))
                        * *prec.add((cl * d + dd) * d + j);
                }
                mahal += y * y;
                j += 32;
            }
            mahal = sum_f32(mahal);
            if lane == 0 {
                *out.add(row) =
                    -0.5_f32 * d as f32 * 1.837_877_f32 + *det.add(cl) + (*w.add(cl)).ln()
                        - 0.5_f32 * mahal;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_logprob_f64(
    x: *const f64,
    w: *const f64,
    means: *const f64,
    prec: *const f64,
    det: *const f64,
    out: *mut f64,
    n: u64,
    d: u64,
    k: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (n, d, k) = (n as usize, d as usize, k as usize);
        while row < n * k {
            let cell = row / k;
            let cl = row % k;
            let mut mahal = 0_f64;
            let mut j = lane;
            while j < d {
                let mut y = 0_f64;
                for dd in 0..=j {
                    y += (*x.add(cell * d + dd) - *means.add(cl * d + dd))
                        * *prec.add((cl * d + dd) * d + j);
                }
                mahal += y * y;
                j += 32;
            }
            mahal = sum_f64(mahal);
            if lane == 0 {
                *out.add(row) =
                    -0.5_f64 * d as f64 * 1.8378770664093453_f64 + *det.add(cl) + (*w.add(cl)).ln()
                        - 0.5_f64 * mahal;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_normalize_f32(lp: *const f32, resp: *mut f32, ll: *mut f32, n: u64, k: u64) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let k = k as usize;
        while row < n as usize {
            let mut mx = f32::NEG_INFINITY;
            let mut cl = lane;
            while cl < k {
                mx = mx.max(*lp.add(row * k + cl));
                cl += 32;
            }
            mx = max_f32(mx);
            let mut sum = 0_f32;
            cl = lane;
            while cl < k {
                sum += (*lp.add(row * k + cl) - mx).exp();
                cl += 32;
            }
            sum = sum_f32(sum);
            let logtotal = sum.ln() + mx;
            if lane == 0 {
                *ll.add(row) = logtotal;
            }
            cl = lane;
            while cl < k {
                *resp.add(row * k + cl) = (*lp.add(row * k + cl) - logtotal).exp();
                cl += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_normalize_f64(lp: *const f64, resp: *mut f64, ll: *mut f64, n: u64, k: u64) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let k = k as usize;
        while row < n as usize {
            let mut mx = f64::NEG_INFINITY;
            let mut cl = lane;
            while cl < k {
                mx = mx.max(*lp.add(row * k + cl));
                cl += 32;
            }
            mx = max_f64(mx);
            let mut sum = 0_f64;
            cl = lane;
            while cl < k {
                sum += (*lp.add(row * k + cl) - mx).exp();
                cl += 32;
            }
            sum = sum_f64(sum);
            let logtotal = sum.ln() + mx;
            if lane == 0 {
                *ll.add(row) = logtotal;
            }
            cl = lane;
            while cl < k {
                *resp.add(row * k + cl) = (*lp.add(row * k + cl) - logtotal).exp();
                cl += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_center_f32(
    x: *const f32,
    means: *const f32,
    resp: *const f32,
    out: *mut f32,
    n: u64,
    d: u64,
    k: u64,
    cl: u64,
    weighted: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (n * d) as usize {
            let mut v = *x.add(i) - *means.add(cl as usize * d as usize + i % d as usize);
            if weighted != 0 {
                v *= (*resp.add(i / d as usize * k as usize + cl as usize)).sqrt();
            }
            *out.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_center_f64(
    x: *const f64,
    means: *const f64,
    resp: *const f64,
    out: *mut f64,
    n: u64,
    d: u64,
    k: u64,
    cl: u64,
    weighted: u32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (n * d) as usize {
            let mut v = *x.add(i) - *means.add(cl as usize * d as usize + i % d as usize);
            if weighted != 0 {
                v *= (*resp.add(i / d as usize * k as usize + cl as usize)).sqrt();
            }
            *out.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_logprob_y_f32(
    y: *const f32,
    w: *const f32,
    det: *const f32,
    out: *mut f32,
    n: u64,
    d: u64,
    k: u64,
    cl: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let d = d as usize;
        while row < n as usize {
            let mut v = 0_f32;
            let mut j = lane;
            while j < d {
                let x = *y.add(row * d + j);
                v += x * x;
                j += 32;
            }
            v = sum_f32(v);
            if lane == 0 {
                *out.add(row * k as usize + cl as usize) = -0.5_f32 * d as f32 * 1.837_877_f32
                    + *det.add(cl as usize)
                    + (*w.add(cl as usize)).ln()
                    - 0.5_f32 * v;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_logprob_y_f64(
    y: *const f64,
    w: *const f64,
    det: *const f64,
    out: *mut f64,
    n: u64,
    d: u64,
    k: u64,
    cl: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let d = d as usize;
        while row < n as usize {
            let mut v = 0_f64;
            let mut j = lane;
            while j < d {
                let x = *y.add(row * d + j);
                v += x * x;
                j += 32;
            }
            v = sum_f64(v);
            if lane == 0 {
                *out.add(row * k as usize + cl as usize) =
                    -0.5_f64 * d as f64 * 1.8378770664093453_f64
                        + *det.add(cl as usize)
                        + (*w.add(cl as usize)).ln()
                        - 0.5_f64 * v;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_means_f32(
    nk: *const f32,
    num: *const f32,
    w: *mut f32,
    means: *mut f32,
    n: u64,
    d: u64,
    k: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (k * d) as usize {
            let cl = i / d as usize;
            let count = *nk.add(cl) + 10_f32 * f32::EPSILON;
            *means.add(i) = *num.add(i) / count;
            if i.is_multiple_of(d as usize) {
                *w.add(cl) = count / n as f32;
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_means_f64(
    nk: *const f64,
    num: *const f64,
    w: *mut f64,
    means: *mut f64,
    n: u64,
    d: u64,
    k: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (k * d) as usize {
            let cl = i / d as usize;
            let count = *nk.add(cl) + 10_f64 * f64::EPSILON;
            *means.add(i) = *num.add(i) / count;
            if i.is_multiple_of(d as usize) {
                *w.add(cl) = count / n as f64;
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_cov_f32(nk: *const f32, cov: *mut f32, d: u64, k: u64, reg: f32) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let d = d as usize;
        while i < k as usize * d * d {
            let cl = i / (d * d);
            let r = (i / d) % d;
            let c = i % d;
            if r <= c {
                let value = *cov.add(cl * d * d + c * d + r)
                    / (*nk.add(cl) + 10_f32 * f32::EPSILON)
                    + if r == c { reg } else { 0_f32 };
                *cov.add(i) = value;
                if r != c {
                    *cov.add(cl * d * d + c * d + r) = value;
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_cov_f64(nk: *const f64, cov: *mut f64, d: u64, k: u64, reg: f64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let d = d as usize;
        while i < k as usize * d * d {
            let cl = i / (d * d);
            let r = (i / d) % d;
            let c = i % d;
            if r <= c {
                let value = *cov.add(cl * d * d + c * d + r)
                    / (*nk.add(cl) + 10_f64 * f64::EPSILON)
                    + if r == c { reg } else { 0_f64 };
                *cov.add(i) = value;
                if r != c {
                    *cov.add(cl * d * d + c * d + r) = value;
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_identity_f32(a: *mut f32, d: u64, k: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (k * d * d) as usize {
            *a.add(i) = if (i / d as usize) % d as usize == i % d as usize {
                1_f32
            } else {
                0_f32
            };
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_identity_f64(a: *mut f64, d: u64, k: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < (k * d * d) as usize {
            *a.add(i) = if (i / d as usize) % d as usize == i % d as usize {
                1_f64
            } else {
                0_f64
            };
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_logdet_f32(a: *const f32, out: *mut f32, d: u64, k: u64) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let d = d as usize;
        while row < k as usize {
            let mut v = 0_f32;
            let mut j = lane;
            while j < d {
                v += (*a.add(row * d * d + j * d + j)).ln();
                j += 32;
            }
            v = sum_f32(v);
            if lane == 0 {
                *out.add(row) = v;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_logdet_f64(a: *const f64, out: *mut f64, d: u64, k: u64) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let d = d as usize;
        while row < k as usize {
            let mut v = 0_f64;
            let mut j = lane;
            while j < d {
                v += (*a.add(row * d * d + j * d + j)).ln();
                j += 32;
            }
            v = sum_f64(v);
            if lane == 0 {
                *out.add(row) = v;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_spherical_f32(
    p: *const f32,
    offsets: *const i32,
    m0: *const f32,
    v0: *const f32,
    m1: *const f32,
    v1: *const f32,
    resp: *mut f32,
    mo: *mut f32,
    vo: *mut f32,
    wo: *mut f32,
    genes: u64,
    len: u64,
    max_iter: u32,
    tol: f32,
    reg: f32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f32);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f32::<32>(
                    p.add(start as usize),
                    resp.add(start as usize),
                    (end - start) as usize,
                    *m0.add(row),
                    *v0.add(row),
                    init_m,
                    init_v.max(reg),
                    max_iter,
                    tol,
                    reg,
                    lane,
                );
            }
            if lane == 0 {
                *mo.add(row) = result.0;
                *vo.add(row) = result.1;
                *wo.add(row) = result.2;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn gmm_spherical_f64(
    p: *const f64,
    offsets: *const i32,
    m0: *const f64,
    v0: *const f64,
    m1: *const f64,
    v1: *const f64,
    resp: *mut f64,
    mo: *mut f64,
    vo: *mut f64,
    wo: *mut f64,
    genes: u64,
    len: u64,
    max_iter: u32,
    tol: f64,
    reg: f64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f64);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f64::<32>(
                    p.add(start as usize),
                    resp.add(start as usize),
                    (end - start) as usize,
                    *m0.add(row),
                    *v0.add(row),
                    init_m,
                    init_v.max(reg),
                    max_iter,
                    tol,
                    reg,
                    lane,
                );
            }
            if lane == 0 {
                *mo.add(row) = result.0;
                *vo.add(row) = result.1;
                *wo.add(row) = result.2;
            }
            row += stride;
        }
    }
}
#[inline(always)]
unsafe fn gmm_project_impl_f32<const SHARED: bool>(
    dat: *const f32,
    dat_off: *const i64,
    ns: *const i32,
    ks: *const i32,
    cells: *const i32,
    features: *const i32,
    ntmean: *const f32,
    guide: *const u8,
    nt: *const u8,
    active: *const i32,
    p: *mut f32,
    resp: *mut f32,
    vec: *mut f32,
    nactive: u64,
    genes: u64,
    maxk: u64,
    datlen: u64,
    celllen: u64,
    featlen: u64,
    max_iter: u32,
    tol: f32,
    reg: f32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < nactive as usize {
            let gene = *active.add(row);
            if gene >= 0 && (gene as u64) < genes {
                let g = gene as usize;
                let n = *ns.add(g);
                let k = *ks.add(g);
                let co = *cells.add(g);
                let feature_offset = *features.add(g);
                let off = *dat_off.add(g);
                if n > 0
                    && k > 0
                    && co >= 0
                    && feature_offset >= 0
                    && off >= 0
                    && (k as u64) <= maxk
                    && (co as u64 + n as u64) <= celllen
                    && (feature_offset as u64 + k as u64) <= featlen
                    && (off as u64 + n as u64 * k as u64) <= datlen
                {
                    let (n, k, co, feature_offset, off) = (
                        n as usize,
                        k as usize,
                        co as usize,
                        feature_offset as usize,
                        off as usize,
                    );
                    let (mut ng, mut nn) = (0_f32, 0_f32);
                    let mut i = lane;
                    while i < n {
                        ng += if read_only_mask(guide.add(co + i)) != 0 {
                            1_f32
                        } else {
                            0_f32
                        };
                        nn += if read_only_mask(nt.add(co + i)) != 0 {
                            1_f32
                        } else {
                            0_f32
                        };
                        i += 32;
                    }
                    ng = sum_f32(ng);
                    nn = sum_f32(nn);
                    // A null global workspace selects the original shared
                    // direction vector. Warp kernels reserve one vector per
                    // warp; the block variant reserves one for the whole gene.
                    let v = if SHARED {
                        cuda_device::DynamicSharedArray::<f32>::get()
                            .add((thread::threadIdx_x() as usize / 32) * maxk as usize)
                    } else {
                        vec.add(row * maxk as usize)
                    };
                    let inv_ng = 1_f32 / ng.max(1_f32);
                    let mut j = lane;
                    while j < k {
                        let s = guide_sum_f32(dat.add(off), guide.add(co), n, k, j);
                        *v.add(j) = s * inv_ng - read_only_f32(ntmean.add(feature_offset + j));
                        j += 32;
                    }
                    warp::sync_mask(u32::MAX);
                    let mut vv = 0_f32;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv = x.mul_add(x, vv);
                        j += 32;
                    }
                    vv = sum_f32(vv).max(1e-12_f32);
                    let inv_vv = 1_f32 / vv;
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f32, 0_f32, 0_f32, 0_f32);
                    while i < n {
                        let s = projection_dot_f32(dat.add(off + i * k), v, k);
                        let y = s * inv_vv;
                        *p.add(co + i) = y;
                        if read_only_mask(guide.add(co + i)) != 0 {
                            sg += y;
                            sg2 = y.mul_add(y, sg2);
                        }
                        if read_only_mask(nt.add(co + i)) != 0 {
                            sn += y;
                            sn2 = y.mul_add(y, sn2);
                        }
                        i += 32;
                    }
                    sg = sum_f32(sg);
                    sg2 = sum_f32(sg2);
                    sn = sum_f32(sn);
                    sn2 = sum_f32(sn2);
                    let [m0, v0, m1, v1] =
                        projection_parameters_f32::<32>(sn, sn2, sg, sg2, ng, nn, lane);
                    warp::sync_mask(u32::MAX);
                    let _ = em_f32::<32>(
                        p.add(co),
                        resp.add(co),
                        n,
                        m0,
                        v0.max(1e-12_f32),
                        m1,
                        v1.max(1e-12_f32),
                        max_iter,
                        tol,
                        reg,
                        lane,
                    );
                }
            }
            row += stride;
        }
    }
}
#[inline(always)]
unsafe fn gmm_project_impl_f64<const SHARED: bool>(
    dat: *const f64,
    dat_off: *const i64,
    ns: *const i32,
    ks: *const i32,
    cells: *const i32,
    features: *const i32,
    ntmean: *const f64,
    guide: *const u8,
    nt: *const u8,
    active: *const i32,
    p: *mut f64,
    resp: *mut f64,
    vec: *mut f64,
    nactive: u64,
    genes: u64,
    maxk: u64,
    datlen: u64,
    celllen: u64,
    featlen: u64,
    max_iter: u32,
    tol: f64,
    reg: f64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = (thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize)
            / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < nactive as usize {
            let gene = *active.add(row);
            if gene >= 0 && (gene as u64) < genes {
                let g = gene as usize;
                let n = *ns.add(g);
                let k = *ks.add(g);
                let co = *cells.add(g);
                let feature_offset = *features.add(g);
                let off = *dat_off.add(g);
                if n > 0
                    && k > 0
                    && co >= 0
                    && feature_offset >= 0
                    && off >= 0
                    && (k as u64) <= maxk
                    && (co as u64 + n as u64) <= celllen
                    && (feature_offset as u64 + k as u64) <= featlen
                    && (off as u64 + n as u64 * k as u64) <= datlen
                {
                    let (n, k, co, feature_offset, off) = (
                        n as usize,
                        k as usize,
                        co as usize,
                        feature_offset as usize,
                        off as usize,
                    );
                    let (mut ng, mut nn) = (0_f64, 0_f64);
                    let mut i = lane;
                    while i < n {
                        ng += if read_only_mask(guide.add(co + i)) != 0 {
                            1_f64
                        } else {
                            0_f64
                        };
                        nn += if read_only_mask(nt.add(co + i)) != 0 {
                            1_f64
                        } else {
                            0_f64
                        };
                        i += 32;
                    }
                    ng = sum_f64(ng);
                    nn = sum_f64(nn);
                    // A null global workspace selects the original shared
                    // direction vector. Warp kernels reserve one vector per
                    // warp; the block variant reserves one for the whole gene.
                    let v = if SHARED {
                        cuda_device::DynamicSharedArray::<f64>::get()
                            .add((thread::threadIdx_x() as usize / 32) * maxk as usize)
                    } else {
                        vec.add(row * maxk as usize)
                    };
                    let inv_ng = 1_f64 / ng.max(1_f64);
                    let mut j = lane;
                    while j < k {
                        let s = guide_sum_f64(dat.add(off), guide.add(co), n, k, j);
                        *v.add(j) = s * inv_ng - read_only_f64(ntmean.add(feature_offset + j));
                        j += 32;
                    }
                    warp::sync_mask(u32::MAX);
                    let mut vv = 0_f64;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv = x.mul_add(x, vv);
                        j += 32;
                    }
                    vv = sum_f64(vv).max(1e-12_f64);
                    let inv_vv = 1_f64 / vv;
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f64, 0_f64, 0_f64, 0_f64);
                    while i < n {
                        let s = projection_dot_f64(dat.add(off + i * k), v, k);
                        let y = s * inv_vv;
                        *p.add(co + i) = y;
                        if read_only_mask(guide.add(co + i)) != 0 {
                            sg += y;
                            sg2 = y.mul_add(y, sg2);
                        }
                        if read_only_mask(nt.add(co + i)) != 0 {
                            sn += y;
                            sn2 = y.mul_add(y, sn2);
                        }
                        i += 32;
                    }
                    sg = sum_f64(sg);
                    sg2 = sum_f64(sg2);
                    sn = sum_f64(sn);
                    sn2 = sum_f64(sn2);
                    let [m0, v0, m1, v1] =
                        projection_parameters_f64::<32>(sn, sn2, sg, sg2, ng, nn, lane);
                    warp::sync_mask(u32::MAX);
                    let _ = em_f64::<32>(
                        p.add(co),
                        resp.add(co),
                        n,
                        m0,
                        v0.max(1e-12_f64),
                        m1,
                        v1.max(1e-12_f64),
                        max_iter,
                        tol,
                        reg,
                        lane,
                    );
                }
            }
            row += stride;
        }
    }
}

// The small embedding regime caches a component's parameters once per block
// and gives each thread a cell. Fixed dimension variants keep centered values
// in registers and eliminate the loop bounds used by the general path.
macro_rules! logprob_small_impl {
    ($name:ident, $value:ty, $dimension:expr; $($column:expr),*) => {
        #[kernel]
        pub unsafe fn $name(
            x: *const $value,
            weights: *const $value,
            means: *const $value,
            precision: *const $value,
            logdet: *const $value,
            out: *mut $value,
            n: u64,
            d: u64,
            k: u64,
        ) {
            use cuda_device::SharedArray;
            const CAPACITY: usize = if $dimension == 0 { 64 } else { $dimension };
            static mut MEAN: SharedArray<$value, CAPACITY> = SharedArray::UNINIT;
            static mut CONSTANT: SharedArray<$value, 1> = SharedArray::UNINIT;
            static mut PRECISION: SharedArray<$value, { CAPACITY * (CAPACITY + 1) / 2 }> =
                SharedArray::UNINIT;
            unsafe {
                let dim = if $dimension == 0 {
                    d as usize
                } else {
                    $dimension
                };
                let cl = thread::blockIdx_y() as usize;
                let tid = thread::threadIdx_x() as usize;
                let row = thread::blockIdx_x() as usize * thread::blockDim_x() as usize + tid;
                if tid == 0 {
                    CONSTANT[0] = -0.5 as $value * dim as $value * 1.8378770664093453_f64 as $value
                        + *logdet.add(cl) + (*weights.add(cl)).ln();
                }
                let mean = SharedArray::as_raw_mut_ptr(&raw mut MEAN);
                let precision_cache = SharedArray::as_raw_mut_ptr(&raw mut PRECISION);
                let mut i = tid;
                while i < dim {
                    *mean.add(i) = *means.add(cl * dim + i);
                    i += thread::blockDim_x() as usize;
                }
                i = tid;
                while i < dim * dim {
                    let r = i / dim;
                    let c = i % dim;
                    if r <= c {
                        *precision_cache.add(c * (c + 1) / 2 + r) =
                            *precision.add(cl * dim * dim + i);
                    }
                    i += thread::blockDim_x() as usize;
                }
                thread::sync_threads();
                if row < n as usize {
                    let mut centered = [0 as $value; CAPACITY];
                    let mut dd = 0;
                    #[unroll]
                    while dd < CAPACITY {
                        if dd < dim {
                            centered[dd] = *x.add(row * dim + dd) - *mean.add(dd);
                        }
                        dd += 1;
                    }
                    let mut mahal = 0 as $value;
                    $(
                        if $column < CAPACITY && $column < dim {
                            let mut y = 0 as $value;
                            let mut dd = 0;
                            #[unroll]
                            while dd <= $column {
                                y = centered[dd].mul_add(
                                    *precision_cache.add($column * ($column + 1) / 2 + dd), y);
                                dd += 1;
                            }
                            mahal = y.mul_add(y, mahal);
                        }
                    )*
                    *out.add(row * k as usize + cl) = CONSTANT[0] - 0.5 as $value * mahal;
                }
            }
        }
    };
}
macro_rules! logprob_small {
    ($name:ident, $value:ty, $dimension:expr) => {
        logprob_small_impl!($name, $value, $dimension;
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
            32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
            48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63);
    };
}
logprob_small!(gmm_small_16_f32, f32, 16);
logprob_small!(gmm_small_32_f32, f32, 32);
logprob_small!(gmm_small_50_f32, f32, 50);
logprob_small!(gmm_small_64_f32, f32, 64);
logprob_small!(gmm_small_dynamic_f32, f32, 0);
logprob_small!(gmm_small_16_f64, f64, 16);
logprob_small!(gmm_small_32_f64, f64, 32);
logprob_small!(gmm_small_50_f64, f64, 50);
logprob_small!(gmm_small_64_f64, f64, 64);
logprob_small!(gmm_small_dynamic_f64, f64, 0);

// Tile the precision factor for larger embeddings while keeping each cell's
// 64 transformed features in registers. Shared memory stays bounded as d grows.
macro_rules! logprob_tiled {
    ($name:ident, $value:ty) => {
        #[kernel]
        pub unsafe fn $name(
            x: *const $value,
            weights: *const $value,
            means: *const $value,
            precision: *const $value,
            logdet: *const $value,
            out: *mut $value,
            n: u64,
            d: u64,
            k: u64,
        ) {
            use cuda_device::SharedArray;
            static mut MEAN: SharedArray<$value, 64> = SharedArray::UNINIT;
            static mut PRECISION: SharedArray<$value, 4096> = SharedArray::UNINIT;
            unsafe {
                let dim = d as usize;
                let cl = thread::blockIdx_y() as usize;
                let tid = thread::threadIdx_x() as usize;
                let block = thread::blockDim_x() as usize;
                let row = thread::blockIdx_x() as usize * block + tid;
                let mean = SharedArray::as_raw_mut_ptr(&raw mut MEAN);
                let tile = SharedArray::as_raw_mut_ptr(&raw mut PRECISION);
                let mut mahal = 0 as $value;
                let mut j_base = 0;
                while j_base < dim {
                    let cols = 64.min(dim - j_base);
                    let dd_limit = dim.min(j_base + 64);
                    let mut y = [0 as $value; 64];
                    let mut dd_base = 0;
                    while dd_base < dd_limit {
                        let features = 64.min(dd_limit - dd_base);
                        let mut i = tid;
                        while i < 64 {
                            *mean.add(i) = if i < features {
                                *means.add(cl * dim + dd_base + i)
                            } else {
                                0 as $value
                            };
                            i += block;
                        }
                        i = tid;
                        while i < 4096 {
                            let feat = i / 64;
                            let col = i % 64;
                            let dd = dd_base + feat;
                            let j = j_base + col;
                            *tile.add(i) = if feat < features && col < cols && dd <= j {
                                *precision.add((cl * dim + dd) * dim + j)
                            } else {
                                0 as $value
                            };
                            i += block;
                        }
                        thread::sync_threads();
                        if row < n as usize {
                            let mut feat = 0;
                            while feat < features {
                                let centered = *x.add(row * dim + dd_base + feat) - *mean.add(feat);
                                let mut col = 0;
                                #[unroll]
                                while col < 64 {
                                    y[col] = centered.mul_add(*tile.add(feat * 64 + col), y[col]);
                                    col += 1;
                                }
                                feat += 1;
                            }
                        }
                        thread::sync_threads();
                        dd_base += 64;
                    }
                    let mut col = 0;
                    #[unroll]
                    while col < 64 {
                        mahal = y[col].mul_add(y[col], mahal);
                        col += 1;
                    }
                    j_base += 64;
                }
                if row < n as usize {
                    *out.add(row * k as usize + cl) =
                        -0.5 as $value * dim as $value * 1.8378770664093453_f64 as $value
                            + *logdet.add(cl)
                            + (*weights.add(cl)).ln()
                            - 0.5 as $value * mahal;
                }
            }
        }
    };
}
logprob_tiled!(gmm_tiled_f32, f32);
logprob_tiled!(gmm_tiled_f64, f64);

#[kernel]
pub unsafe fn gmm_spherical_block_f32(
    p: *const f32,
    offsets: *const i32,
    m0: *const f32,
    v0: *const f32,
    m1: *const f32,
    v1: *const f32,
    resp: *mut f32,
    mo: *mut f32,
    vo: *mut f32,
    wo: *mut f32,
    genes: u64,
    len: u64,
    max_iter: u32,
    tol: f32,
    reg: f32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f32);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f32::<256>(
                    p.add(start as usize),
                    resp.add(start as usize),
                    (end - start) as usize,
                    *m0.add(row),
                    *v0.add(row),
                    init_m,
                    init_v.max(reg),
                    max_iter,
                    tol,
                    reg,
                    lane,
                );
            }
            if lane == 0 {
                *mo.add(row) = result.0;
                *vo.add(row) = result.1;
                *wo.add(row) = result.2;
            }
            row += stride;
        }
    }
}

#[kernel]
pub unsafe fn gmm_spherical_block_f64(
    p: *const f64,
    offsets: *const i32,
    m0: *const f64,
    v0: *const f64,
    m1: *const f64,
    v1: *const f64,
    resp: *mut f64,
    mo: *mut f64,
    vo: *mut f64,
    wo: *mut f64,
    genes: u64,
    len: u64,
    max_iter: u32,
    tol: f64,
    reg: f64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f64);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f64::<256>(
                    p.add(start as usize),
                    resp.add(start as usize),
                    (end - start) as usize,
                    *m0.add(row),
                    *v0.add(row),
                    init_m,
                    init_v.max(reg),
                    max_iter,
                    tol,
                    reg,
                    lane,
                );
            }
            if lane == 0 {
                *mo.add(row) = result.0;
                *vo.add(row) = result.1;
                *wo.add(row) = result.2;
            }
            row += stride;
        }
    }
}

#[inline(always)]
unsafe fn gmm_project_block_impl_f32<const SHARED: bool>(
    dat: *const f32,
    dat_off: *const i64,
    ns: *const i32,
    ks: *const i32,
    cells: *const i32,
    features: *const i32,
    ntmean: *const f32,
    guide: *const u8,
    nt: *const u8,
    active: *const i32,
    p: *mut f32,
    resp: *mut f32,
    vec: *mut f32,
    nactive: u64,
    genes: u64,
    maxk: u64,
    datlen: u64,
    celllen: u64,
    featlen: u64,
    max_iter: u32,
    tol: f32,
    reg: f32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        while row < nactive as usize {
            let gene = *active.add(row);
            if gene >= 0 && (gene as u64) < genes {
                let g = gene as usize;
                let n = *ns.add(g);
                let k = *ks.add(g);
                let co = *cells.add(g);
                let feature_offset = *features.add(g);
                let off = *dat_off.add(g);
                if n > 0
                    && k > 0
                    && co >= 0
                    && feature_offset >= 0
                    && off >= 0
                    && (k as u64) <= maxk
                    && (co as u64 + n as u64) <= celllen
                    && (feature_offset as u64 + k as u64) <= featlen
                    && (off as u64 + n as u64 * k as u64) <= datlen
                {
                    let (n, k, co, feature_offset, off) = (
                        n as usize,
                        k as usize,
                        co as usize,
                        feature_offset as usize,
                        off as usize,
                    );
                    let (mut ng, mut nn) = (0_f32, 0_f32);
                    let mut i = lane;
                    while i < n {
                        ng += if read_only_mask(guide.add(co + i)) != 0 {
                            1_f32
                        } else {
                            0_f32
                        };
                        nn += if read_only_mask(nt.add(co + i)) != 0 {
                            1_f32
                        } else {
                            0_f32
                        };
                        i += 256;
                    }
                    [ng, nn, _, _] = em_reduce4_f32([ng, nn, 0.0, 0.0]);
                    // A null global workspace selects the original shared
                    // direction vector. Warp kernels reserve one vector per
                    // warp; the block variant reserves one for the whole gene.
                    let v = if SHARED {
                        cuda_device::DynamicSharedArray::<f32>::get().add(0)
                    } else {
                        vec.add(row * maxk as usize)
                    };
                    let inv_ng = 1_f32 / ng.max(1_f32);
                    let mut j = lane;
                    while j < k {
                        let s = guide_sum_f32(dat.add(off), guide.add(co), n, k, j);
                        *v.add(j) = s * inv_ng - read_only_f32(ntmean.add(feature_offset + j));
                        j += 256;
                    }
                    thread::sync_threads();
                    let mut vv = 0_f32;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv = x.mul_add(x, vv);
                        j += 256;
                    }
                    vv = em_reduce4_f32([vv, 0.0, 0.0, 0.0])[0].max(1e-12_f32);
                    let inv_vv = 1_f32 / vv;
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f32, 0_f32, 0_f32, 0_f32);
                    while i < n {
                        let s = projection_dot_f32(dat.add(off + i * k), v, k);
                        let y = s * inv_vv;
                        *p.add(co + i) = y;
                        if read_only_mask(guide.add(co + i)) != 0 {
                            sg += y;
                            sg2 = y.mul_add(y, sg2);
                        }
                        if read_only_mask(nt.add(co + i)) != 0 {
                            sn += y;
                            sn2 = y.mul_add(y, sn2);
                        }
                        i += 256;
                    }
                    [sg, sg2, sn, sn2] = em_reduce4_f32([sg, sg2, sn, sn2]);
                    let [m0, v0, m1, v1] =
                        projection_parameters_f32::<256>(sn, sn2, sg, sg2, ng, nn, lane);
                    let _ = em_f32::<256>(
                        p.add(co),
                        resp.add(co),
                        n,
                        m0,
                        v0.max(1e-12_f32),
                        m1,
                        v1.max(1e-12_f32),
                        max_iter,
                        tol,
                        reg,
                        lane,
                    );
                }
            }
            row += stride;
        }
    }
}

#[inline(always)]
unsafe fn gmm_project_block_impl_f64<const SHARED: bool>(
    dat: *const f64,
    dat_off: *const i64,
    ns: *const i32,
    ks: *const i32,
    cells: *const i32,
    features: *const i32,
    ntmean: *const f64,
    guide: *const u8,
    nt: *const u8,
    active: *const i32,
    p: *mut f64,
    resp: *mut f64,
    vec: *mut f64,
    nactive: u64,
    genes: u64,
    maxk: u64,
    datlen: u64,
    celllen: u64,
    featlen: u64,
    max_iter: u32,
    tol: f64,
    reg: f64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize;
        let mut row = thread::blockIdx_x() as usize;
        let stride = thread::gridDim_x() as usize;
        while row < nactive as usize {
            let gene = *active.add(row);
            if gene >= 0 && (gene as u64) < genes {
                let g = gene as usize;
                let n = *ns.add(g);
                let k = *ks.add(g);
                let co = *cells.add(g);
                let feature_offset = *features.add(g);
                let off = *dat_off.add(g);
                if n > 0
                    && k > 0
                    && co >= 0
                    && feature_offset >= 0
                    && off >= 0
                    && (k as u64) <= maxk
                    && (co as u64 + n as u64) <= celllen
                    && (feature_offset as u64 + k as u64) <= featlen
                    && (off as u64 + n as u64 * k as u64) <= datlen
                {
                    let (n, k, co, feature_offset, off) = (
                        n as usize,
                        k as usize,
                        co as usize,
                        feature_offset as usize,
                        off as usize,
                    );
                    let (mut ng, mut nn) = (0_f64, 0_f64);
                    let mut i = lane;
                    while i < n {
                        ng += if read_only_mask(guide.add(co + i)) != 0 {
                            1_f64
                        } else {
                            0_f64
                        };
                        nn += if read_only_mask(nt.add(co + i)) != 0 {
                            1_f64
                        } else {
                            0_f64
                        };
                        i += 256;
                    }
                    [ng, nn, _, _] = em_reduce4_f64([ng, nn, 0.0, 0.0]);
                    // A null global workspace selects the original shared
                    // direction vector. Warp kernels reserve one vector per
                    // warp; the block variant reserves one for the whole gene.
                    let v = if SHARED {
                        cuda_device::DynamicSharedArray::<f64>::get().add(0)
                    } else {
                        vec.add(row * maxk as usize)
                    };
                    let inv_ng = 1_f64 / ng.max(1_f64);
                    let mut j = lane;
                    while j < k {
                        let s = guide_sum_f64(dat.add(off), guide.add(co), n, k, j);
                        *v.add(j) = s * inv_ng - read_only_f64(ntmean.add(feature_offset + j));
                        j += 256;
                    }
                    thread::sync_threads();
                    let mut vv = 0_f64;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv = x.mul_add(x, vv);
                        j += 256;
                    }
                    vv = em_reduce4_f64([vv, 0.0, 0.0, 0.0])[0].max(1e-12_f64);
                    let inv_vv = 1_f64 / vv;
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f64, 0_f64, 0_f64, 0_f64);
                    while i < n {
                        let s = projection_dot_f64(dat.add(off + i * k), v, k);
                        let y = s * inv_vv;
                        *p.add(co + i) = y;
                        if read_only_mask(guide.add(co + i)) != 0 {
                            sg += y;
                            sg2 = y.mul_add(y, sg2);
                        }
                        if read_only_mask(nt.add(co + i)) != 0 {
                            sn += y;
                            sn2 = y.mul_add(y, sn2);
                        }
                        i += 256;
                    }
                    [sg, sg2, sn, sn2] = em_reduce4_f64([sg, sg2, sn, sn2]);
                    let [m0, v0, m1, v1] =
                        projection_parameters_f64::<256>(sn, sn2, sg, sg2, ng, nn, lane);
                    let _ = em_f64::<256>(
                        p.add(co),
                        resp.add(co),
                        n,
                        m0,
                        v0.max(1e-12_f64),
                        m1,
                        v1.max(1e-12_f64),
                        max_iter,
                        tol,
                        reg,
                        lane,
                    );
                }
            }
            row += stride;
        }
    }
}

// Specializing at the entry point keeps projection-vector loads in their
// native shared/global address space and avoids a generic pointer in the dot loop.
macro_rules! project_entry {
    ($name:ident, $implementation:ident, $value:ty, $specialization:literal) => {
        #[kernel]
        pub unsafe fn $name(
            dat: *const $value,
            dat_off: *const i64,
            ns: *const i32,
            ks: *const i32,
            cells: *const i32,
            features: *const i32,
            ntmean: *const $value,
            guide: *const u8,
            nt: *const u8,
            active: *const i32,
            p: *mut $value,
            resp: *mut $value,
            vec: *mut $value,
            nactive: u64,
            genes: u64,
            maxk: u64,
            datlen: u64,
            celllen: u64,
            featlen: u64,
            max_iter: u32,
            tol: $value,
            reg: $value,
        ) {
            unsafe {
                $implementation::<$specialization>(
                    dat, dat_off, ns, ks, cells, features, ntmean, guide, nt, active, p, resp, vec,
                    nactive, genes, maxk, datlen, celllen, featlen, max_iter, tol, reg,
                )
            }
        }
    };
}
project_entry!(gmm_project_f32, gmm_project_impl_f32, f32, false);
project_entry!(gmm_project_shared_f32, gmm_project_impl_f32, f32, true);
project_entry!(gmm_project_f64, gmm_project_impl_f64, f64, false);
project_entry!(gmm_project_shared_f64, gmm_project_impl_f64, f64, true);
project_entry!(
    gmm_project_block_f32,
    gmm_project_block_impl_f32,
    f32,
    false
);
project_entry!(
    gmm_project_block_shared_f32,
    gmm_project_block_impl_f32,
    f32,
    true
);
project_entry!(
    gmm_project_block_f64,
    gmm_project_block_impl_f64,
    f64,
    false
);
project_entry!(
    gmm_project_block_shared_f64,
    gmm_project_block_impl_f64,
    f64,
    true
);
