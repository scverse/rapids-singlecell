//! Gaussian-mixture device kernels and per-gene fused EM.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
use super::harmony::{max_f32, max_f64, sum_f32, sum_f64};
use cuda_device::{kernel, thread, warp};

unsafe fn em_f32(
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
                i += 32;
            }
            n1 = sum_f32(n1);
            s1 = sum_f32(s1);
            s2 = sum_f32(s2);
            ll = sum_f32(ll);
            let mean = ll / n as f32;
            if (mean - prev).abs() < tol {
                break;
            }
            prev = mean;
            let inv = 1_f32 / n1.max(1e-12_f32);
            m1 = s1 * inv;
            v1 = (s2 * inv - m1 * m1 + reg).max(reg);
            w = n1 / n as f32;
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
            i += 32;
        }
        (m1, v1, w)
    }
}

unsafe fn em_f64(
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
                i += 32;
            }
            n1 = sum_f64(n1);
            s1 = sum_f64(s1);
            s2 = sum_f64(s2);
            ll = sum_f64(ll);
            let mean = ll / n as f64;
            if (mean - prev).abs() < tol {
                break;
            }
            prev = mean;
            let inv = 1_f64 / n1.max(1e-12_f64);
            m1 = s1 * inv;
            v1 = (s2 * inv - m1 * m1 + reg).max(reg);
            w = n1 / n as f64;
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
            i += 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
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
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f32);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f32(
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
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        while row < genes as usize {
            let start = *offsets.add(row);
            let end = *offsets.add(row + 1);
            let init_m = *m1.add(row);
            let init_v = *v1.add(row);
            let mut result = (init_m, init_v, 0.5_f64);
            if start >= 0 && end > start && (end as u64) <= len {
                result = em_f64(
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
pub unsafe fn gmm_project_f32(
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
        let mut row = thread::index_1d().get() as usize / 32;
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
                        ng += if *guide.add(co + i) != 0 {
                            1_f32
                        } else {
                            0_f32
                        };
                        nn += if *nt.add(co + i) != 0 { 1_f32 } else { 0_f32 };
                        i += 32;
                    }
                    ng = sum_f32(ng);
                    nn = sum_f32(nn);
                    let v = vec.add(row * maxk as usize);
                    let mut j = lane;
                    while j < k {
                        let mut s = 0_f32;
                        for i in 0..n {
                            if *guide.add(co + i) != 0 {
                                s += *dat.add(off + i * k + j);
                            }
                        }
                        *v.add(j) = s / ng.max(1_f32) - *ntmean.add(feature_offset + j);
                        j += 32;
                    }
                    warp::sync_mask(u32::MAX);
                    let mut vv = 0_f32;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv += x * x;
                        j += 32;
                    }
                    vv = sum_f32(vv).max(1e-12_f32);
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f32, 0_f32, 0_f32, 0_f32);
                    while i < n {
                        let mut s = 0_f32;
                        for j in 0..k {
                            s += *dat.add(off + i * k + j) * *v.add(j);
                        }
                        let y = s / vv;
                        *p.add(co + i) = y;
                        if *guide.add(co + i) != 0 {
                            sg += y;
                            sg2 += y * y;
                        }
                        if *nt.add(co + i) != 0 {
                            sn += y;
                            sn2 += y * y;
                        }
                        i += 32;
                    }
                    sg = sum_f32(sg);
                    sg2 = sum_f32(sg2);
                    sn = sum_f32(sn);
                    sn2 = sum_f32(sn2);
                    let m0 = sn / nn.max(1_f32);
                    let m1 = sg / ng.max(1_f32);
                    let v0 = if nn > 1_f32 {
                        (sn2 - sn * sn / nn) / (nn - 1_f32)
                    } else {
                        1e-12_f32
                    };
                    let v1 = if ng > 1_f32 {
                        (sg2 - sg * sg / ng) / (ng - 1_f32)
                    } else {
                        1e-12_f32
                    };
                    warp::sync_mask(u32::MAX);
                    let _ = em_f32(
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
#[kernel]
pub unsafe fn gmm_project_f64(
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
        let mut row = thread::index_1d().get() as usize / 32;
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
                        ng += if *guide.add(co + i) != 0 {
                            1_f64
                        } else {
                            0_f64
                        };
                        nn += if *nt.add(co + i) != 0 { 1_f64 } else { 0_f64 };
                        i += 32;
                    }
                    ng = sum_f64(ng);
                    nn = sum_f64(nn);
                    let v = vec.add(row * maxk as usize);
                    let mut j = lane;
                    while j < k {
                        let mut s = 0_f64;
                        for i in 0..n {
                            if *guide.add(co + i) != 0 {
                                s += *dat.add(off + i * k + j);
                            }
                        }
                        *v.add(j) = s / ng.max(1_f64) - *ntmean.add(feature_offset + j);
                        j += 32;
                    }
                    warp::sync_mask(u32::MAX);
                    let mut vv = 0_f64;
                    j = lane;
                    while j < k {
                        let x = *v.add(j);
                        vv += x * x;
                        j += 32;
                    }
                    vv = sum_f64(vv).max(1e-12_f64);
                    i = lane;
                    let (mut sg, mut sg2, mut sn, mut sn2) = (0_f64, 0_f64, 0_f64, 0_f64);
                    while i < n {
                        let mut s = 0_f64;
                        for j in 0..k {
                            s += *dat.add(off + i * k + j) * *v.add(j);
                        }
                        let y = s / vv;
                        *p.add(co + i) = y;
                        if *guide.add(co + i) != 0 {
                            sg += y;
                            sg2 += y * y;
                        }
                        if *nt.add(co + i) != 0 {
                            sn += y;
                            sn2 += y * y;
                        }
                        i += 32;
                    }
                    sg = sum_f64(sg);
                    sg2 = sum_f64(sg2);
                    sn = sum_f64(sn);
                    sn2 = sum_f64(sn2);
                    let m0 = sn / nn.max(1_f64);
                    let m1 = sg / ng.max(1_f64);
                    let v0 = if nn > 1_f64 {
                        (sn2 - sn * sn / nn) / (nn - 1_f64)
                    } else {
                        1e-12_f64
                    };
                    let v1 = if ng > 1_f64 {
                        (sg2 - sg * sg / ng) / (ng - 1_f64)
                    } else {
                        1e-12_f64
                    };
                    warp::sync_mask(u32::MAX);
                    let _ = em_f64(
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
