//! Native guide_assignment CUDA kernels.
use super::*;
unsafe extern "C" {
    fn __nv_lgammaf(x: f32) -> f32;
}
#[inline(always)]
fn guide_posterior(y: f32, lam: f32, mu: f32, sigma: f32, pi0: f32) -> f32 {
    let lam = lam.max(1e-10);
    let sigma = sigma.max(1e-10);
    let pi0 = pi0.clamp(1e-10, 1.0 - 1e-10);
    let z = (y - mu) / sigma;
    let p0 = y * lam.ln() - lam - unsafe { __nv_lgammaf(y + 1.0) } + pi0.ln();
    let p1 = -0.5 * z * z - sigma.ln() - 0.918_938_5 + (1.0 - pi0).ln();
    1.0 / (1.0 + (p0 - p1).exp())
}
// All 256 threads in a guide block participate in these reductions.
#[device]
fn guide_total(v: f64) -> f64 {
    domain_block_total(v)
}
#[device]
fn guide_max(mut v: f64) -> f64 {
    static mut PARTIAL: cuda_device::SharedArray<f64, 8> = cuda_device::SharedArray::UNINIT;
    let lane = tid() % 256;
    let mut d = 16;
    while d > 0 {
        v = v.max(warp::shuffle_down_f64_sync(u32::MAX, v, d));
        d /= 2;
    }
    if lane.is_multiple_of(32) {
        unsafe {
            PARTIAL[(lane / 32) as usize] = v;
        }
    }
    thread::sync_threads();
    if lane == 0 {
        unsafe {
            let mut warp = 1;
            while warp < 8 {
                PARTIAL[0] = PARTIAL[0].max(PARTIAL[warp]);
                warp += 1;
            }
        }
    }
    thread::sync_threads();
    let result = unsafe { PARTIAL[0] };
    thread::sync_threads();
    result
}
#[device]
fn guide_broadcast(v: f32) -> f32 {
    static mut VALUE: cuda_device::SharedArray<f32, 1> = cuda_device::SharedArray::UNINIT;
    if tid().is_multiple_of(256) {
        unsafe {
            VALUE[0] = v;
        }
    }
    thread::sync_threads();
    let result = unsafe { VALUE[0] };
    thread::sync_threads();
    result
}

// Reduce all five EM sufficient statistics together. Keeping the original
// float32 arithmetic and sharing barriers avoids five serial double reductions.
#[inline(always)]
fn guide_em_total(mut values: [f32; 5]) -> [f32; 5] {
    static mut PARTIAL: cuda_device::SharedArray<f32, 40> = cuda_device::SharedArray::UNINIT;
    let lane = thread::threadIdx_x() as usize;
    let warp_id = lane / 32;
    let mut d = 16;
    while d > 0 {
        values[0] += warp::shuffle_down_f32_sync(u32::MAX, values[0], d);
        values[1] += warp::shuffle_down_f32_sync(u32::MAX, values[1], d);
        values[2] += warp::shuffle_down_f32_sync(u32::MAX, values[2], d);
        values[3] += warp::shuffle_down_f32_sync(u32::MAX, values[3], d);
        values[4] += warp::shuffle_down_f32_sync(u32::MAX, values[4], d);
        d /= 2;
    }
    if lane.is_multiple_of(32) {
        unsafe {
            PARTIAL[warp_id] = values[0];
            PARTIAL[8 + warp_id] = values[1];
            PARTIAL[16 + warp_id] = values[2];
            PARTIAL[24 + warp_id] = values[3];
            PARTIAL[32 + warp_id] = values[4];
        }
    }
    thread::sync_threads();
    if warp_id == 0 {
        unsafe {
            values[0] = if lane < 8 { PARTIAL[lane] } else { 0.0 };
            values[1] = if lane < 8 { PARTIAL[8 + lane] } else { 0.0 };
            values[2] = if lane < 8 { PARTIAL[16 + lane] } else { 0.0 };
            values[3] = if lane < 8 { PARTIAL[24 + lane] } else { 0.0 };
            values[4] = if lane < 8 { PARTIAL[32 + lane] } else { 0.0 };
        }
        d = 16;
        while d > 0 {
            values[0] += warp::shuffle_down_f32_sync(u32::MAX, values[0], d);
            values[1] += warp::shuffle_down_f32_sync(u32::MAX, values[1], d);
            values[2] += warp::shuffle_down_f32_sync(u32::MAX, values[2], d);
            values[3] += warp::shuffle_down_f32_sync(u32::MAX, values[3], d);
            values[4] += warp::shuffle_down_f32_sync(u32::MAX, values[4], d);
            d /= 2;
        }
        if lane == 0 {
            unsafe {
                PARTIAL[0] = values[0];
                PARTIAL[1] = values[1];
                PARTIAL[2] = values[2];
                PARTIAL[3] = values[3];
                PARTIAL[4] = values[4];
            }
        }
    }
    thread::sync_threads();
    let result = unsafe { [PARTIAL[0], PARTIAL[1], PARTIAL[2], PARTIAL[3], PARTIAL[4]] };
    thread::sync_threads();
    result
}

#[inline(always)]
fn guide_percentile(x: Buffer, guide: u64, cells: u64, count: u64) -> f32 {
    static mut HISTOGRAM: cuda_device::SharedArray<i32, 4097> = cuda_device::SharedArray::UNINIT;
    let lane = thread::threadIdx_x() as u64;
    let mut bin = lane;
    while bin <= 4096 {
        unsafe {
            HISTOGRAM[bin as usize] = 0;
        }
        bin += 256;
    }
    thread::sync_threads();
    let mut cell = lane;
    while cell < cells {
        let value = x.single_at(cell, guide);
        if value > 0.0 {
            let bin = value.ceil().clamp(1.0, 4096.0) as usize;
            unsafe {
                DeviceAtomicI32::from_ptr(core::ptr::addr_of_mut!(HISTOGRAM[bin]))
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        cell += 256;
    }
    thread::sync_threads();
    let mut location = 0.0;
    if lane == 0 {
        let target = (count as f32 * 0.75) as u64;
        let mut cumulative = 0;
        let mut bin = 1;
        while bin <= 4096 {
            cumulative += unsafe { HISTOGRAM[bin] } as u64;
            if cumulative > target {
                let q = (bin as f32).log2();
                location = if q > 0.5 { q } else { 3.0 };
                break;
            }
            bin += 1;
        }
    }
    guide_broadcast(location)
}

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_guide_assignment_assign_threshold_dense(
    X: u64,
    valid_guides: u64,
    lam: u64,
    mu: u64,
    sigma: u64,
    pi0: u64,
    assignments: u64,
    thresholds: u64,
    n_cells: u64,
    n_guides: u64,
    n_valid_guides: u64,
    posterior_threshold: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    valid_guides_len: u64,
    valid_guides_kind: u64,
    valid_guides_order: u64,
    valid_guides_rows: u64,
    valid_guides_cols: u64,
    lam_len: u64,
    lam_kind: u64,
    lam_order: u64,
    lam_rows: u64,
    lam_cols: u64,
    mu_len: u64,
    mu_kind: u64,
    mu_order: u64,
    mu_rows: u64,
    mu_cols: u64,
    sigma_len: u64,
    sigma_kind: u64,
    sigma_order: u64,
    sigma_rows: u64,
    sigma_cols: u64,
    pi0_len: u64,
    pi0_kind: u64,
    pi0_order: u64,
    pi0_rows: u64,
    pi0_cols: u64,
    assignments_len: u64,
    assignments_kind: u64,
    assignments_order: u64,
    assignments_rows: u64,
    assignments_cols: u64,
    thresholds_len: u64,
    thresholds_kind: u64,
    thresholds_order: u64,
    thresholds_rows: u64,
    thresholds_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let valid_guides = Buffer {
        pointer: valid_guides,
        len: valid_guides_len,
        kind: valid_guides_kind,
        order: valid_guides_order,
        rows: valid_guides_rows,
        cols: valid_guides_cols,
    };
    let lam = Buffer {
        pointer: lam,
        len: lam_len,
        kind: lam_kind,
        order: lam_order,
        rows: lam_rows,
        cols: lam_cols,
    };
    let mu = Buffer {
        pointer: mu,
        len: mu_len,
        kind: mu_kind,
        order: mu_order,
        rows: mu_rows,
        cols: mu_cols,
    };
    let sigma = Buffer {
        pointer: sigma,
        len: sigma_len,
        kind: sigma_kind,
        order: sigma_order,
        rows: sigma_rows,
        cols: sigma_cols,
    };
    let pi0 = Buffer {
        pointer: pi0,
        len: pi0_len,
        kind: pi0_kind,
        order: pi0_order,
        rows: pi0_rows,
        cols: pi0_cols,
    };
    let assignments = Buffer {
        pointer: assignments,
        len: assignments_len,
        kind: assignments_kind,
        order: assignments_order,
        rows: assignments_rows,
        cols: assignments_cols,
    };
    let thresholds = Buffer {
        pointer: thresholds,
        len: thresholds_len,
        kind: thresholds_kind,
        order: thresholds_order,
        rows: thresholds_rows,
        cols: thresholds_cols,
    };
    let posterior_threshold = f64::from_bits(posterior_threshold);
    let mut valid = tid() / 256;
    let lane = tid() % 256;
    while valid < n_valid_guides {
        let guide = valid_guides.i(valid);
        let mut cell = lane;
        let mut maximum = 0.0f64;
        while cell < n_cells {
            maximum = maximum.max(f64::from(X.single_at(cell, guide)).ceil());
            cell += 256;
        }
        maximum = guide_max(maximum);
        let mut threshold = f32::NAN;
        if lane == 0 {
            let mut raw = 1;
            while raw <= maximum as u64 {
                let post = guide_posterior(
                    (raw as f32).log2(),
                    lam.f(valid) as f32,
                    mu.f(valid) as f32,
                    sigma.f(valid) as f32,
                    pi0.f(valid) as f32,
                );
                if post > posterior_threshold as f32 {
                    threshold = raw as f32;
                    break;
                }
                raw += 1;
            }
            thresholds.put(valid, threshold as f64);
        }
        threshold = guide_broadcast(threshold);
        cell = lane;
        while cell < n_cells {
            assignments.put_i(
                valid * n_cells + cell,
                u64::from(
                    !threshold.is_nan() && f64::from(X.single_at(cell, guide)) >= threshold as f64,
                ),
            );
            cell += 256;
        }
        valid += stride() / 256;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_guide_assignment_fit_assign_dense(
    X: u64,
    assignments: u64,
    thresholds: u64,
    lam: u64,
    mu: u64,
    sigma: u64,
    pi0: u64,
    valid_mask: u64,
    nonzero_counts: u64,
    max_counts: u64,
    n_cells: u64,
    n_guides: u64,
    max_iter: u64,
    tol: u64,
    posterior_threshold: u64,
    X_len: u64,
    X_kind: u64,
    X_order: u64,
    X_rows: u64,
    X_cols: u64,
    assignments_len: u64,
    assignments_kind: u64,
    assignments_order: u64,
    assignments_rows: u64,
    assignments_cols: u64,
    thresholds_len: u64,
    thresholds_kind: u64,
    thresholds_order: u64,
    thresholds_rows: u64,
    thresholds_cols: u64,
    lam_len: u64,
    lam_kind: u64,
    lam_order: u64,
    lam_rows: u64,
    lam_cols: u64,
    mu_len: u64,
    mu_kind: u64,
    mu_order: u64,
    mu_rows: u64,
    mu_cols: u64,
    sigma_len: u64,
    sigma_kind: u64,
    sigma_order: u64,
    sigma_rows: u64,
    sigma_cols: u64,
    pi0_len: u64,
    pi0_kind: u64,
    pi0_order: u64,
    pi0_rows: u64,
    pi0_cols: u64,
    valid_mask_len: u64,
    valid_mask_kind: u64,
    valid_mask_order: u64,
    valid_mask_rows: u64,
    valid_mask_cols: u64,
    nonzero_counts_len: u64,
    nonzero_counts_kind: u64,
    nonzero_counts_order: u64,
    nonzero_counts_rows: u64,
    nonzero_counts_cols: u64,
    max_counts_len: u64,
    max_counts_kind: u64,
    max_counts_order: u64,
    max_counts_rows: u64,
    max_counts_cols: u64,
) {
    let X = Buffer {
        pointer: X,
        len: X_len,
        kind: X_kind,
        order: X_order,
        rows: X_rows,
        cols: X_cols,
    };
    let assignments = Buffer {
        pointer: assignments,
        len: assignments_len,
        kind: assignments_kind,
        order: assignments_order,
        rows: assignments_rows,
        cols: assignments_cols,
    };
    let thresholds = Buffer {
        pointer: thresholds,
        len: thresholds_len,
        kind: thresholds_kind,
        order: thresholds_order,
        rows: thresholds_rows,
        cols: thresholds_cols,
    };
    let lam = Buffer {
        pointer: lam,
        len: lam_len,
        kind: lam_kind,
        order: lam_order,
        rows: lam_rows,
        cols: lam_cols,
    };
    let mu = Buffer {
        pointer: mu,
        len: mu_len,
        kind: mu_kind,
        order: mu_order,
        rows: mu_rows,
        cols: mu_cols,
    };
    let sigma = Buffer {
        pointer: sigma,
        len: sigma_len,
        kind: sigma_kind,
        order: sigma_order,
        rows: sigma_rows,
        cols: sigma_cols,
    };
    let pi0 = Buffer {
        pointer: pi0,
        len: pi0_len,
        kind: pi0_kind,
        order: pi0_order,
        rows: pi0_rows,
        cols: pi0_cols,
    };
    let valid_mask = Buffer {
        pointer: valid_mask,
        len: valid_mask_len,
        kind: valid_mask_kind,
        order: valid_mask_order,
        rows: valid_mask_rows,
        cols: valid_mask_cols,
    };
    let nonzero_counts = Buffer {
        pointer: nonzero_counts,
        len: nonzero_counts_len,
        kind: nonzero_counts_kind,
        order: nonzero_counts_order,
        rows: nonzero_counts_rows,
        cols: nonzero_counts_cols,
    };
    let max_counts = Buffer {
        pointer: max_counts,
        len: max_counts_len,
        kind: max_counts_kind,
        order: max_counts_order,
        rows: max_counts_rows,
        cols: max_counts_cols,
    };
    let tol = f64::from_bits(tol);
    let posterior_threshold = f64::from_bits(posterior_threshold);
    let mut guide = tid() / 256;
    let lane = tid() % 256;
    while guide < n_guides {
        let mut cell = lane;
        let mut count = 0.0;
        let mut logsum = 0.0f32;
        let mut maximum = 0.0f64;
        while cell < n_cells {
            let x = X.single_at(cell, guide);
            if x > 0.0 {
                count += 1.0;
                logsum += x.log2();
                maximum = maximum.max(x.ceil() as f64);
            }
            cell += 256;
        }
        let nz = guide_total(count);
        let logsum = guide_total(logsum as f64) as f32;
        maximum = guide_max(maximum);
        let valid = nz >= 2.0 && maximum >= 2.0;
        let mut lambda = f32::NAN;
        let mut location = f32::NAN;
        let mut deviation = f32::NAN;
        let mut weight = f32::NAN;
        let mut threshold = f32::NAN;
        if valid {
            lambda = (logsum / nz as f32 * 0.5).clamp(0.01, 0.5);
            location = guide_percentile(X, guide, n_cells, nz as u64);
            deviation = 1.0;
            weight = 0.85;
            let mut iter = 0;
            while iter < max_iter {
                cell = lane;
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut y0 = 0.0f32;
                let mut y1 = 0.0f32;
                let mut y2 = 0.0f32;
                while cell < n_cells {
                    let x = X.single_at(cell, guide);
                    if x > 0.0 {
                        let y = x.log2();
                        let r1 = guide_posterior(y, lambda, location, deviation, weight);
                        let r0 = 1.0 - r1;
                        s0 += r0;
                        s1 += r1;
                        y0 += r0 * y;
                        y1 += r1 * y;
                        y2 += r1 * y * y;
                    }
                    cell += 256;
                }
                let [s0, s1, y0, y1, y2] = guide_em_total([s0, s1, y0, y1, y2]);
                let lmle = y0 / s0.max(1e-10);
                let nl = (s0 * lmle.max(1e-10).ln() / (s0 + 1.0)).exp();
                let ss = (deviation * deviation).max(1e-10);
                let nm = (y1 / ss + 0.75) / (s1 / ss + 0.25);
                let smle = ((y2 - 2.0 * nm * y1 + nm * nm * s1) / s1.max(1e-10))
                    .max(0.0)
                    .sqrt()
                    .max(0.01);
                let ns = ((s1 * smle.ln() + 2.0) / (s1 + 1.0)).exp().max(0.01);
                let nw = ((s0 - 0.1) / (nz as f32 - 1.0).max(1e-10)).clamp(0.01, 0.99);
                let change = (nl - lambda)
                    .abs()
                    .max((nm - location).abs())
                    .max((ns - deviation).abs())
                    .max((nw - weight).abs());
                lambda = nl;
                location = nm;
                deviation = ns;
                weight = nw;
                if change < tol as f32 {
                    break;
                }
                iter += 1;
            }
            if lane == 0 {
                let mut raw = 1;
                while raw <= maximum as u64 {
                    if guide_posterior((raw as f32).log2(), lambda, location, deviation, weight)
                        > posterior_threshold as f32
                    {
                        threshold = raw as f32;
                        break;
                    }
                    raw += 1;
                }
            }
        }
        threshold = guide_broadcast(threshold);
        if lane == 0 {
            lam.put(guide, lambda as f64);
            mu.put(guide, location as f64);
            sigma.put(guide, deviation as f64);
            pi0.put(guide, weight as f64);
            thresholds.put(guide, threshold as f64);
            valid_mask.put_i(guide, u64::from(valid));
            nonzero_counts.put_i(guide, nz as u64);
            max_counts.put_i(guide, maximum as u64);
        }
        cell = lane;
        while cell < n_cells {
            assignments.put_i(
                guide * n_cells + cell,
                u64::from(
                    !threshold.is_nan() && f64::from(X.single_at(cell, guide)) >= threshold as f64,
                ),
            );
            cell += 256;
        }
        guide += stride() / 256;
    }
}
