//! Weighted regression and multi-covariate correction kernels.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
use super::harmony::{add_f32, add_f64, sum_f32, sum_f64};
use cuda_device::{SharedArray, kernel, ptx_asm, thread, warp};
#[kernel]
pub unsafe fn harmony_inverse_f32(
    o: *const f32,
    lambda: *const f32,
    inv: *mut f32,
    factor: *mut f32,
    p0: *mut f32,
    batches: u64,
    clusters: u64,
    cluster: i32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (batches, clusters) = (batches as usize, clusters as usize);
        let count = if cluster < 0 { clusters } else { 1 };
        while row < count {
            let k = if cluster < 0 { row } else { cluster as usize };
            let mut nk = 0_f32;
            let mut neg = 0_f32;
            let mut b = lane;
            while b < batches {
                let v = *o.add(b * clusters + k);
                let f = 1_f32 / (v + *lambda.add(b * clusters + k));
                *factor.add(row * batches + b) = f;
                *p0.add(row * batches + b) = -f * v;
                nk += v;
                neg += f * v * v;
                b += 32;
            }
            nk = sum_f32(nk);
            neg = sum_f32(neg);
            warp::sync_mask(u32::MAX);
            let ci = 1_f32 / (nk - neg);
            let nb = batches + 1;
            let mut j = lane;
            while j < nb * nb {
                let a = j / nb;
                let b = j % nb;
                let mut val = ci;
                if a > 0 {
                    val *= *p0.add(row * batches + a - 1);
                }
                if b > 0 {
                    val *= *p0.add(row * batches + b - 1);
                }
                if a == b && a > 0 {
                    val += *factor.add(row * batches + a - 1);
                }
                *inv.add(row * nb * nb + j) = val;
                j += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_inverse_f64(
    o: *const f64,
    lambda: *const f64,
    inv: *mut f64,
    factor: *mut f64,
    p0: *mut f64,
    batches: u64,
    clusters: u64,
    cluster: i32,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (batches, clusters) = (batches as usize, clusters as usize);
        let count = if cluster < 0 { clusters } else { 1 };
        while row < count {
            let k = if cluster < 0 { row } else { cluster as usize };
            let mut nk = 0_f64;
            let mut neg = 0_f64;
            let mut b = lane;
            while b < batches {
                let v = *o.add(b * clusters + k);
                let f = 1_f64 / (v + *lambda.add(b * clusters + k));
                *factor.add(row * batches + b) = f;
                *p0.add(row * batches + b) = -f * v;
                nk += v;
                neg += f * v * v;
                b += 32;
            }
            nk = sum_f64(nk);
            neg = sum_f64(neg);
            warp::sync_mask(u32::MAX);
            let ci = 1_f64 / (nk - neg);
            let nb = batches + 1;
            let mut j = lane;
            while j < nb * nb {
                let a = j / nb;
                let b = j % nb;
                let mut val = ci;
                if a > 0 {
                    val *= *p0.add(row * batches + a - 1);
                }
                if b > 0 {
                    val *= *p0.add(row * batches + b - 1);
                }
                if a == b && a > 0 {
                    val += *factor.add(row * batches + a - 1);
                }
                *inv.add(row * nb * nb + j) = val;
                j += 32;
            }
            row += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_column_f32(src: *const f32, dst: *mut f32, rows: u64, cols: u64, col: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < rows as usize {
            *dst.add(i) = *src.add(i * cols as usize + col as usize);
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_column_f64(src: *const f64, dst: *mut f64, rows: u64, cols: u64, col: u64) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        while i < rows as usize {
            *dst.add(i) = *src.add(i * cols as usize + col as usize);
            i += stride;
        }
    }
}
// Read the original paired-PC vector only when both coordinates are present
// and naturally aligned; odd feature counts retain the scalar tail.
#[inline(always)]
unsafe fn rhs_load2_f32(pointer: *const f32) -> [f32; 2] {
    let (a, b): (f32, f32);
    unsafe {
        ptx_asm!("ld.global.v2.f32 {%0, %1}, [%2];", out("=f") a, out("=f") b, in("l") pointer as u64, clobber("memory"));
    }
    [a, b]
}
#[inline(always)]
unsafe fn rhs_load2_f64(pointer: *const f64) -> [f64; 2] {
    let (a, b): (f64, f64);
    unsafe {
        ptx_asm!("ld.global.v2.f64 {%0, %1}, [%2];", out("=d") a, out("=d") b, in("l") pointer as u64, clobber("memory"));
    }
    [a, b]
}
macro_rules! weighted_rhs {
    ($name:ident, $value:ty, $load:ident, $sum:ident, $atomic:ident) => {
        #[kernel]
        #[cuda_device::launch_bounds(1024)]
        pub unsafe fn $name(
            x: *const $value,
            bias: *const $value,
            offsets: *const i32,
            indices: *const i32,
            out: *mut $value,
            rows: u64,
            pcs: u64,
            batches: u64,
        ) {
            unsafe {
                static mut PARTIAL: SharedArray<$value, 64> = SharedArray::UNINIT;
                let tid = thread::threadIdx_x() as usize;
                let block = thread::blockDim_x() as usize;
                let stride = thread::gridDim_x() as usize;
                let (n, d, b) = (rows as usize, pcs as usize, batches as usize);
                let pairs = d.div_ceil(2);
                let parts = if n < 300_000 { 8 } else { 0 };
                let mut task = thread::blockIdx_x() as usize;
                while task < (b + parts) * pairs {
                    let segment = task / pairs;
                    let pc = (task % pairs) * 2;
                    let second = pc + 1 < d;
                    let intercept = segment < parts;
                    let batch = if intercept { 0 } else { segment - parts + 1 };
                    let (begin, end) = if intercept {
                        let span = n.div_ceil(8);
                        (
                            (segment * span).min(n) as i32,
                            ((segment + 1) * span).min(n) as i32,
                        )
                    } else {
                        (*offsets.add(batch - 1), *offsets.add(batch))
                    };
                    let (mut a, mut z) = (0.0 as $value, 0.0 as $value);
                    if begin >= 0 && end >= begin && end as usize <= n {
                        let mut position = begin as usize + tid;
                        while position < end as usize {
                            let cell = if intercept {
                                position as i32
                            } else {
                                *indices.add(position)
                            };
                            if cell >= 0 && (cell as usize) < n {
                                let pointer = x.add(cell as usize * d + pc);
                                let weight = *bias.add(cell as usize);
                                let values = if second
                                    && (pointer as usize)
                                        .is_multiple_of(2 * core::mem::size_of::<$value>())
                                {
                                    $load(pointer)
                                } else {
                                    [*pointer, if second { *pointer.add(1) } else { 0.0 }]
                                };
                                a = values[0].mul_add(weight, a);
                                z = values[1].mul_add(weight, z);
                            }
                            position += block;
                        }
                    }
                    a = $sum(a);
                    z = $sum(z);
                    if tid.is_multiple_of(32) {
                        PARTIAL[tid / 32] = a;
                        PARTIAL[32 + tid / 32] = z;
                    }
                    thread::sync_threads();
                    if tid < 32 {
                        a = $sum(if tid < block / 32 { PARTIAL[tid] } else { 0.0 });
                        z = $sum(if tid < block / 32 {
                            PARTIAL[32 + tid]
                        } else {
                            0.0
                        });
                        if tid == 0 {
                            if intercept {
                                $atomic(out.add(pc), a);
                                if second {
                                    $atomic(out.add(pc + 1), z);
                                }
                            } else {
                                *out.add(batch * d + pc) = a;
                                if second {
                                    *out.add(batch * d + pc + 1) = z;
                                }
                            }
                        }
                    }
                    // All first-warp shared reads finish before the next task
                    // overwrites the same bounded pair reduction workspace.
                    thread::sync_threads();
                    task += stride;
                }
            }
        }
    };
}
weighted_rhs!(
    harmony_weighted_rhs_f32,
    f32,
    rhs_load2_f32,
    sum_f32,
    add_f32
);
weighted_rhs!(
    harmony_weighted_rhs_f64,
    f64,
    rhs_load2_f64,
    sum_f64,
    add_f64
);
#[kernel]
pub unsafe fn harmony_apply_f32(
    x: *const f32,
    r: *const f32,
    w: *const f32,
    cats: *const i32,
    z: *mut f32,
    rows: u64,
    pcs: u64,
    clusters: u64,
    batches: u64,
    covs: u64,
    initialize: u32,
    cluster: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (n, d, k, b, f) = (
            rows as usize,
            pcs as usize,
            clusters as usize,
            batches as usize,
            covs as usize,
        );
        while i < n * d {
            let cell = i / d;
            let pc = i % d;
            let mut correction = 0_f32;
            let (start, end) = if cluster < 0 {
                (0, k)
            } else {
                (cluster as usize, cluster as usize + 1)
            };
            for cl in start..end {
                let mut coef = 0_f32;
                for c in 0..f {
                    let cat = *cats.add(cell * f + c);
                    if cat >= 0 && (cat as usize) < b {
                        let wc = if cluster < 0 { cl } else { 0 };
                        coef += *w.add((wc * (b + 1) + cat as usize + 1) * d + pc);
                    }
                }
                correction += *r.add(cell * k + cl) * coef;
            }
            let base = if initialize != 0 {
                *x.add(i)
            } else {
                *z.add(i)
            };
            *z.add(i) = base - correction;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_apply_f64(
    x: *const f64,
    r: *const f64,
    w: *const f64,
    cats: *const i32,
    z: *mut f64,
    rows: u64,
    pcs: u64,
    clusters: u64,
    batches: u64,
    covs: u64,
    initialize: u32,
    cluster: i32,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (n, d, k, b, f) = (
            rows as usize,
            pcs as usize,
            clusters as usize,
            batches as usize,
            covs as usize,
        );
        while i < n * d {
            let cell = i / d;
            let pc = i % d;
            let mut correction = 0_f64;
            let (start, end) = if cluster < 0 {
                (0, k)
            } else {
                (cluster as usize, cluster as usize + 1)
            };
            for cl in start..end {
                let mut coef = 0_f64;
                for c in 0..f {
                    let cat = *cats.add(cell * f + c);
                    if cat >= 0 && (cat as usize) < b {
                        let wc = if cluster < 0 { cl } else { 0 };
                        coef += *w.add((wc * (b + 1) + cat as usize + 1) * d + pc);
                    }
                }
                correction += *r.add(cell * k + cl) * coef;
            }
            let base = if initialize != 0 {
                *x.add(i)
            } else {
                *z.add(i)
            };
            *z.add(i) = base - correction;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_gram_f32(
    o: *const f32,
    lambda: *const f32,
    active: *const u8,
    joint: *const f32,
    gram: *mut f32,
    batches: u64,
    clusters: u64,
    joints: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (b, k, j) = (batches as usize, clusters as usize, joints as usize);
        let nb = b + 1;
        while i < k * nb * nb {
            let cl = i / (nb * nb);
            let row = (i / nb) % nb;
            let col = i % nb;
            let mut v = 0_f32;
            if row == 0 && col == 0 {
                for jj in 0..j {
                    v += *joint.add(jj * k + cl);
                }
            } else if row == col {
                let bk = (row - 1) * k + cl;
                v = if *active.add(bk) != 0 {
                    *o.add(bk) + *lambda.add(bk)
                } else {
                    1_f32
                };
            } else if row == 0 || col == 0 {
                let batch = if row == 0 { col - 1 } else { row - 1 };
                let bk = batch * k + cl;
                if *active.add(bk) != 0 {
                    v = *o.add(bk);
                }
            }
            *gram.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_gram_f64(
    o: *const f64,
    lambda: *const f64,
    active: *const u8,
    joint: *const f64,
    gram: *mut f64,
    batches: u64,
    clusters: u64,
    joints: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (b, k, j) = (batches as usize, clusters as usize, joints as usize);
        let nb = b + 1;
        while i < k * nb * nb {
            let cl = i / (nb * nb);
            let row = (i / nb) % nb;
            let col = i % nb;
            let mut v = 0_f64;
            if row == 0 && col == 0 {
                for jj in 0..j {
                    v += *joint.add(jj * k + cl);
                }
            } else if row == col {
                let bk = (row - 1) * k + cl;
                v = if *active.add(bk) != 0 {
                    *o.add(bk) + *lambda.add(bk)
                } else {
                    1_f64
                };
            } else if row == 0 || col == 0 {
                let batch = if row == 0 { col - 1 } else { row - 1 };
                let bk = batch * k + cl;
                if *active.add(bk) != 0 {
                    v = *o.add(bk);
                }
            }
            *gram.add(i) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_cross_f32(
    joint: *const f32,
    cats: *const i32,
    active: *const u8,
    gram: *mut f32,
    batches: u64,
    clusters: u64,
    joints: u64,
    covs: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (b, k, j, f) = (
            batches as usize,
            clusters as usize,
            joints as usize,
            covs as usize,
        );
        while i < j * k {
            let jj = i / k;
            let cl = i % k;
            let val = *joint.add(i);
            for left in 0..f {
                let lb = *cats.add(jj * f + left);
                if lb >= 0 && (lb as usize) < b && *active.add(lb as usize * k + cl) != 0 {
                    for right in left + 1..f {
                        let rb = *cats.add(jj * f + right);
                        if rb >= 0 && (rb as usize) < b && *active.add(rb as usize * k + cl) != 0 {
                            let offset = cl * (b + 1) * (b + 1);
                            add_f32(
                                gram.add(offset + (lb as usize + 1) * (b + 1) + rb as usize + 1),
                                val,
                            );
                            add_f32(
                                gram.add(offset + (rb as usize + 1) * (b + 1) + lb as usize + 1),
                                val,
                            );
                        }
                    }
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_cross_f64(
    joint: *const f64,
    cats: *const i32,
    active: *const u8,
    gram: *mut f64,
    batches: u64,
    clusters: u64,
    joints: u64,
    covs: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (b, k, j, f) = (
            batches as usize,
            clusters as usize,
            joints as usize,
            covs as usize,
        );
        while i < j * k {
            let jj = i / k;
            let cl = i % k;
            let val = *joint.add(i);
            for left in 0..f {
                let lb = *cats.add(jj * f + left);
                if lb >= 0 && (lb as usize) < b && *active.add(lb as usize * k + cl) != 0 {
                    for right in left + 1..f {
                        let rb = *cats.add(jj * f + right);
                        if rb >= 0 && (rb as usize) < b && *active.add(rb as usize * k + cl) != 0 {
                            let offset = cl * (b + 1) * (b + 1);
                            add_f64(
                                gram.add(offset + (lb as usize + 1) * (b + 1) + rb as usize + 1),
                                val,
                            );
                            add_f64(
                                gram.add(offset + (rb as usize + 1) * (b + 1) + lb as usize + 1),
                                val,
                            );
                        }
                    }
                }
            }
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_joint_rhs_f32(
    x: *const f32,
    r: *const f32,
    offsets: *const i32,
    indices: *const i32,
    out: *mut f32,
    rows: u64,
    pcs: u64,
    clusters: u64,
    joints: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (n, d, k, j) = (
            rows as usize,
            pcs as usize,
            clusters as usize,
            joints as usize,
        );
        while row < j * k * d {
            let jj = row / (k * d);
            let cl = (row / d) % k;
            let pc = row % d;
            let begin = *offsets.add(jj);
            let end = *offsets.add(jj + 1);
            let mut v = 0_f32;
            if begin >= 0 && end >= begin && end <= n as i32 {
                let mut p = begin as usize + lane;
                while p < end as usize {
                    let cell = *indices.add(p);
                    if cell >= 0 && (cell as usize) < n {
                        v += *x.add(cell as usize * d + pc) * *r.add(cell as usize * k + cl);
                    }
                    p += 32;
                }
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
pub unsafe fn harmony_joint_rhs_f64(
    x: *const f64,
    r: *const f64,
    offsets: *const i32,
    indices: *const i32,
    out: *mut f64,
    rows: u64,
    pcs: u64,
    clusters: u64,
    joints: u64,
) {
    unsafe {
        let lane = thread::threadIdx_x() as usize % 32;
        let mut row = thread::index_1d().get() as usize / 32;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize / 32;
        let (n, d, k, j) = (
            rows as usize,
            pcs as usize,
            clusters as usize,
            joints as usize,
        );
        while row < j * k * d {
            let jj = row / (k * d);
            let cl = (row / d) % k;
            let pc = row % d;
            let begin = *offsets.add(jj);
            let end = *offsets.add(jj + 1);
            let mut v = 0_f64;
            if begin >= 0 && end >= begin && end <= n as i32 {
                let mut p = begin as usize + lane;
                while p < end as usize {
                    let cell = *indices.add(p);
                    if cell >= 0 && (cell as usize) < n {
                        v += *x.add(cell as usize * d + pc) * *r.add(cell as usize * k + cl);
                    }
                    p += 32;
                }
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
pub unsafe fn harmony_marginal_rhs_f32(
    joint: *const f32,
    offsets: *const i32,
    indices: *const i32,
    active: *const u8,
    out: *mut f32,
    pcs: u64,
    clusters: u64,
    batches: u64,
    joints: u64,
    nnz: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (d, k, b, j) = (
            pcs as usize,
            clusters as usize,
            batches as usize,
            joints as usize,
        );
        while i < k * b * d {
            let pc = i % d;
            let batch = (i / d) % b;
            let cl = i / (d * b);
            let mut v = 0_f32;
            let begin = *offsets.add(batch);
            let end = *offsets.add(batch + 1);
            if *active.add(batch * k + cl) != 0 && begin >= 0 && end >= begin && (end as u64) <= nnz
            {
                for p in begin..end {
                    let jj = *indices.add(p as usize);
                    if jj >= 0 && (jj as usize) < j {
                        v += *joint.add((jj as usize * k + cl) * d + pc);
                    }
                }
            }
            *out.add((cl * (b + 1) + batch + 1) * d + pc) = v;
            i += stride;
        }
    }
}
#[kernel]
pub unsafe fn harmony_marginal_rhs_f64(
    joint: *const f64,
    offsets: *const i32,
    indices: *const i32,
    active: *const u8,
    out: *mut f64,
    pcs: u64,
    clusters: u64,
    batches: u64,
    joints: u64,
    nnz: u64,
) {
    unsafe {
        let mut i = thread::index_1d().get() as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let (d, k, b, j) = (
            pcs as usize,
            clusters as usize,
            batches as usize,
            joints as usize,
        );
        while i < k * b * d {
            let pc = i % d;
            let batch = (i / d) % b;
            let cl = i / (d * b);
            let mut v = 0_f64;
            let begin = *offsets.add(batch);
            let end = *offsets.add(batch + 1);
            if *active.add(batch * k + cl) != 0 && begin >= 0 && end >= begin && (end as u64) <= nnz
            {
                for p in begin..end {
                    let jj = *indices.add(p as usize);
                    if jj >= 0 && (jj as usize) < j {
                        v += *joint.add((jj as usize * k + cl) * d + pc);
                    }
                }
            }
            *out.add((cl * (b + 1) + batch + 1) * d + pc) = v;
            i += stride;
        }
    }
}
