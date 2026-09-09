//! Native sinkhorn CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_sinkhorn_auto_eps(
    cost: u64,
    cost_off: u64,
    n: u64,
    m: u64,
    scale: u64,
    floor: u64,
    eps: u64,
    cost_len: u64,
    cost_kind: u64,
    cost_order: u64,
    cost_rows: u64,
    cost_cols: u64,
    cost_off_len: u64,
    cost_off_kind: u64,
    cost_off_order: u64,
    cost_off_rows: u64,
    cost_off_cols: u64,
    n_len: u64,
    n_kind: u64,
    n_order: u64,
    n_rows: u64,
    n_cols: u64,
    m_len: u64,
    m_kind: u64,
    m_order: u64,
    m_rows: u64,
    m_cols: u64,
    eps_len: u64,
    eps_kind: u64,
    eps_order: u64,
    eps_rows: u64,
    eps_cols: u64,
) {
    let cost = Buffer {
        pointer: cost,
        len: cost_len,
        kind: cost_kind,
        order: cost_order,
        rows: cost_rows,
        cols: cost_cols,
    };
    let cost_off = Buffer {
        pointer: cost_off,
        len: cost_off_len,
        kind: cost_off_kind,
        order: cost_off_order,
        rows: cost_off_rows,
        cols: cost_off_cols,
    };
    let n = Buffer {
        pointer: n,
        len: n_len,
        kind: n_kind,
        order: n_order,
        rows: n_rows,
        cols: n_cols,
    };
    let m = Buffer {
        pointer: m,
        len: m_len,
        kind: m_kind,
        order: m_order,
        rows: m_rows,
        cols: m_cols,
    };
    let scale = f64::from_bits(scale);
    let floor = f64::from_bits(floor);
    let eps = Buffer {
        pointer: eps,
        len: eps_len,
        kind: eps_kind,
        order: eps_order,
        rows: eps_rows,
        cols: eps_cols,
    };
    let mut b = tid() / 32;
    let lane = tid() % 32;
    while b < n.len {
        let size = n.i(b) * m.i(b);
        let offset = cost_off.i(b);
        let mut i = lane;
        let mut s = 0.0;
        let mut s2 = 0.0;
        while i < size {
            let v = cost.f(offset + i);
            s += v;
            s2 += v * v;
            i += 32;
        }
        s = sum_warp(s);
        s2 = sum_warp(s2);
        if lane == 0 {
            let den = size.max(1) as f64;
            let mu = s / den;
            eps.put(b, (scale * (s2 / den - mu * mu).max(0.0).sqrt()).max(floor));
        }
        b += stride() / 32;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_sinkhorn_update_g(
    cost: u64,
    cost_off: u64,
    n: u64,
    m: u64,
    f: u64,
    f_off: u64,
    g: u64,
    g_off: u64,
    col2pair: u64,
    eps: u64,
    log_b: u64,
    conv: u64,
    omega: u64,
    cost_len: u64,
    cost_kind: u64,
    cost_order: u64,
    cost_rows: u64,
    cost_cols: u64,
    cost_off_len: u64,
    cost_off_kind: u64,
    cost_off_order: u64,
    cost_off_rows: u64,
    cost_off_cols: u64,
    n_len: u64,
    n_kind: u64,
    n_order: u64,
    n_rows: u64,
    n_cols: u64,
    m_len: u64,
    m_kind: u64,
    m_order: u64,
    m_rows: u64,
    m_cols: u64,
    f_len: u64,
    f_kind: u64,
    f_order: u64,
    f_rows: u64,
    f_cols: u64,
    f_off_len: u64,
    f_off_kind: u64,
    f_off_order: u64,
    f_off_rows: u64,
    f_off_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
    g_off_len: u64,
    g_off_kind: u64,
    g_off_order: u64,
    g_off_rows: u64,
    g_off_cols: u64,
    col2pair_len: u64,
    col2pair_kind: u64,
    col2pair_order: u64,
    col2pair_rows: u64,
    col2pair_cols: u64,
    eps_len: u64,
    eps_kind: u64,
    eps_order: u64,
    eps_rows: u64,
    eps_cols: u64,
    log_b_len: u64,
    log_b_kind: u64,
    log_b_order: u64,
    log_b_rows: u64,
    log_b_cols: u64,
    conv_len: u64,
    conv_kind: u64,
    conv_order: u64,
    conv_rows: u64,
    conv_cols: u64,
) {
    let cost = Buffer {
        pointer: cost,
        len: cost_len,
        kind: cost_kind,
        order: cost_order,
        rows: cost_rows,
        cols: cost_cols,
    };
    let cost_off = Buffer {
        pointer: cost_off,
        len: cost_off_len,
        kind: cost_off_kind,
        order: cost_off_order,
        rows: cost_off_rows,
        cols: cost_off_cols,
    };
    let n = Buffer {
        pointer: n,
        len: n_len,
        kind: n_kind,
        order: n_order,
        rows: n_rows,
        cols: n_cols,
    };
    let m = Buffer {
        pointer: m,
        len: m_len,
        kind: m_kind,
        order: m_order,
        rows: m_rows,
        cols: m_cols,
    };
    let f = Buffer {
        pointer: f,
        len: f_len,
        kind: f_kind,
        order: f_order,
        rows: f_rows,
        cols: f_cols,
    };
    let f_off = Buffer {
        pointer: f_off,
        len: f_off_len,
        kind: f_off_kind,
        order: f_off_order,
        rows: f_off_rows,
        cols: f_off_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let g_off = Buffer {
        pointer: g_off,
        len: g_off_len,
        kind: g_off_kind,
        order: g_off_order,
        rows: g_off_rows,
        cols: g_off_cols,
    };
    let col2pair = Buffer {
        pointer: col2pair,
        len: col2pair_len,
        kind: col2pair_kind,
        order: col2pair_order,
        rows: col2pair_rows,
        cols: col2pair_cols,
    };
    let eps = Buffer {
        pointer: eps,
        len: eps_len,
        kind: eps_kind,
        order: eps_order,
        rows: eps_rows,
        cols: eps_cols,
    };
    let log_b = Buffer {
        pointer: log_b,
        len: log_b_len,
        kind: log_b_kind,
        order: log_b_order,
        rows: log_b_rows,
        cols: log_b_cols,
    };
    let conv = Buffer {
        pointer: conv,
        len: conv_len,
        kind: conv_kind,
        order: conv_order,
        rows: conv_rows,
        cols: conv_cols,
    };
    let omega = f64::from_bits(omega);
    if cost.kind == 0 {
        let omega = omega as f32;
        let mut c = tid();
        while c < col2pair.len {
            let b = col2pair.i(c);
            if conv.i(b) == 0 {
                let j = c - g_off.i(b);
                let nr = n.i(b);
                let nc = m.i(b);
                let offset = cost_off.i(b);
                let off = f_off.i(b);
                let e = eps.single(b);
                let inv_e = 1.0 / e;
                let mut i = 0;
                let mut maximum = f32::NEG_INFINITY;
                let mut sum = 0.0;
                while i < nr {
                    let v = (f.single(off + i) - cost.single(offset + i * nc + j)) * inv_e;
                    if v > maximum {
                        sum = sum * (maximum - v).exp() + 1.0;
                        maximum = v;
                    } else {
                        sum += (v - maximum).exp();
                    }
                    i += 1;
                }
                let new = e * (log_b.single(b) - (sum.ln() + maximum));
                g.put_single(
                    c,
                    if omega == 1.0 {
                        new
                    } else {
                        (1.0 - omega) * g.single(c) + omega * new
                    },
                );
            }
            c += stride();
        }
    } else {
        let mut c = tid();
        while c < col2pair.len {
            let b = col2pair.i(c);
            if conv.i(b) == 0 {
                let j = c - g_off.i(b);
                let nr = n.i(b);
                let nc = m.i(b);
                let offset = cost_off.i(b);
                let off = f_off.i(b);
                let e = eps.f(b);
                let mut i = 0;
                let mut maximum = f64::NEG_INFINITY;
                let mut sum = 0.0;
                while i < nr {
                    let v = cost.round((f.f(off + i) - cost.f(offset + i * nc + j)) / e);
                    if v > maximum {
                        sum = sum * (maximum - v).exp() + 1.0;
                        maximum = v;
                    } else {
                        sum += (v - maximum).exp();
                    }
                    i += 1;
                }
                let new = e * (log_b.f(b) - (sum.ln() + maximum));
                g.put(
                    c,
                    if omega == 1.0 {
                        new
                    } else {
                        (1.0 - omega) * g.f(c) + omega * new
                    },
                );
            }
            c += stride();
        }
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_sinkhorn_update_f(
    cost: u64,
    cost_off: u64,
    m: u64,
    g: u64,
    g_off: u64,
    f: u64,
    f_off: u64,
    row2pair: u64,
    eps: u64,
    log_a: u64,
    conv: u64,
    omega: u64,
    cost_len: u64,
    cost_kind: u64,
    cost_order: u64,
    cost_rows: u64,
    cost_cols: u64,
    cost_off_len: u64,
    cost_off_kind: u64,
    cost_off_order: u64,
    cost_off_rows: u64,
    cost_off_cols: u64,
    m_len: u64,
    m_kind: u64,
    m_order: u64,
    m_rows: u64,
    m_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
    g_off_len: u64,
    g_off_kind: u64,
    g_off_order: u64,
    g_off_rows: u64,
    g_off_cols: u64,
    f_len: u64,
    f_kind: u64,
    f_order: u64,
    f_rows: u64,
    f_cols: u64,
    f_off_len: u64,
    f_off_kind: u64,
    f_off_order: u64,
    f_off_rows: u64,
    f_off_cols: u64,
    row2pair_len: u64,
    row2pair_kind: u64,
    row2pair_order: u64,
    row2pair_rows: u64,
    row2pair_cols: u64,
    eps_len: u64,
    eps_kind: u64,
    eps_order: u64,
    eps_rows: u64,
    eps_cols: u64,
    log_a_len: u64,
    log_a_kind: u64,
    log_a_order: u64,
    log_a_rows: u64,
    log_a_cols: u64,
    conv_len: u64,
    conv_kind: u64,
    conv_order: u64,
    conv_rows: u64,
    conv_cols: u64,
) {
    let cost = Buffer {
        pointer: cost,
        len: cost_len,
        kind: cost_kind,
        order: cost_order,
        rows: cost_rows,
        cols: cost_cols,
    };
    let cost_off = Buffer {
        pointer: cost_off,
        len: cost_off_len,
        kind: cost_off_kind,
        order: cost_off_order,
        rows: cost_off_rows,
        cols: cost_off_cols,
    };
    let m = Buffer {
        pointer: m,
        len: m_len,
        kind: m_kind,
        order: m_order,
        rows: m_rows,
        cols: m_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let g_off = Buffer {
        pointer: g_off,
        len: g_off_len,
        kind: g_off_kind,
        order: g_off_order,
        rows: g_off_rows,
        cols: g_off_cols,
    };
    let f = Buffer {
        pointer: f,
        len: f_len,
        kind: f_kind,
        order: f_order,
        rows: f_rows,
        cols: f_cols,
    };
    let f_off = Buffer {
        pointer: f_off,
        len: f_off_len,
        kind: f_off_kind,
        order: f_off_order,
        rows: f_off_rows,
        cols: f_off_cols,
    };
    let row2pair = Buffer {
        pointer: row2pair,
        len: row2pair_len,
        kind: row2pair_kind,
        order: row2pair_order,
        rows: row2pair_rows,
        cols: row2pair_cols,
    };
    let eps = Buffer {
        pointer: eps,
        len: eps_len,
        kind: eps_kind,
        order: eps_order,
        rows: eps_rows,
        cols: eps_cols,
    };
    let log_a = Buffer {
        pointer: log_a,
        len: log_a_len,
        kind: log_a_kind,
        order: log_a_order,
        rows: log_a_rows,
        cols: log_a_cols,
    };
    let conv = Buffer {
        pointer: conv,
        len: conv_len,
        kind: conv_kind,
        order: conv_order,
        rows: conv_rows,
        cols: conv_cols,
    };
    let omega = f64::from_bits(omega);
    if cost.kind == 0 {
        let omega = omega as f32;
        let mut r = tid() / 32;
        let lane = tid() % 32;
        while r < row2pair.len {
            let b = row2pair.i(r);
            if conv.i(b) == 0 {
                let i = r - f_off.i(b);
                let nc = m.i(b);
                let offset = cost_off.i(b) + i * nc;
                let off = g_off.i(b);
                let e = eps.single(b);
                let inv_e = 1.0 / e;
                let mut j = lane;
                let mut maximum = f32::NEG_INFINITY;
                let mut sum = 0.0;
                while j < nc {
                    let v = (g.single(off + j) - cost.single(offset + j)) * inv_e;
                    if v > maximum {
                        sum = sum * (maximum - v).exp() + 1.0;
                        maximum = v;
                    } else {
                        sum += (v - maximum).exp();
                    }
                    j += 32;
                }
                let mut d = 16;
                while d > 0 {
                    let om = warp::shuffle_down_f32_sync(u32::MAX, maximum, d);
                    let os = warp::shuffle_down_f32_sync(u32::MAX, sum, d);
                    let mc = maximum.max(om);
                    sum = if sum > 0.0 {
                        sum * (maximum - mc).exp()
                    } else {
                        0.0
                    } + if os > 0.0 { os * (om - mc).exp() } else { 0.0 };
                    maximum = mc;
                    d /= 2;
                }
                if lane == 0 {
                    let new = e * (log_a.single(b) - (sum.ln() + maximum));
                    f.put_single(
                        r,
                        if omega == 1.0 {
                            new
                        } else {
                            (1.0 - omega) * f.single(r) + omega * new
                        },
                    );
                }
            }
            r += stride() / 32;
        }
    } else {
        let mut r = tid() / 32;
        let lane = tid() % 32;
        while r < row2pair.len {
            let b = row2pair.i(r);
            if conv.i(b) == 0 {
                let i = r - f_off.i(b);
                let nc = m.i(b);
                let offset = cost_off.i(b) + i * nc;
                let off = g_off.i(b);
                let e = eps.f(b);
                let mut j = lane;
                let mut maximum = f64::NEG_INFINITY;
                let mut sum = 0.0;
                while j < nc {
                    let v = cost.round((g.f(off + j) - cost.f(offset + j)) / e);
                    if v > maximum {
                        sum = sum * (maximum - v).exp() + 1.0;
                        maximum = v;
                    } else {
                        sum += (v - maximum).exp();
                    }
                    j += 32;
                }
                let mut d = 16;
                while d > 0 {
                    let om = warp::shuffle_down_f64_sync(u32::MAX, maximum, d);
                    let os = warp::shuffle_down_f64_sync(u32::MAX, sum, d);
                    let mc = maximum.max(om);
                    sum = if sum > 0.0 {
                        sum * (maximum - mc).exp()
                    } else {
                        0.0
                    } + if os > 0.0 { os * (om - mc).exp() } else { 0.0 };
                    maximum = mc;
                    d /= 2;
                }
                if lane == 0 {
                    let new = e * (log_a.f(b) - (sum.ln() + maximum));
                    f.put(
                        r,
                        if omega == 1.0 {
                            new
                        } else {
                            (1.0 - omega) * f.f(r) + omega * new
                        },
                    );
                }
            }
            r += stride() / 32;
        }
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_sinkhorn_check_convergence(
    f: u64,
    f_prev: u64,
    f_off: u64,
    n: u64,
    g: u64,
    g_prev: u64,
    g_off: u64,
    m: u64,
    tol: u64,
    conv: u64,
    f_len: u64,
    f_kind: u64,
    f_order: u64,
    f_rows: u64,
    f_cols: u64,
    f_prev_len: u64,
    f_prev_kind: u64,
    f_prev_order: u64,
    f_prev_rows: u64,
    f_prev_cols: u64,
    f_off_len: u64,
    f_off_kind: u64,
    f_off_order: u64,
    f_off_rows: u64,
    f_off_cols: u64,
    n_len: u64,
    n_kind: u64,
    n_order: u64,
    n_rows: u64,
    n_cols: u64,
    g_len: u64,
    g_kind: u64,
    g_order: u64,
    g_rows: u64,
    g_cols: u64,
    g_prev_len: u64,
    g_prev_kind: u64,
    g_prev_order: u64,
    g_prev_rows: u64,
    g_prev_cols: u64,
    g_off_len: u64,
    g_off_kind: u64,
    g_off_order: u64,
    g_off_rows: u64,
    g_off_cols: u64,
    m_len: u64,
    m_kind: u64,
    m_order: u64,
    m_rows: u64,
    m_cols: u64,
    conv_len: u64,
    conv_kind: u64,
    conv_order: u64,
    conv_rows: u64,
    conv_cols: u64,
) {
    let f = Buffer {
        pointer: f,
        len: f_len,
        kind: f_kind,
        order: f_order,
        rows: f_rows,
        cols: f_cols,
    };
    let f_prev = Buffer {
        pointer: f_prev,
        len: f_prev_len,
        kind: f_prev_kind,
        order: f_prev_order,
        rows: f_prev_rows,
        cols: f_prev_cols,
    };
    let f_off = Buffer {
        pointer: f_off,
        len: f_off_len,
        kind: f_off_kind,
        order: f_off_order,
        rows: f_off_rows,
        cols: f_off_cols,
    };
    let n = Buffer {
        pointer: n,
        len: n_len,
        kind: n_kind,
        order: n_order,
        rows: n_rows,
        cols: n_cols,
    };
    let g = Buffer {
        pointer: g,
        len: g_len,
        kind: g_kind,
        order: g_order,
        rows: g_rows,
        cols: g_cols,
    };
    let g_prev = Buffer {
        pointer: g_prev,
        len: g_prev_len,
        kind: g_prev_kind,
        order: g_prev_order,
        rows: g_prev_rows,
        cols: g_prev_cols,
    };
    let g_off = Buffer {
        pointer: g_off,
        len: g_off_len,
        kind: g_off_kind,
        order: g_off_order,
        rows: g_off_rows,
        cols: g_off_cols,
    };
    let m = Buffer {
        pointer: m,
        len: m_len,
        kind: m_kind,
        order: m_order,
        rows: m_rows,
        cols: m_cols,
    };
    let tol = f64::from_bits(tol);
    let conv = Buffer {
        pointer: conv,
        len: conv_len,
        kind: conv_kind,
        order: conv_order,
        rows: conv_rows,
        cols: conv_cols,
    };
    let mut b = tid() / 32;
    let lane = tid() % 32;
    while b < n.len {
        if conv.i(b) == 0 {
            let mut change = 0.0f64;
            let mut scale = 0.0f64;
            let mut i = lane;
            while i < n.i(b) {
                let p = f_off.i(b) + i;
                let v = f.f(p);
                change = change.max((v - f_prev.f(p)).abs());
                scale = scale.max(v.abs());
                i += 32;
            }
            let mut j = lane;
            while j < m.i(b) {
                let p = g_off.i(b) + j;
                let v = g.f(p);
                change = change.max((v - g_prev.f(p)).abs());
                scale = scale.max(v.abs());
                j += 32;
            }
            let mut d = 16;
            while d > 0 {
                change = change.max(warp::shuffle_down_f64_sync(u32::MAX, change, d));
                scale = scale.max(warp::shuffle_down_f64_sync(u32::MAX, scale, d));
                d /= 2;
            }
            if lane == 0 && change / (scale + 1.0) < tol {
                conv.put_i(b, 1);
            }
        }
        b += stride() / 32;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_sinkhorn_build_cost(
    emb: u64,
    cidx_l: u64,
    f_off: u64,
    cidx_r: u64,
    g_off: u64,
    n: u64,
    m: u64,
    cost_off: u64,
    tile_pair: u64,
    tile_i0: u64,
    tile_j0: u64,
    cost: u64,
    emb_len: u64,
    emb_kind: u64,
    emb_order: u64,
    emb_rows: u64,
    emb_cols: u64,
    cidx_l_len: u64,
    cidx_l_kind: u64,
    cidx_l_order: u64,
    cidx_l_rows: u64,
    cidx_l_cols: u64,
    f_off_len: u64,
    f_off_kind: u64,
    f_off_order: u64,
    f_off_rows: u64,
    f_off_cols: u64,
    cidx_r_len: u64,
    cidx_r_kind: u64,
    cidx_r_order: u64,
    cidx_r_rows: u64,
    cidx_r_cols: u64,
    g_off_len: u64,
    g_off_kind: u64,
    g_off_order: u64,
    g_off_rows: u64,
    g_off_cols: u64,
    n_len: u64,
    n_kind: u64,
    n_order: u64,
    n_rows: u64,
    n_cols: u64,
    m_len: u64,
    m_kind: u64,
    m_order: u64,
    m_rows: u64,
    m_cols: u64,
    cost_off_len: u64,
    cost_off_kind: u64,
    cost_off_order: u64,
    cost_off_rows: u64,
    cost_off_cols: u64,
    tile_pair_len: u64,
    tile_pair_kind: u64,
    tile_pair_order: u64,
    tile_pair_rows: u64,
    tile_pair_cols: u64,
    tile_i0_len: u64,
    tile_i0_kind: u64,
    tile_i0_order: u64,
    tile_i0_rows: u64,
    tile_i0_cols: u64,
    tile_j0_len: u64,
    tile_j0_kind: u64,
    tile_j0_order: u64,
    tile_j0_rows: u64,
    tile_j0_cols: u64,
    cost_len: u64,
    cost_kind: u64,
    cost_order: u64,
    cost_rows: u64,
    cost_cols: u64,
) {
    let emb = Buffer {
        pointer: emb,
        len: emb_len,
        kind: emb_kind,
        order: emb_order,
        rows: emb_rows,
        cols: emb_cols,
    };
    let cidx_l = Buffer {
        pointer: cidx_l,
        len: cidx_l_len,
        kind: cidx_l_kind,
        order: cidx_l_order,
        rows: cidx_l_rows,
        cols: cidx_l_cols,
    };
    let f_off = Buffer {
        pointer: f_off,
        len: f_off_len,
        kind: f_off_kind,
        order: f_off_order,
        rows: f_off_rows,
        cols: f_off_cols,
    };
    let cidx_r = Buffer {
        pointer: cidx_r,
        len: cidx_r_len,
        kind: cidx_r_kind,
        order: cidx_r_order,
        rows: cidx_r_rows,
        cols: cidx_r_cols,
    };
    let g_off = Buffer {
        pointer: g_off,
        len: g_off_len,
        kind: g_off_kind,
        order: g_off_order,
        rows: g_off_rows,
        cols: g_off_cols,
    };
    let n = Buffer {
        pointer: n,
        len: n_len,
        kind: n_kind,
        order: n_order,
        rows: n_rows,
        cols: n_cols,
    };
    let m = Buffer {
        pointer: m,
        len: m_len,
        kind: m_kind,
        order: m_order,
        rows: m_rows,
        cols: m_cols,
    };
    let cost_off = Buffer {
        pointer: cost_off,
        len: cost_off_len,
        kind: cost_off_kind,
        order: cost_off_order,
        rows: cost_off_rows,
        cols: cost_off_cols,
    };
    let tile_pair = Buffer {
        pointer: tile_pair,
        len: tile_pair_len,
        kind: tile_pair_kind,
        order: tile_pair_order,
        rows: tile_pair_rows,
        cols: tile_pair_cols,
    };
    let tile_i0 = Buffer {
        pointer: tile_i0,
        len: tile_i0_len,
        kind: tile_i0_kind,
        order: tile_i0_order,
        rows: tile_i0_rows,
        cols: tile_i0_cols,
    };
    let tile_j0 = Buffer {
        pointer: tile_j0,
        len: tile_j0_len,
        kind: tile_j0_kind,
        order: tile_j0_order,
        rows: tile_j0_rows,
        cols: tile_j0_cols,
    };
    let cost = Buffer {
        pointer: cost,
        len: cost_len,
        kind: cost_kind,
        order: cost_order,
        rows: cost_rows,
        cols: cost_cols,
    };
    let mut p = tid();
    while p < tile_pair.len * 256 {
        let tile = p / 256;
        let lane = p % 256;
        let b = tile_pair.i(tile);
        let i = tile_i0.i(tile) + lane / 16;
        let j = tile_j0.i(tile) + lane % 16;
        if i < n.i(b) && j < m.i(b) {
            let a = cidx_l.i(f_off.i(b) + i);
            let z = cidx_r.i(g_off.i(b) + j);
            if emb.kind == 0 {
                let mut d = 0;
                let mut square = 0.0f32;
                while d < emb.cols {
                    let diff = emb.single(a * emb.cols + d) - emb.single(z * emb.cols + d);
                    square += diff * diff;
                    d += 1;
                }
                cost.put_single(cost_off.i(b) + i * m.i(b) + j, square);
            } else {
                let mut d = 0;
                let mut s = 0.0;
                while d < emb.cols {
                    let diff = emb.round(emb.at(a, d) - emb.at(z, d));
                    s = emb.round(s + emb.round(diff * diff));
                    d += 1;
                }
                cost.put(cost_off.i(b) + i * m.i(b) + j, s);
            }
        }
        p += stride();
    }
}
