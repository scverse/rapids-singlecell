//! Native kde CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_kde_gaussian_kde_2d(
    xy: u64,
    out: u64,
    n: u64,
    a: u64,
    b: u64,
    c: u64,
    xy_len: u64,
    xy_kind: u64,
    xy_order: u64,
    xy_rows: u64,
    xy_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    let xy = Buffer {
        pointer: xy,
        len: xy_len,
        kind: xy_kind,
        order: xy_order,
        rows: xy_rows,
        cols: xy_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    // The pair loop can use direct typed loads after this one extent check.
    if n > xy.len / 2 || n > out.len {
        return;
    }
    let a = f64::from_bits(a);
    let b = f64::from_bits(b);
    let c = f64::from_bits(c);
    if xy.kind == 0 {
        let (a, b, c) = (a as f32, b as f32, c as f32);
        let mut i = tid();
        while i < n {
            let xi = unsafe { *(xy.pointer as *const f32).add((i * 2) as usize) };
            let yi = unsafe { *(xy.pointer as *const f32).add((i * 2 + 1) as usize) };
            let mut maximum = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            let mut j = 0;
            while j < n {
                let dx = xi - unsafe { *(xy.pointer as *const f32).add((j * 2) as usize) };
                let dy = yi - unsafe { *(xy.pointer as *const f32).add((j * 2 + 1) as usize) };
                let q = a * dx * dx + b * dx * dy + c * dy * dy;
                if q > maximum {
                    sum = sum * (maximum - q).exp() + 1.0;
                    maximum = q;
                } else {
                    sum += (q - maximum).exp();
                }
                j += 1;
            }
            out.put_single(i, sum.ln() + maximum);
            i += stride();
        }
    } else {
        let mut i = tid();
        while i < n {
            let xi = xy.f(i * 2);
            let yi = xy.f(i * 2 + 1);
            let mut maximum = f64::NEG_INFINITY;
            let mut sum = 0.0;
            let mut j = 0;
            while j < n {
                let dx = xy.round(xi - xy.f(j * 2));
                let dy = xy.round(yi - xy.f(j * 2 + 1));
                let q = xy.round(
                    xy.round(xy.round(a * dx) * dx)
                        + xy.round(xy.round(b * dx) * dy)
                        + xy.round(xy.round(c * dy) * dy),
                );
                if q > maximum {
                    sum = xy.round(xy.round(sum * xy.round((maximum - q).exp())) + 1.0);
                    maximum = q;
                } else {
                    sum = xy.round(sum + xy.round((q - maximum).exp()));
                }
                j += 1;
            }
            out.put(i, xy.round(sum.ln()) + maximum);
            i += stride();
        }
    }
}
