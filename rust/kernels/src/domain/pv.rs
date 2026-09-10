//! Native pv CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_pv_rev_cummin64(
    x: u64,
    out: u64,
    n_rows: u64,
    m: u64,
    x_len: u64,
    x_kind: u64,
    x_order: u64,
    x_rows: u64,
    x_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
) {
    // The typed host API checks both full extents once before this launch.
    let x = x as *const f64;
    let out = out as *mut f64;
    let mut row = tid();
    while row < n_rows {
        if m != 0 {
            let input = unsafe { x.add((row * m) as usize) };
            let output = unsafe { out.add((row * m) as usize) };
            // A trailing NaN propagates left; earlier NaNs fail the original
            // ordered comparison. Reading before writing supports in-place use.
            let mut column = m - 1;
            let mut minimum = unsafe { *input.add(column as usize) };
            unsafe {
                *output.add(column as usize) = minimum;
            }
            while column > 0 {
                column -= 1;
                let value = unsafe { *input.add(column as usize) };
                minimum = if value < minimum { value } else { minimum };
                unsafe {
                    *output.add(column as usize) = minimum;
                }
            }
        }
        row += stride();
    }
}
