//! Native bbknn CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_bbknn_find_top_k_per_row(
    data: u64,
    indptr: u64,
    n_rows: u64,
    trim: u64,
    vals: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    indptr_len: u64,
    indptr_kind: u64,
    indptr_order: u64,
    indptr_rows: u64,
    indptr_cols: u64,
    vals_len: u64,
    vals_kind: u64,
    vals_order: u64,
    vals_rows: u64,
    vals_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let indptr = Buffer {
        pointer: indptr,
        len: indptr_len,
        kind: indptr_kind,
        order: indptr_order,
        rows: indptr_rows,
        cols: indptr_cols,
    };
    let vals = Buffer {
        pointer: vals,
        len: vals_len,
        kind: vals_kind,
        order: vals_order,
        rows: vals_rows,
        cols: vals_cols,
    };
    let mut row = tid() / 32;
    let lane = tid() % 32;
    while row < n_rows {
        let start = indptr.i(row);
        let end = indptr.i(row + 1).min(data.len);
        let mut cut = 0f32;
        if end > start + trim && trim > 0 {
            let mut prefix = 0u32;
            let mut remaining = trim;
            let mut bit = 0x80000000u32;
            while bit != 0 {
                let mut count = 0u32;
                let mut p = start + lane;
                while p < end {
                    let v = data.f(p) as f32;
                    let bits = v.to_bits();
                    let key = if bits & 0x80000000 != 0 {
                        !bits
                    } else {
                        bits ^ 0x80000000
                    };
                    if (key & !(bit - 1)) == (prefix | bit) {
                        count += 1;
                    }
                    p += 32;
                }
                let mut d = 16;
                while d > 0 {
                    count += warp::shuffle_down_sync(u32::MAX, count, d);
                    d /= 2;
                }
                let count = warp::shuffle_sync(u32::MAX, count, 0) as u64;
                if count >= remaining {
                    prefix |= bit;
                } else {
                    remaining -= count;
                }
                bit >>= 1;
            }
            let bits = if prefix & 0x80000000 != 0 {
                prefix ^ 0x80000000
            } else {
                !prefix
            };
            cut = f32::from_bits(bits).max(0.0);
        }
        if lane == 0 {
            vals.put(row, cut as f64);
        }
        row += stride() / 32;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_bbknn_find_top_k_per_row_sorted(
    data: u64,
    indptr: u64,
    n_rows: u64,
    trim: u64,
    vals: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    indptr_len: u64,
    indptr_kind: u64,
    indptr_order: u64,
    indptr_rows: u64,
    indptr_cols: u64,
    vals_len: u64,
    vals_kind: u64,
    vals_order: u64,
    vals_rows: u64,
    vals_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let indptr = Buffer {
        pointer: indptr,
        len: indptr_len,
        kind: indptr_kind,
        order: indptr_order,
        rows: indptr_rows,
        cols: indptr_cols,
    };
    let vals = Buffer {
        pointer: vals,
        len: vals_len,
        kind: vals_kind,
        order: vals_order,
        rows: vals_rows,
        cols: vals_cols,
    };
    let mut row = tid() / 32;
    let lane = tid() % 32;
    while row < n_rows {
        let start = indptr.i(row);
        let end = indptr.i(row + 1).min(data.len);
        let mut cut = 0f32;
        if end > start + trim && trim > 0 {
            let mut prefix = 0u32;
            let mut remaining = trim;
            let mut bit = 0x80000000u32;
            while bit != 0 {
                let mut count = 0u32;
                let mut p = start + lane;
                while p < end {
                    let v = data.single(p);
                    let bits = v.to_bits();
                    let key = if bits & 0x7fffffff == 0 {
                        0x80000000
                    } else if bits & 0x80000000 != 0 {
                        !bits
                    } else {
                        bits ^ 0x80000000
                    };
                    if (key & !(bit - 1)) == (prefix | bit) {
                        count += 1;
                    }
                    p += 32;
                }
                // CUB sorts a full 2048-key tile padded with -Inf. Those
                // padding keys precede negative NaNs in descending order.
                if lane == 0 && (0x007fffff & !(bit - 1)) == (prefix | bit) {
                    count += 2048_u64.saturating_sub(end - start) as u32;
                }
                let mut d = 16;
                while d > 0 {
                    count += warp::shuffle_down_sync(u32::MAX, count, d);
                    d /= 2;
                }
                let count = warp::shuffle_sync(u32::MAX, count, 0) as u64;
                if count >= remaining {
                    prefix |= bit;
                } else {
                    remaining -= count;
                }
                bit >>= 1;
            }
            let bits = if prefix & 0x80000000 != 0 {
                prefix ^ 0x80000000
            } else {
                !prefix
            };
            cut = f32::from_bits(bits);
            if prefix == 0x80000000 {
                // CUB considers both zero signs equal and preserves input
                // order. Recover the remaining-th zero's original bit pattern.
                let mut base = start;
                while base < end {
                    let position = base + lane;
                    let bits = if position < end {
                        data.single(position).to_bits()
                    } else {
                        0
                    };
                    let mut mask =
                        warp::ballot_sync(u32::MAX, position < end && bits & 0x7fffffff == 0);
                    let count = mask.count_ones() as u64;
                    if remaining <= count {
                        let mut skip = remaining - 1;
                        while skip != 0 {
                            mask &= mask - 1;
                            skip -= 1;
                        }
                        cut = f32::from_bits(warp::shuffle_sync(
                            u32::MAX,
                            bits,
                            mask.trailing_zeros(),
                        ));
                        break;
                    }
                    remaining -= count;
                    base += 32;
                }
            }
        }
        if lane == 0 {
            vals.put_single(row, cut);
        }
        row += stride() / 32;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_bbknn_cut_smaller(
    indptr: u64,
    index: u64,
    data: u64,
    vals: u64,
    n_rows: u64,
    indptr_len: u64,
    indptr_kind: u64,
    indptr_order: u64,
    indptr_rows: u64,
    indptr_cols: u64,
    index_len: u64,
    index_kind: u64,
    index_order: u64,
    index_rows: u64,
    index_cols: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    vals_len: u64,
    vals_kind: u64,
    vals_order: u64,
    vals_rows: u64,
    vals_cols: u64,
) {
    let indptr = Buffer {
        pointer: indptr,
        len: indptr_len,
        kind: indptr_kind,
        order: indptr_order,
        rows: indptr_rows,
        cols: indptr_cols,
    };
    let index = Buffer {
        pointer: index,
        len: index_len,
        kind: index_kind,
        order: index_order,
        rows: index_rows,
        cols: index_cols,
    };
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let vals = Buffer {
        pointer: vals,
        len: vals_len,
        kind: vals_kind,
        order: vals_order,
        rows: vals_rows,
        cols: vals_cols,
    };
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < n_rows {
        let mut p = indptr.i(row) + lane;
        let end = indptr.i(row + 1).min(data.len).min(index.len);
        while p < end {
            let neighbor = index.i(p);
            let cut = vals.f(row).max(vals.f(neighbor));
            if data.f(p) < cut {
                data.put(p, 0.0);
            }
            p += 128;
        }
        row += stride() / 128;
    }
}
