//! Native cooc CUDA kernels.
use super::*;
#[inline(always)]
fn cooc_value(result: Buffer, i: u64, j: u64, t: u64, k: u64, l: u64, format: u64) -> f64 {
    if format == 0 {
        result.i((i * k + j) * l * 2 + t) as f64
            + result.i((i * k + j) * l * 2 + t + l) as f64
            + result.i((j * k + i) * l * 2 + t) as f64
            + result.i((j * k + i) * l * 2 + t + l) as f64
    } else {
        result.i((i * k + j) * l + t) as f64 + result.i((j * k + i) * l + t) as f64
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_cooc_count_csr_catpairs(
    spatial: u64,
    thresholds: u64,
    cat_offsets: u64,
    cell_indices: u64,
    pair_left: u64,
    pair_right: u64,
    counts: u64,
    num_pairs: u64,
    k: u64,
    l_val: u64,
    blocks_per_pair: u64,
    cell_tile: u64,
    block_size: u64,
    shared_mem: u64,
    spatial_len: u64,
    spatial_kind: u64,
    spatial_order: u64,
    spatial_rows: u64,
    spatial_cols: u64,
    thresholds_len: u64,
    thresholds_kind: u64,
    thresholds_order: u64,
    thresholds_rows: u64,
    thresholds_cols: u64,
    cat_offsets_len: u64,
    cat_offsets_kind: u64,
    cat_offsets_order: u64,
    cat_offsets_rows: u64,
    cat_offsets_cols: u64,
    cell_indices_len: u64,
    cell_indices_kind: u64,
    cell_indices_order: u64,
    cell_indices_rows: u64,
    cell_indices_cols: u64,
    pair_left_len: u64,
    pair_left_kind: u64,
    pair_left_order: u64,
    pair_left_rows: u64,
    pair_left_cols: u64,
    pair_right_len: u64,
    pair_right_kind: u64,
    pair_right_order: u64,
    pair_right_rows: u64,
    pair_right_cols: u64,
    counts_len: u64,
    counts_kind: u64,
    counts_order: u64,
    counts_rows: u64,
    counts_cols: u64,
) {
    let spatial = Buffer {
        pointer: spatial,
        len: spatial_len,
        kind: spatial_kind,
        order: spatial_order,
        rows: spatial_rows,
        cols: spatial_cols,
    };
    let thresholds = Buffer {
        pointer: thresholds,
        len: thresholds_len,
        kind: thresholds_kind,
        order: thresholds_order,
        rows: thresholds_rows,
        cols: thresholds_cols,
    };
    let cat_offsets = Buffer {
        pointer: cat_offsets,
        len: cat_offsets_len,
        kind: cat_offsets_kind,
        order: cat_offsets_order,
        rows: cat_offsets_rows,
        cols: cat_offsets_cols,
    };
    let cell_indices = Buffer {
        pointer: cell_indices,
        len: cell_indices_len,
        kind: cell_indices_kind,
        order: cell_indices_order,
        rows: cell_indices_rows,
        cols: cell_indices_cols,
    };
    let pair_left = Buffer {
        pointer: pair_left,
        len: pair_left_len,
        kind: pair_left_kind,
        order: pair_left_order,
        rows: pair_left_rows,
        cols: pair_left_cols,
    };
    let pair_right = Buffer {
        pointer: pair_right,
        len: pair_right_len,
        kind: pair_right_kind,
        order: pair_right_order,
        rows: pair_right_rows,
        cols: pair_right_cols,
    };
    let counts = Buffer {
        pointer: counts,
        len: counts_len,
        kind: counts_kind,
        order: counts_order,
        rows: counts_rows,
        cols: counts_cols,
    };
    // Private warp histograms avoid a global atomic for every threshold and
    // cell pair. Threshold windows keep shared memory bounded for large grids.
    static mut HIST: cuda_device::SharedArray<u64, 1024> = cuda_device::SharedArray::UNINIT;
    static mut COORDS: cuda_device::SharedArray<f32, 256> = cuda_device::SharedArray::UNINIT;
    let hist = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut HIST) };
    let mut tile = tid() / 128;
    let lane = tid() % 128;
    while tile < num_pairs * blocks_per_pair {
        let pair = tile / blocks_per_pair;
        let part = tile % blocks_per_pair;
        let l = pair_left.i(pair);
        let r = pair_right.i(pair);
        let mut ls = cat_offsets.i(l);
        let mut le = cat_offsets.i(l + 1);
        let mut rs = cat_offsets.i(r);
        let mut re = cat_offsets.i(r + 1);
        if le.saturating_sub(ls) < re.saturating_sub(rs) {
            core::mem::swap(&mut ls, &mut rs);
            core::mem::swap(&mut le, &mut re);
        }
        let na = le.saturating_sub(ls);
        let nb = re.saturating_sub(rs);
        let mut threshold_base = 0;
        while threshold_base < l_val {
            let width = (l_val - threshold_base).min(256);
            let mut q = lane;
            while q < 1024 {
                unsafe {
                    *hist.add(q as usize) = 0;
                }
                q += 128;
            }
            thread::sync_threads();
            let mut bbase = 0;
            while bbase < nb {
                let cells = (nb - bbase).min(128);
                q = lane;
                while q < 256 {
                    let local = q / 2;
                    let b = cell_indices.i(rs + bbase + local);
                    unsafe {
                        COORDS[q as usize] = if local < cells {
                            spatial.single(b * 2 + q % 2)
                        } else {
                            0.0
                        };
                    }
                    q += 128;
                }
                thread::sync_threads();
                let mut ai = part * 128 + lane;
                while ai < na {
                    let a = cell_indices.i(ls + ai);
                    let x = spatial.single(a * 2);
                    let y = spatial.single(a * 2 + 1);
                    let mut bi = 0;
                    while bi < cells {
                        if l != r || ai < bbase + bi {
                            let dx = x - unsafe { COORDS[(bi * 2) as usize] };
                            let dy = y - unsafe { COORDS[(bi * 2 + 1) as usize] };
                            let distance = dx * dx + dy * dy;
                            let mut lo = 0;
                            let mut hi = width;
                            while lo < hi {
                                let mid = (lo + hi) / 2;
                                if distance <= thresholds.single(threshold_base + mid) {
                                    hi = mid;
                                } else {
                                    lo = mid + 1;
                                }
                            }
                            if lo < width {
                                unsafe {
                                    cuda_device::atomic::BlockAtomicU64::from_ptr(
                                        hist.add(((lane / 32) * 256 + lo) as usize),
                                    )
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        bi += 1;
                    }
                    ai += blocks_per_pair * 128;
                }
                thread::sync_threads();
                bbase += 128;
            }
            q = lane;
            while q < width {
                unsafe {
                    *hist.add(q as usize) += *hist.add((q + 256) as usize)
                        + *hist.add((q + 512) as usize)
                        + *hist.add((q + 768) as usize);
                }
                q += 128;
            }
            thread::sync_threads();
            if lane == 0 {
                let mut total = 0;
                q = 0;
                while q < width {
                    unsafe {
                        total += *hist.add(q as usize);
                        *hist.add(q as usize) = total;
                    }
                    q += 1;
                }
            }
            thread::sync_threads();
            q = lane;
            while q < width {
                let output = (l * k + r) * l_val + threshold_base + q;
                if output < counts.len {
                    unsafe {
                        DeviceAtomicU64::from_ptr(
                            (counts.pointer as *mut u64).add(output as usize),
                        )
                        .fetch_add(*hist.add(q as usize), AtomicOrdering::Relaxed);
                    }
                }
                q += 128;
            }
            thread::sync_threads();
            threshold_base += 256;
        }
        tile += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_cooc_count_pairwise(
    spatial: u64,
    thresholds: u64,
    labels: u64,
    result: u64,
    n: u64,
    k: u64,
    l_val: u64,
    spatial_len: u64,
    spatial_kind: u64,
    spatial_order: u64,
    spatial_rows: u64,
    spatial_cols: u64,
    thresholds_len: u64,
    thresholds_kind: u64,
    thresholds_order: u64,
    thresholds_rows: u64,
    thresholds_cols: u64,
    labels_len: u64,
    labels_kind: u64,
    labels_order: u64,
    labels_rows: u64,
    labels_cols: u64,
    result_len: u64,
    result_kind: u64,
    result_order: u64,
    result_rows: u64,
    result_cols: u64,
) {
    let spatial = Buffer {
        pointer: spatial,
        len: spatial_len,
        kind: spatial_kind,
        order: spatial_order,
        rows: spatial_rows,
        cols: spatial_cols,
    };
    let thresholds = Buffer {
        pointer: thresholds,
        len: thresholds_len,
        kind: thresholds_kind,
        order: thresholds_order,
        rows: thresholds_rows,
        cols: thresholds_cols,
    };
    let labels = Buffer {
        pointer: labels,
        len: labels_len,
        kind: labels_kind,
        order: labels_order,
        rows: labels_rows,
        cols: labels_cols,
    };
    let result = Buffer {
        pointer: result,
        len: result_len,
        kind: result_kind,
        order: result_order,
        rows: result_rows,
        cols: result_cols,
    };
    let mut i = tid() / 128;
    let lane = tid() % 128;
    while i < n {
        let mut j = i + 1 + lane;
        while j < n {
            let dx = spatial.single(i * 2) - spatial.single(j * 2);
            let dy = spatial.single(i * 2 + 1) - spatial.single(j * 2 + 1);
            let ds = dx * dx + dy * dy;
            let a = labels.i(i);
            let b = labels.i(j);
            let mut low = a.min(b);
            let mut high = a.max(b);
            if i.is_multiple_of(2) {
                core::mem::swap(&mut low, &mut high);
            }
            let offset = if i % 4 < 2 { 0 } else { l_val };
            let mut r = 0;
            while r < l_val {
                if ds <= thresholds.single(r) {
                    result.add((low * k + high) * l_val * 2 + r + offset, 1.0);
                }
                r += 1;
            }
            j += 128;
        }
        i += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_cooc_reduce_shared(
    result: u64,
    out: u64,
    k: u64,
    l_val: u64,
    format: u64,
    workspace: u64,
    result_len: u64,
    result_kind: u64,
    result_order: u64,
    result_rows: u64,
    result_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
    workspace_len: u64,
    workspace_kind: u64,
    workspace_order: u64,
    workspace_rows: u64,
    workspace_cols: u64,
) {
    let result = Buffer {
        pointer: result,
        len: result_len,
        kind: result_kind,
        order: result_order,
        rows: result_rows,
        cols: result_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let workspace = Buffer {
        pointer: workspace,
        len: workspace_len,
        kind: workspace_kind,
        order: workspace_order,
        rows: workspace_rows,
        cols: workspace_cols,
    };
    let mut t = tid() / 128;
    let lane = tid() % 128;
    while t < l_val {
        let base = t * (k + 1);
        let mut i = lane;
        while i < k {
            let mut sum = 0.0;
            let mut j = 0;
            while j < k {
                sum += cooc_value(result, i, j, t, k, l_val, format);
                j += 1;
            }
            workspace.put(base + i, sum);
            i += 128;
        }
        cuda_device::thread::sync_threads();
        if lane == 0 {
            let mut total = 0.0;
            i = 0;
            while i < k {
                total += workspace.f(base + i);
                i += 1;
            }
            workspace.put(base + k, total);
        }
        cuda_device::thread::sync_threads();
        let total = workspace.f(base + k);
        let mut p = lane;
        while p < k * k {
            let i = p / k;
            let j = p % k;
            let row = workspace.f(base + i);
            let col = workspace.f(base + j);
            let v = cooc_value(result, i, j, t, k, l_val, format);
            out.put(
                p * l_val + t,
                if row > 0.0 && col > 0.0 {
                    v * total / (row * col)
                } else {
                    0.0
                },
            );
            p += 128;
        }
        cuda_device::thread::sync_threads();
        t += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_cooc_reduce_global(
    result: u64,
    inter_out: u64,
    out: u64,
    k: u64,
    l_val: u64,
    format: u64,
    workspace: u64,
    result_len: u64,
    result_kind: u64,
    result_order: u64,
    result_rows: u64,
    result_cols: u64,
    inter_out_len: u64,
    inter_out_kind: u64,
    inter_out_order: u64,
    inter_out_rows: u64,
    inter_out_cols: u64,
    out_len: u64,
    out_kind: u64,
    out_order: u64,
    out_rows: u64,
    out_cols: u64,
    workspace_len: u64,
    workspace_kind: u64,
    workspace_order: u64,
    workspace_rows: u64,
    workspace_cols: u64,
) {
    let result = Buffer {
        pointer: result,
        len: result_len,
        kind: result_kind,
        order: result_order,
        rows: result_rows,
        cols: result_cols,
    };
    let inter_out = Buffer {
        pointer: inter_out,
        len: inter_out_len,
        kind: inter_out_kind,
        order: inter_out_order,
        rows: inter_out_rows,
        cols: inter_out_cols,
    };
    let out = Buffer {
        pointer: out,
        len: out_len,
        kind: out_kind,
        order: out_order,
        rows: out_rows,
        cols: out_cols,
    };
    let workspace = Buffer {
        pointer: workspace,
        len: workspace_len,
        kind: workspace_kind,
        order: workspace_order,
        rows: workspace_rows,
        cols: workspace_cols,
    };
    let mut t = tid() / 128;
    let lane = tid() % 128;
    while t < l_val {
        let base = t * (k + 1);
        let mut i = lane;
        while i < k {
            let mut sum = 0.0;
            let mut j = 0;
            while j < k {
                sum += cooc_value(result, i, j, t, k, l_val, format);
                j += 1;
            }
            workspace.put(base + i, sum);
            i += 128;
        }
        cuda_device::thread::sync_threads();
        if lane == 0 {
            let mut total = 0.0;
            i = 0;
            while i < k {
                total += workspace.f(base + i);
                i += 1;
            }
            workspace.put(base + k, total);
        }
        cuda_device::thread::sync_threads();
        let total = workspace.f(base + k);
        let mut p = lane;
        while p < k * k {
            let i = p / k;
            let j = p % k;
            let row = workspace.f(base + i);
            let col = workspace.f(base + j);
            let v = cooc_value(result, i, j, t, k, l_val, format);
            out.put(
                p * l_val + t,
                if row > 0.0 && col > 0.0 {
                    v * total / (row * col)
                } else {
                    0.0
                },
            );
            p += 128;
        }
        cuda_device::thread::sync_threads();
        t += stride() / 128;
    }
}
