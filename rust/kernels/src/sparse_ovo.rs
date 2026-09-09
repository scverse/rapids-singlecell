//! Sparse reference comparisons with compact segments and implicit-zero ranks.
#![allow(clippy::too_many_arguments)]
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64, DeviceAtomicI32, DeviceAtomicU64};
use cuda_device::{kernel, thread};

#[inline(always)]
fn stride() -> u64 {
    thread::blockDim_x() as u64 * thread::gridDim_x() as u64
}
#[inline(always)]
unsafe fn index(p: *const u8, at: u64, wide: u32) -> i64 {
    unsafe {
        if wide != 0 {
            *p.cast::<i64>().add(at as usize)
        } else {
            *p.cast::<i32>().add(at as usize) as i64
        }
    }
}
#[inline(always)]
unsafe fn value(p: *const u8, at: u64, wide: u32) -> f64 {
    unsafe {
        if wide != 0 {
            *p.cast::<f64>().add(at as usize)
        } else {
            *p.cast::<f32>().add(at as usize) as f64
        }
    }
}
#[inline(always)]
fn ordered(v: f32) -> u32 {
    if v == 0.0 {
        0x8000_0000
    } else {
        let bits = v.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }
}
#[inline(always)]
unsafe fn bounds(values: *const u32, len: u64, key: u32) -> (u64, u64) {
    let mut lo = 0;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *values.add(mid as usize) } < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let first = lo;
    hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if unsafe { *values.add(mid as usize) } <= key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (first, lo)
}
#[inline(always)]
unsafe fn add(p: *mut f64, v: f64) {
    unsafe { DeviceAtomicF64::from_ptr(p) }.fetch_add(v, AtomicOrdering::Relaxed);
}

/// Map reference and grouped row IDs to compact segment labels.
/// # Safety
/// Arrays have nref/nall/groups+1/rows entries; offsets partition group rows.
/// Error flags has one initialized i32 entry. Invalid or duplicated row IDs
/// are recorded before the host permits any output mutations.
#[kernel]
pub unsafe fn sparse_ovo_memberships(
    refs: *const i64,
    grps: *const i64,
    offsets: *const u64,
    labels: *mut i32,
    errors: *mut i32,
    nref: u64,
    nall: u64,
    groups: u64,
    rows: u64,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < nref + nall {
        let (row, label) = if i < nref {
            (unsafe { *refs.add(i as usize) }, 0)
        } else {
            let j = i - nref;
            let mut lo = 0;
            let mut hi = groups;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if unsafe { *offsets.add((mid + 1) as usize) } <= j {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            (unsafe { *grps.add(j as usize) }, (lo + 1) as i32)
        };
        if row >= 0 && (row as u64) < rows {
            let previous = unsafe { DeviceAtomicI32::from_ptr(labels.add(row as usize)) }
                .swap(label, AtomicOrdering::Relaxed);
            if previous >= 0 {
                unsafe { DeviceAtomicI32::from_ptr(errors) }.fetch_or(2, AtomicOrdering::Relaxed);
            }
        } else {
            unsafe { DeviceAtomicI32::from_ptr(errors) }.fetch_or(1, AtomicOrdering::Relaxed);
        }
        i += stride();
    }
}

/// Count or pack only stored values belonging to a selected population.
/// # Safety
/// Valid CSC arrays span nnz entries and cols+1 pointers. Labels has rows
/// entries. Counts and offsets have (groups+1)*cols and one extra entry.
/// Mode 0 counts; mode 1 uses precomputed offsets and zeroed write cursors.
/// Optional statistics outputs have (groups+1)*out_cols entries.
#[kernel]
pub unsafe fn sparse_ovo_csc(
    data: *const u8,
    indices: *const u8,
    indptr: *const u8,
    labels: *const i32,
    counts: *mut u64,
    offsets: *const u64,
    packed: *mut u32,
    sums: *mut f64,
    nonzero: *mut f64,
    rows: u64,
    cols: u64,
    groups: u64,
    nnz: u64,
    out_cols: u64,
    output_begin: u64,
    data_wide: u32,
    index_wide: u32,
    pointer_wide: u32,
    mode: u32,
) {
    let mut col = thread::blockIdx_x() as u64;
    while col < cols {
        let first = unsafe { index(indptr, col, pointer_wide) };
        let last = unsafe { index(indptr, col + 1, pointer_wide) };
        if first >= 0 && last >= first && last as u64 <= nnz {
            let mut p = first as u64 + thread::threadIdx_x() as u64;
            while p < last as u64 {
                let row = unsafe { index(indices, p, index_wide) };
                if row >= 0 && (row as u64) < rows {
                    let group = unsafe { *labels.add(row as usize) };
                    if group >= 0 && (group as u64) <= groups {
                        let segment = group as u64 * cols + col;
                        let offset =
                            unsafe { DeviceAtomicU64::from_ptr(counts.add(segment as usize)) }
                                .fetch_add(1, AtomicOrdering::Relaxed);
                        if mode != 0 {
                            let first = unsafe { *offsets.add(segment as usize) };
                            let last = unsafe { *offsets.add(segment as usize + 1) };
                            if offset < last - first {
                                let v = unsafe { value(data, p, data_wide) };
                                unsafe {
                                    *packed.add((first + offset) as usize) = ordered(v as f32)
                                };
                                // Historical statistics rows contain all test
                                // groups first and the reference group last.
                                let stat_group = if group == 0 { groups } else { group as u64 - 1 };
                                let out = (stat_group * out_cols + output_begin + col) as usize;
                                unsafe {
                                    if !sums.is_null() {
                                        add(sums.add(out), v);
                                    }
                                    if !nonzero.is_null() && v != 0.0 {
                                        add(nonzero.add(out), 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
                p += thread::blockDim_x() as u64;
            }
        }
        col += thread::gridDim_x() as u64;
    }
}

/// Stage a bounded segment batch for sorting, or copy it back compactly.
/// # Safety
/// Offsets describes compact buffers. Padded has width*segments entries and
/// width is at least every selected segment length. Mode 0 packs, mode 1 copies
/// sorted padded values into the separate compact destination buffer.
#[kernel]
pub unsafe fn sparse_ovo_segment_copy(
    compact: *const u32,
    offsets: *const u64,
    padded: *mut u32,
    output: *mut u32,
    first: u64,
    segments: u64,
    width: u64,
    mode: u32,
) {
    let mut i = thread::index_1d().get() as u64;
    while i < segments * width {
        let segment = first + i / width;
        let at = i % width;
        let start = unsafe { *offsets.add(segment as usize) };
        let stop = unsafe { *offsets.add(segment as usize + 1) };
        unsafe {
            if mode == 0 {
                *padded.add(i as usize) = if at < stop - start {
                    *compact.add((start + at) as usize)
                } else {
                    u32::MAX
                };
            } else if at < stop - start {
                *output.add((start + at) as usize) = *padded.add(i as usize);
            }
        }
        i += stride();
    }
}

/// Rank each compact group against a compact reference, inserting zero masses.
/// # Safety
/// Sorted values and offsets define (groups+1)*cols segments. Sizes contains
/// reference/group populations and bounds every segment count. Rank and
/// optional tie outputs have groups*out_cols entries. Launch 256 threads/block.
#[kernel]
pub unsafe fn sparse_ovo_rank(
    sorted: *const u32,
    offsets: *const u64,
    sizes: *const u64,
    ranks: *mut f64,
    ties: *mut f64,
    groups: u64,
    cols: u64,
    out_cols: u64,
    output_begin: u64,
) {
    static mut PARTIAL: cuda_device::SharedArray<f64, 16> = cuda_device::SharedArray::UNINIT;
    let partial = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let warp = tid / 32;
    let mut pair = thread::blockIdx_x() as u64;
    while pair < groups * cols {
        let col = pair % cols;
        let group = pair / cols;
        let rstart = unsafe { *offsets.add(col as usize) };
        let rcount = unsafe { *offsets.add(col as usize + 1) } - rstart;
        let segment = (group + 1) * cols + col;
        let gstart = unsafe { *offsets.add(segment as usize) };
        let gcount = unsafe { *offsets.add(segment as usize + 1) } - gstart;
        let reference = unsafe { sorted.add(rstart as usize) };
        let values = unsafe { sorted.add(gstart as usize) };
        let nref = unsafe { *sizes };
        let ngrp = unsafe { *sizes.add(group as usize + 1) };
        let rimplicit = nref.saturating_sub(rcount);
        let gimplicit = ngrp.saturating_sub(gcount);
        let (rneg, rpos) = unsafe { bounds(reference, rcount, 0x8000_0000) };
        let (gneg, gpos) = unsafe { bounds(values, gcount, 0x8000_0000) };
        let rzero = rimplicit + rpos - rneg;
        let gzero = gimplicit + gpos - gneg;
        let mut rank = if tid == 0 {
            ngrp as f64 * (ngrp as f64 + 1.0) * 0.5
                + gimplicit as f64 * (rneg as f64 + rzero as f64 * 0.5)
        } else {
            0.0
        };
        let mut tie = if tid == 0 {
            let zero = (rzero + gzero) as f64;
            zero * zero * zero - zero
        } else {
            0.0
        };
        let mut at = tid as u64;
        while at < gcount + rcount {
            if at < gcount {
                let key = unsafe { *values.add(at as usize) };
                if at == 0 || unsafe { *values.add(at as usize - 1) } != key {
                    let (_, end) = unsafe { bounds(values, gcount, key) };
                    let (lo, hi) = unsafe { bounds(reference, rcount, key) };
                    let extra = if key > 0x8000_0000 {
                        rimplicit as f64
                    } else if key == 0x8000_0000 {
                        rimplicit as f64 * 0.5
                    } else {
                        0.0
                    };
                    rank += (end - at) as f64 * (lo as f64 + (hi - lo) as f64 * 0.5 + extra);
                    if !ties.is_null() && key != 0x8000_0000 {
                        let n = (end - at + hi - lo) as f64;
                        tie += n * n * n - n;
                    }
                }
            } else if !ties.is_null() {
                let at = at - gcount;
                let key = unsafe { *reference.add(at as usize) };
                if key != 0x8000_0000
                    && (at == 0 || unsafe { *reference.add(at as usize - 1) } != key)
                {
                    let (lo, hi) = unsafe { bounds(values, gcount, key) };
                    if lo == hi {
                        let (_, end) = unsafe { bounds(reference, rcount, key) };
                        let n = (end - at) as f64;
                        tie += n * n * n - n;
                    }
                }
            }
            at += 256;
        }
        rank = crate::harmony::sum_f64(rank);
        tie = crate::harmony::sum_f64(tie);
        if lane == 0 {
            unsafe {
                *partial.add(warp) = rank;
                *partial.add(warp + 8) = tie;
            }
        }
        thread::sync_threads();
        if warp == 0 {
            let rank = crate::harmony::sum_f64(if lane < 8 {
                unsafe { *partial.add(lane) }
            } else {
                0.0
            });
            let tie = crate::harmony::sum_f64(if lane < 8 {
                unsafe { *partial.add(lane + 8) }
            } else {
                0.0
            });
            if lane == 0 {
                let at = (group * out_cols + output_begin + col) as usize;
                unsafe {
                    *ranks.add(at) = rank;
                    if !ties.is_null() {
                        let n = (nref + ngrp) as f64;
                        *ties.add(at) = if n < 2.0 {
                            1.0
                        } else {
                            1.0 - tie / (n * n * n - n)
                        };
                    }
                }
            }
        }
        thread::sync_threads();
        pair += thread::gridDim_x() as u64;
    }
}
