//! Native aucell CUDA kernels.
use super::*;

/// Score a row/set subset while checking indirect gene metadata.
#[inline(always)]
unsafe fn score_set(
    ranks: *const i32,
    row: u64,
    columns: u64,
    genes: *const i32,
    start: i32,
    length: i32,
    genes_len: u64,
    cutoff: i32,
    offset: u64,
    step: u64,
) -> i64 {
    let mut score = 0_i64;
    if start >= 0 && length > 0 {
        let stop = (start as u64 + length as u64).min(genes_len);
        let mut position = start as u64 + offset;
        let row_data = unsafe { ranks.add((row * columns) as usize) };
        while position < stop {
            let gene = unsafe { *genes.add(position as usize) };
            if gene >= 0 && (gene as u64) < columns {
                let rank = unsafe { *row_data.add(gene as usize) };
                if rank <= cutoff {
                    score += cutoff as i64 - rank as i64;
                }
            }
            position += step;
        }
    }
    score
}

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_aucell_auc(
    ranks: u64,
    R: u64,
    C: u64,
    cnct: u64,
    starts: u64,
    lens: u64,
    n_sets: u64,
    n_up: u64,
    max_aucs: u64,
    es: u64,
    ranks_len: u64,
    ranks_kind: u64,
    ranks_order: u64,
    ranks_rows: u64,
    ranks_cols: u64,
    cnct_len: u64,
    cnct_kind: u64,
    cnct_order: u64,
    cnct_rows: u64,
    cnct_cols: u64,
    starts_len: u64,
    starts_kind: u64,
    starts_order: u64,
    starts_rows: u64,
    starts_cols: u64,
    lens_len: u64,
    lens_kind: u64,
    lens_order: u64,
    lens_rows: u64,
    lens_cols: u64,
    max_aucs_len: u64,
    max_aucs_kind: u64,
    max_aucs_order: u64,
    max_aucs_rows: u64,
    max_aucs_cols: u64,
    es_len: u64,
    es_kind: u64,
    es_order: u64,
    es_rows: u64,
    es_cols: u64,
) {
    let ranks = ranks as *const i32;
    let genes = cnct as *const i32;
    let starts = starts as *const i32;
    let lengths = lens as *const i32;
    let normalization = max_aucs as *const f32;
    let output = es as *mut f32;
    if thread::blockDim_x() == 256 {
        // Long gene sets share one score across a full warp. This coalesces
        // gene-index reads and gathers ranks within one expression row instead
        // of serially walking unrelated rows. Integer reduction is exact.
        let lane = (thread::threadIdx_x() & 31) as u64;
        let mut item = tid() / 32;
        while item < R * n_sets {
            let row = item / n_sets;
            let set = item % n_sets;
            let mut score = unsafe {
                score_set(
                    ranks,
                    row,
                    C,
                    genes,
                    *starts.add(set as usize),
                    *lengths.add(set as usize),
                    cnct_len,
                    n_up as i32,
                    lane,
                    32,
                )
            };
            // Every lane in an active warp reaches every shuffle, including
            // sets shorter than a warp and tail blocks with inactive warps.
            let mut delta = 16;
            while delta > 0 {
                score += warp::shuffle_down_u64_sync(u32::MAX, score as u64, delta) as i64;
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *output.add(item as usize) = score as f32 / *normalization.add(set as usize);
                }
            }
            item += stride() / 32;
        }
    } else {
        let mut item = tid();
        while item < R * n_sets {
            let row = item / n_sets;
            let set = item % n_sets;
            let score = unsafe {
                score_set(
                    ranks,
                    row,
                    C,
                    genes,
                    *starts.add(set as usize),
                    *lengths.add(set as usize),
                    cnct_len,
                    n_up as i32,
                    0,
                    1,
                )
            };
            unsafe {
                *output.add(item as usize) = score as f32 / *normalization.add(set as usize);
            }
            item += stride();
        }
    }
}
