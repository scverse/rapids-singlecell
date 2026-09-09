//! Device kernels compiled to PTX by cuda-oxide, separately from the PyO3 host.

pub mod domain;
pub mod elementwise;
pub mod gmm;
pub mod harmony;
pub mod harmony_correction;
pub mod preprocessing;
pub mod rank_stream;
pub mod ranking;
mod ranking_sort;
pub mod sparse2dense;
mod sparse_ovo;
mod sparse_ovr;

use cuda_device::{kernel, thread};

/// Compute the existing exclude-first-column Jaccard weight for each KNN edge.
///
/// # Safety
/// `knn` and `out` must point to disjoint, four-byte aligned device allocations
/// with at least `n_obs * k` elements. Launch with a nonempty one-dimensional grid/block.
/// The host validates sizes, dtype, layout, device and allocation overlap.
#[kernel]
pub unsafe fn jaccard_shared_counts(knn: *const i32, n_obs: u64, k: u64, out: *mut f32) {
    let mut edge = thread::index_1d().get() as u64;
    let stride = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
    let n_edges = n_obs * k;
    while edge < n_edges {
        let i = edge / k;
        // SAFETY: edge is in the validated input allocation.
        let j = unsafe { *knn.add(edge as usize) };
        let mut weight = 0.0;
        if j >= 0 && (j as u64) < n_obs && j as u64 != i {
            // Preserve pair counting, including duplicate entries, instead of
            // converting the neighbor lists to mathematical sets.
            let mut count = 0_i64;
            let mut a = 1;
            while a < k {
                // SAFETY: i and j are valid rows and a/b are valid columns.
                let value = unsafe { *knn.add((i * k + a) as usize) };
                let mut b = 1;
                while b < k {
                    let neighbor = unsafe { *knn.add((j as u64 * k + b) as usize) };
                    count += (value == neighbor) as i64;
                    b += 1;
                }
                a += 1;
            }
            let denominator = 2 * (k as i64 - 1) - count;
            if denominator > 0 {
                weight = count as f32 / denominator as f32;
            }
        }
        // SAFETY: each thread visits distinct in-range edges, including when
        // the host caps the grid and threads process multiple edges.
        unsafe { *out.add(edge as usize) = weight };
        edge += stride;
    }
}
