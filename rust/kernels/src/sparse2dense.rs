//! Sparse row/column scatter with additive handling of duplicate indices.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64};
use cuda_device::{kernel, ptx_asm, thread};

/// Add with the same native float atomic instruction as CUDA `atomicAdd`.
///
/// cuda-oxide's float32 `fetch_add` currently lowers to a CAS loop that
/// preserves subnormals. The native instruction preserves the existing
/// backend's flush-to-zero behavior and avoids that loop under contention.
///
/// # Safety
/// `out` must point to an aligned, valid global-memory f32 allocation, with
/// no concurrent non-atomic accesses to that location.
#[inline(always)]
unsafe fn atomic_add_f32(out: *mut f32, value: f32) {
    let _previous: f32;
    unsafe {
        ptx_asm!(
            "atom.global.add.f32 %0, [%1], %2;",
            out("=f") _previous,
            in("l") out as u64,
            in("f") value,
            clobber("memory"),
        );
    }
}

/// # Safety
/// `out` must point to an aligned, valid global-memory f64 allocation, with
/// no concurrent non-atomic accesses to that location.
#[inline(always)]
unsafe fn atomic_add_f64(out: *mut f64, value: f64) {
    unsafe { DeviceAtomicF64::from_ptr(out) }.fetch_add(value, AtomicOrdering::Relaxed);
}

// Concrete entry points keep the CUDA launch ABI independent of Rust generics.
macro_rules! sparse2dense_kernel {
    ($name:ident, $value:ty, $index:ty, $atomic_add:ident) => {
        /// Add compressed sparse values into an existing dense allocation.
        ///
        /// `c_order != 0` uses `row * minor + index`; zero uses
        /// `row + index * major`. The host chooses this flag according to the
        /// sparse format and the dense allocation's physical layout.
        /// Invalid row spans and minor indices are skipped.
        ///
        /// # Safety
        /// The naturally aligned device allocations must contain at least
        /// `major + 1` index values in `indptr`, `nnz` values in `index` and
        /// `data`, and `major * minor` values in `out`. These sizes and offsets
        /// must fit in `isize`. The output must not overlap an input, and
        /// concurrent accesses to the output must be atomic. Launch with a
        /// nonempty two-dimensional grid/block and unit Z dimensions.
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            indptr: *const $index,
            index: *const $index,
            data: *const $value,
            out: *mut $value,
            major: u64,
            minor: u64,
            nnz: u64,
            c_order: u32,
        ) {
            let stride_x = thread::blockDim_x() as u64 * thread::gridDim_x() as u64;
            let stride_y = thread::blockDim_y() as u64 * thread::gridDim_y() as u64;
            let mut row = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64
                + thread::threadIdx_x() as u64;
            let first_col = thread::blockIdx_y() as u64 * thread::blockDim_y() as u64
                + thread::threadIdx_y() as u64;

            while row < major {
                // SAFETY: the host validates the indptr allocation length.
                let start = unsafe { *indptr.add(row as usize) } as i64;
                let stop = unsafe { *indptr.add((row + 1) as usize) } as i64;
                // Validate on-device so malformed sparse metadata cannot read
                // past index/data without a host copy or stream synchronization.
                if start >= 0 && stop >= start && stop as u64 <= nnz {
                    let mut col = first_col;
                    let row_nnz = (stop - start) as u64;
                    while col < row_nnz {
                        let position = start as u64 + col;
                        // SAFETY: the checked row span is inside index/data.
                        let idx = unsafe { *index.add(position as usize) } as i64;
                        if idx >= 0 && (idx as u64) < minor {
                            let offset = if c_order != 0 {
                                row * minor + idx as u64
                            } else {
                                row + idx as u64 * major
                            };
                            let value = unsafe { *data.add(position as usize) };
                            // SAFETY: the host validates the dense allocation
                            // size/alignment and excludes overlap with inputs.
                            // Device scope includes duplicate indices handled
                            // by different blocks in the Y dimension.
                            unsafe { $atomic_add(out.add(offset as usize), value) };
                        }
                        col += stride_y;
                    }
                }
                row += stride_x;
            }
        }
    };
}

sparse2dense_kernel!(sparse2dense_f32_i32, f32, i32, atomic_add_f32);
sparse2dense_kernel!(sparse2dense_f32_i64, f32, i64, atomic_add_f32);
sparse2dense_kernel!(sparse2dense_f64_i32, f64, i32, atomic_add_f64);
sparse2dense_kernel!(sparse2dense_f64_i64, f64, i64, atomic_add_f64);
