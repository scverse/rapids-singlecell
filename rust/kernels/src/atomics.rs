//! Native CUDA floating-point atomics matching the original kernel semantics.

use cuda_device::ptx_asm;

/// CUDA's native f32 atomic add flushes subnormals to zero. The generic
/// cuda-oxide fetch_add currently emits a slower CAS loop with different
/// subnormal behavior, so use the native instruction explicitly.
///
/// # Safety
/// `pointer` is an aligned global-memory f32 location. Concurrent accesses to
/// that location must be atomic, and its allocation must outlive the kernel.
#[inline(always)]
pub(crate) unsafe fn add_f32(pointer: *mut f32, value: f32) {
    let previous: f32;
    unsafe {
        ptx_asm!(
            "atom.global.add.f32 %0, [%1], %2;",
            out("=f") previous,
            in("l") pointer as u64,
            in("f") value,
            clobber("memory"),
        );
    }
    let _ = previous;
}
