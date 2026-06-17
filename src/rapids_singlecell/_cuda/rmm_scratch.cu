#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <rmm/mr/per_device_resource.hpp>

#include "rmm_scratch.h"

// Use the resource-ref API (`get_current_device_resource_ref()` +
// value-semantic `allocate_sync`/`deallocate_sync`) rather than the raw-pointer
// `get_current_device_resource()`. RMM 26.06 removes both the raw-pointer
// accessor and `<rmm/mr/device_memory_resource.hpp>` as it migrates to the cccl
// `cuda::mr` resource-concept model. The ref form compiles unchanged from
// RMM 25.12 through 26.06 (and onward), so it covers 26.04+.
void* rmm_allocate(size_t bytes) {
    try {
        return rmm::mr::get_current_device_resource_ref().allocate_sync(bytes);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            std::string("RMM scratch allocation failed (") +
            std::to_string(bytes) + " bytes): " + e.what());
    }
}

void rmm_deallocate(void* ptr, size_t bytes) {
    rmm::mr::get_current_device_resource_ref().deallocate_sync(ptr, bytes);
}

// `fraction` * the free device memory reported by cudaMemGetInfo.
//
// Deliberately a plain query, NOT a trial-allocation probe. Probing a pool's
// internal free by allocating until it grows permanently RATCHETS the pool
// (RMM pools never shrink): repeated wilcoxon calls would grow it toward the
// whole device and then starve non-pool allocations like cudaStreamCreate
// ("out of memory" on stream creation). cudaMemGetInfo free is correct and
// safe everywhere:
//   * Plain cuda: exact.
//   * Pool: the memory OUTSIDE the pool's reservation; the pool also serves
//     from its internal free, so this is conservative but never over-budgets
//     and never grows the pool. The host-streaming paths transfer each nonzero
//     once regardless of batch size (per-row cursor gather), so a smaller
//     budget only adds a few more passes -- it does not re-stream.
//   * Managed/UVM: device-resident free, so sizing to it avoids host spill.
size_t rmm_available_device_bytes(double fraction) {
    if (fraction <= 0.0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (size_t)(free_b * fraction);
}
