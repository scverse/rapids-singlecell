#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <rmm/mr/per_device_resource.hpp>

#include "rmm_scratch.h"

// Use the RMM resource-ref API; RMM 26.06 removed the raw-pointer accessor.
// The ref form compiles unchanged from RMM 25.12 through 26.06+.
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

// Plain cudaMemGetInfo budget query, never a pool-probing trial allocation.
// Probing ratchets RMM pools and can starve non-pool allocations like streams.
size_t rmm_available_device_bytes(double fraction) {
    if (fraction <= 0.0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (size_t)(free_b * fraction);
}
