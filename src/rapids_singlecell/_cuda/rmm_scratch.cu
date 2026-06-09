#include <cstddef>
#include <stdexcept>
#include <string>

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
