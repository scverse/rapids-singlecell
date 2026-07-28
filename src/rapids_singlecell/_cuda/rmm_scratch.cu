#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

#include "rmm_scratch.h"

namespace nb = nanobind;
using namespace nb::literals;

// Scratch is backed by a Python-supplied allocator (CuPy), injected at import
// via _set_scratch_allocator. This keeps temporaries on the same current device
// resource as the caller's CuPy arrays (RMM pool / UVM aware) without linking
// librmm's C++ ABI, which is versioned per RAPIDS release.
namespace {
std::function<std::uintptr_t(std::size_t)> g_alloc;
std::function<void(std::uintptr_t)> g_dealloc;

void set_scratch_allocator(nb::callable alloc, nb::callable dealloc) {
    g_alloc = [alloc](std::size_t bytes) -> std::uintptr_t {
        nb::gil_scoped_acquire gil;
        return nb::cast<std::uintptr_t>(alloc(bytes));
    };
    g_dealloc = [dealloc](std::uintptr_t ptr) {
        nb::gil_scoped_acquire gil;
        dealloc(ptr);
    };
}

void ensure_scratch_allocator() {
    nb::gil_scoped_acquire gil;
    if (g_alloc) return;
    nb::tuple callbacks =
        nb::cast<nb::tuple>(nb::module_::import_("rapids_singlecell._cuda")
                                .attr("_get_scratch_allocator")());
    set_scratch_allocator(nb::borrow<nb::callable>(callbacks[0]),
                          nb::borrow<nb::callable>(callbacks[1]));
}
}  // namespace

void register_scratch_allocator(nb::module_& m) {
    m.def(
        "_set_scratch_allocator",
        [](nb::callable alloc, nb::callable dealloc) {
            set_scratch_allocator(std::move(alloc), std::move(dealloc));
        },
        "alloc"_a, "dealloc"_a);
}

void* rmm_allocate(size_t bytes) {
    ensure_scratch_allocator();
    return reinterpret_cast<void*>(g_alloc(bytes));
}

void rmm_deallocate(void* ptr, size_t /*bytes*/) {
    if (g_dealloc) g_dealloc(reinterpret_cast<std::uintptr_t>(ptr));
}

// Plain cudaMemGetInfo budget query, never a pool-probing trial allocation.
// Probing ratchets RMM pools and can starve non-pool allocations like streams.
size_t rmm_available_device_bytes(double fraction) {
    if (fraction <= 0.0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (size_t)(free_b * fraction);
}
