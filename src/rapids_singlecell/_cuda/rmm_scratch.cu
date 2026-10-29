#include <cstddef>
#include <cstdint>
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
// Read and replaced under the GIL. Never destroyed: static destructors run
// after Py_Finalize, when releasing a Python object crashes.
nb::object& g_alloc = *new nb::object();
nb::object& g_dealloc = *new nb::object();

void set_scratch_allocator(nb::callable alloc, nb::callable dealloc) {
    g_alloc = std::move(alloc);
    g_dealloc = std::move(dealloc);
}
}  // namespace

void register_scratch_allocator(nb::module_& m) {
    m.def("_set_scratch_allocator", &set_scratch_allocator, "alloc"_a,
          "dealloc"_a);
}

void* rmm_allocate(size_t bytes) {
    nb::gil_scoped_acquire gil;
    if (!g_alloc) {
        nb::tuple callbacks =
            nb::cast<nb::tuple>(nb::module_::import_("rapids_singlecell._cuda")
                                    .attr("_get_scratch_allocator")());
        // Python can switch threads above: install once, so that every
        // pointer is freed by the allocator that made it.
        if (!g_alloc)
            set_scratch_allocator(nb::borrow<nb::callable>(callbacks[0]),
                                  nb::borrow<nb::callable>(callbacks[1]));
    }
    nb::object alloc = g_alloc;  // alive should Python replace it meanwhile
    return reinterpret_cast<void*>(nb::cast<std::uintptr_t>(alloc(bytes)));
}

// Frees run in destructors and must not throw: a dealloc's error (e.g. a
// signal handler's KeyboardInterrupt at its first bytecode) is reported as
// unraisable. The default dealloc runs no bytecode.
void rmm_deallocate(void* ptr, size_t /*bytes*/) {
    nb::gil_scoped_acquire gil;
    nb::object dealloc = g_dealloc;
    try {
        if (dealloc) dealloc(reinterpret_cast<std::uintptr_t>(ptr));
    } catch (nb::python_error& e) {
        e.discard_as_unraisable("rapids_singlecell scratch free");
    }
}

// Plain cudaMemGetInfo budget query, never a pool-probing trial allocation.
// Probing ratchets RMM pools and can starve non-pool allocations like streams.
size_t rmm_available_device_bytes(double fraction) {
    if (fraction <= 0.0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (size_t)(free_b * fraction);
}
