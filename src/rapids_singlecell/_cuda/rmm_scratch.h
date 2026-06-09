#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Shared RMM-backed device scratch, usable by any CUDA module that links
// rmm::rmm (see add_rmm_cuda_module in CMakeLists.txt). Allocations come from
// the current RMM device resource, so scratch participates in the same pool as
// CuPy/RAPIDS allocations.
void* rmm_allocate(size_t bytes);
void rmm_deallocate(void* ptr, size_t bytes);

// ---------------------------------------------------------------------------
// Small allocation pool for temporary CUDA buffers. Frees everything on scope
// exit; reuse a single pool across a kernel pipeline.
// ---------------------------------------------------------------------------
struct RmmScratchPool {
    struct Allocation {
        void* ptr = nullptr;
        size_t bytes = 0;
    };
    std::vector<Allocation> bufs;

    ~RmmScratchPool() {
        for (Allocation alloc : bufs) {
            if (!alloc.ptr) continue;
            rmm_deallocate(alloc.ptr, alloc.bytes);
        }
    }

    template <typename T>
    T* alloc(size_t count) {
        if (count == 0) count = 1;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("RMM scratch allocation size overflow");
        }
        size_t bytes = count * sizeof(T);
        void* ptr = rmm_allocate(bytes);
        bufs.push_back({ptr, bytes});
        return static_cast<T*>(ptr);
    }
};

// Single RAII RMM device buffer (frees on scope exit).
struct ScopedCudaBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    explicit ScopedCudaBuffer(size_t requested_bytes) {
        bytes = requested_bytes == 0 ? 1 : requested_bytes;
        ptr = rmm_allocate(bytes);
    }

    ~ScopedCudaBuffer() {
        if (!ptr) return;
        rmm_deallocate(ptr, bytes);
    }

    void* data() {
        return ptr;
    }

    ScopedCudaBuffer(const ScopedCudaBuffer&) = delete;
    ScopedCudaBuffer& operator=(const ScopedCudaBuffer&) = delete;
};
