#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Shared device scratch backed by a Python-supplied (CuPy) allocator, injected
// at import via _set_scratch_allocator (see register_scratch_allocator). Keeps
// temporaries on the caller's current device resource (RMM pool / UVM aware)
// without linking librmm's per-release-versioned C++ ABI.
void* rmm_allocate(size_t bytes);
void rmm_deallocate(void* ptr, size_t bytes);

// Register the _set_scratch_allocator binding on a module (call from
// NB_MODULE).
namespace nanobind {
class module_;
}
void register_scratch_allocator(nanobind::module_& m);

// fraction * cudaMemGetInfo free; never trial-probe a pool.
// Probing ratchets RMM pools and can starve cudaStreamCreate.
size_t rmm_available_device_bytes(double fraction);

// Allocation pool for temporary CUDA buffers; frees everything on scope exit.
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
        size_t bytes = size_of<T>(count);
        void* ptr = rmm_allocate(bytes);
        bufs.push_back({ptr, bytes});
        return static_cast<T*>(ptr);
    }

    // Arrays freed together, from one allocation: alloc_all(a, n_a, b, n_b,
    // ...) points T* a, b, ... at n_a, n_b, ... values, each 256-byte aligned
    // as by cudaMalloc.
    template <typename... Arrays>
    void alloc_all(Arrays&&... arrays) {
        char* base = alloc<char>(carve(nullptr, 0, arrays...));
        carve(base, 0, arrays...);
    }

   private:
    template <typename T>
    static size_t size_of(size_t count) {
        if (count == 0) count = 1;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("RMM scratch allocation size overflow");
        }
        return count * sizeof(T);
    }
    static size_t carve(char*, size_t end) {
        return end;
    }
    template <typename T, typename... Rest>
    static size_t carve(char* base, size_t end, T*& array, size_t count,
                        Rest&... rest) {
        end = (end + 255) / 256 * 256;
        if (base) array = reinterpret_cast<T*>(base + end);
        return carve(base, end + size_of<T>(count), rest...);
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
