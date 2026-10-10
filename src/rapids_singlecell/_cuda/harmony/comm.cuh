#pragma once

// Exact sums of int64 values across the GPUs of one process. One host thread
// per GPU runs the same sequence of allreduce calls; every GPU copies the
// others' values (by peer copy, or through pinned host memory when peer
// copies were not validated) and adds them. Integer sums are exact, so all
// GPUs hold the bits of a single-GPU run. The caller allocates all buffers.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace harmony_comm {

struct Comm {
    struct Rank {
        int device;
        long long *send[2], *recv[2];  // recv: n_ranks x capacity values
        cudaEvent_t sent[2], read[2];  // read: the slot is free again
        unsigned count = 0;
    };
    std::vector<Rank> ranks;
    size_t capacity;  // values per rank and slot
    bool peer;
    std::atomic<int> arrived{0};
    std::atomic<unsigned> generation{0};
    std::atomic<bool> aborted{false};  // a rank failed: release the others

    // buffers: per rank send[0], send[1] (device, or pinned host without
    // peer copies), recv[0], recv[1] (device).
    Comm(const std::vector<int>& devices, const std::vector<uintptr_t>& buffers,
         size_t capacity, bool peer)
        : ranks(devices.size()), capacity(capacity), peer(peer) {
        if (buffers.size() != 4 * devices.size())
            throw std::invalid_argument("harmony comm: 4 buffers per GPU");
        int previous;
        cudaGetDevice(&previous);
        for (size_t r = 0; r < devices.size(); ++r) {
            Rank& me = ranks[r];
            me.device = devices[r];
            check(cudaSetDevice(me.device));
            for (int s = 0; s < 2; ++s) {
                me.send[s] = reinterpret_cast<long long*>(buffers[4 * r + s]);
                me.recv[s] =
                    reinterpret_cast<long long*>(buffers[4 * r + 2 + s]);
                check(cudaEventCreateWithFlags(&me.sent[s],
                                               cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&me.read[s],
                                               cudaEventDisableTiming));
            }
            for (int other : devices)
                if (peer && other != me.device &&
                    cudaDeviceEnablePeerAccess(other, 0) != cudaSuccess)
                    cudaGetLastError();  // already enabled
        }
        cudaSetDevice(previous);
    }
    ~Comm() {
        int previous;
        cudaGetDevice(&previous);
        for (Rank& me : ranks) {
            cudaSetDevice(me.device);
            cudaDeviceSynchronize();
            for (int s = 0; s < 2; ++s) {
                cudaEventDestroy(me.sent[s]);
                cudaEventDestroy(me.read[s]);
            }
        }
        cudaSetDevice(previous);
    }
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("harmony comm: ") +
                                     cudaGetErrorString(status));
    }
    // Host barrier of the rank threads (spins: exchanges are microseconds).
    void barrier() {
        unsigned gen = generation.load();
        if (arrived.fetch_add(1) + 1 == (int)ranks.size()) {
            arrived.store(0);
            generation.fetch_add(1);
            return;
        }
        while (generation.load() == gen) {
            if (aborted.load())
                throw std::runtime_error("harmony comm: another GPU failed");
            std::this_thread::yield();
        }
    }
};

__global__ void sum_ranks_kernel(const long long* __restrict__ recv,
                                 long long* data, size_t n, size_t stride,
                                 int rank, int n_ranks) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        long long sum = 0;
        for (int r = 0; r < n_ranks; ++r)
            sum += r == rank ? data[i] : recv[r * stride + i];
        data[i] = sum;
    }
}

// One exchange of n <= capacity values (see allreduce).
inline void exchange(Comm* comm, int rank, long long* data, size_t n,
                     cudaStream_t stream) {
    auto& me = comm->ranks[rank];
    int slot = me.count++ & 1;
    // Every rank (this one too, possibly on another stream) must be done
    // with this slot's previous exchange before it is reused.
    for (auto& peer : comm->ranks) cudaStreamWaitEvent(stream, peer.read[slot]);
    size_t bytes = n * sizeof(long long);
    Comm::check(
        cudaMemcpyAsync(me.send[slot], data, bytes, cudaMemcpyDefault, stream));
    Comm::check(cudaEventRecord(me.sent[slot], stream));
    comm->barrier();
    for (int r = 0; r < (int)comm->ranks.size(); ++r) {
        if (r == rank) continue;
        auto& peer = comm->ranks[r];
        cudaStreamWaitEvent(stream, peer.sent[slot]);
        long long* dst = me.recv[slot] + r * comm->capacity;
        // Peer copies: stream-ordered pool memory needs the peer API.
        Comm::check(comm->peer
                        ? cudaMemcpyPeerAsync(dst, me.device, peer.send[slot],
                                              peer.device, bytes, stream)
                        : cudaMemcpyAsync(dst, peer.send[slot], bytes,
                                          cudaMemcpyHostToDevice, stream));
    }
    int blocks = (int)std::min<size_t>((n + 255) / 256, 1024);
    sum_ranks_kernel<<<blocks, 256, 0, stream>>>(
        me.recv[slot], data, n, comm->capacity, rank, (int)comm->ranks.size());
    Comm::check(cudaGetLastError());
    Comm::check(cudaEventRecord(me.read[slot], stream));
}

// data (n values on rank's GPU, stream-ordered) = its sum over the GPUs, in
// exchanges of at most `capacity` values. Every rank makes the same sequence
// of calls; without comm this is a no-op.
inline void allreduce(Comm* comm, int rank, long long* data, size_t n,
                      cudaStream_t stream) {
    if (!comm || comm->ranks.size() < 2) return;
    for (size_t i = 0; i < n; i += comm->capacity)
        exchange(comm, rank, data + i, std::min(comm->capacity, n - i), stream);
}

}  // namespace harmony_comm
