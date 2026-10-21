#pragma once

// Exact sums of int64 values across the GPUs of one process. One host thread
// per GPU runs the same sequence of allreduce calls; values move by peer copy,
// or through pinned host memory when peer copies were not validated. When the
// others' values fit the receive slot, every GPU copies and adds them; larger
// sums go in P pieces: GPU r adds piece r of every GPU's values, then copies
// the others' summed pieces (2 (P-1)/P instead of P-1 times the values).
// Integer sums are exact in any order, so all GPUs hold the bits of a
// single-GPU run. The caller allocates all buffers.

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
        long long *send[2], *recv[2];  // recv: capacity + n_ranks values
        // reduced: this rank's summed piece is in send; read: slot free again
        cudaEvent_t sent[2], reduced[2], read[2];
        unsigned count = 0;
    };
    std::vector<Rank> ranks;
    size_t capacity;  // values per exchange (send slot)
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
                for (cudaEvent_t* e :
                     {&me.sent[s], &me.reduced[s], &me.read[s]})
                    check(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
            }
            for (int other : devices)
                if (peer && other != me.device &&
                    cudaDeviceEnablePeerAccess(other, 0) != cudaSuccess)
                    cudaGetLastError();  // already enabled
        }
        cudaSetDevice(previous);
    }
    ~Comm() {
        for (Rank& me : ranks)
            for (int s = 0; s < 2; ++s)
                for (cudaEvent_t e : {me.sent[s], me.reduced[s], me.read[s]})
                    cudaEventDestroy(e);
    }
    // After a failure, other ranks may still have exchanges in flight on the
    // buffers: wait for their GPUs before the buffers are released.
    void synchronize() {
        int previous;
        cudaGetDevice(&previous);
        for (Rank& me : ranks) {
            cudaSetDevice(me.device);
            cudaDeviceSynchronize();
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
    // n values at src, in rank `from`'s send slot, to dst on rank `to`'s GPU.
    void pull(int to, int from, long long* dst, const long long* src, size_t n,
              cudaStream_t stream) {
        if (n == 0) return;
        // Peer copies: stream-ordered pool memory needs the peer API.
        check(peer ? cudaMemcpyPeerAsync(dst, ranks[to].device, src,
                                         ranks[from].device, n * 8, stream)
                   : cudaMemcpyAsync(dst, src, n * 8, cudaMemcpyHostToDevice,
                                     stream));
    }
};

// data[i] += value i of each of `pieces` received pieces of n values.
__global__ void sum_pieces_kernel(const long long* __restrict__ recv,
                                  long long* data, size_t n, int pieces) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        long long sum = data[i];
        for (int p = 0; p < pieces; ++p) sum += recv[p * n + i];
        data[i] = sum;
    }
}

// One exchange of n <= capacity values (see allreduce).
inline void exchange(Comm* comm, int rank, long long* data, size_t n,
                     cudaStream_t stream) {
    auto& me = comm->ranks[rank];
    int slot = me.count++ & 1, n_ranks = (int)comm->ranks.size();
    // Every rank (this one too, possibly on another stream) must be done
    // with this slot's previous exchange before it is reused.
    for (auto& peer : comm->ranks) cudaStreamWaitEvent(stream, peer.read[slot]);
    Comm::check(cudaMemcpyAsync(me.send[slot], data, n * sizeof(long long),
                                cudaMemcpyDefault, stream));
    Comm::check(cudaEventRecord(me.sent[slot], stream));
    comm->barrier();
    // The same on every rank: n is a global size.
    bool pieces = (size_t)(n_ranks - 1) * n > comm->capacity;
    auto lo = [&](int r) { return pieces ? n * r / n_ranks : 0; };
    auto hi = [&](int r) { return pieces ? n * (r + 1) / n_ranks : n; };
    size_t first = lo(rank), m = hi(rank) - first;
    for (int r = 0, p = 0; r < n_ranks; ++r) {
        if (r == rank) continue;
        cudaStreamWaitEvent(stream, comm->ranks[r].sent[slot]);
        comm->pull(rank, r, me.recv[slot] + p++ * m,
                   comm->ranks[r].send[slot] + first, m, stream);
    }
    if (m > 0) {
        sum_pieces_kernel<<<(int)std::min<size_t>((m + 255) / 256, 1024), 256,
                            0, stream>>>(me.recv[slot], data + first, m,
                                         n_ranks - 1);
        Comm::check(cudaGetLastError());
    }
    if (pieces) {
        // The summed piece replaces this rank's own in its send slot (the
        // others read only their pieces of it), then the others' are copied.
        Comm::check(cudaMemcpyAsync(me.send[slot] + first, data + first,
                                    m * sizeof(long long), cudaMemcpyDefault,
                                    stream));
        Comm::check(cudaEventRecord(me.reduced[slot], stream));
        comm->barrier();
        for (int r = 0; r < n_ranks; ++r) {
            if (r == rank) continue;
            cudaStreamWaitEvent(stream, comm->ranks[r].reduced[slot]);
            comm->pull(rank, r, data + lo(r), comm->ranks[r].send[slot] + lo(r),
                       hi(r) - lo(r), stream);
        }
    }
    Comm::check(cudaEventRecord(me.read[slot], stream));
}

// data (n values on rank's GPU, stream-ordered) = its sum over the GPUs, in
// exchanges of at most `capacity` values. Every rank makes the same sequence
// of calls; without comm this is a no-op. Summed pieces land in data by peer
// copy: read it with a kernel, or synchronize before freeing it.
inline void allreduce(Comm* comm, int rank, long long* data, size_t n,
                      cudaStream_t stream) {
    if (!comm || comm->ranks.size() < 2) return;
    for (size_t i = 0; i < n; i += comm->capacity)
        exchange(comm, rank, data + i, std::min(comm->capacity, n - i), stream);
}

}  // namespace harmony_comm
