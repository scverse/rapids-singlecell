#pragma once

// Exact sums of int64 values across the GPUs of one process. One host thread
// per GPU runs the same sequence of allreduce calls; values move by peer copy,
// or through pinned host memory when peer copies were not validated. When the
// others' values fit the receive slot, every GPU copies and adds them; larger
// sums go in P pieces: GPU r adds piece r of every GPU's values, then copies
// the others' summed pieces (2 (P-1)/P instead of P-1 times the values).
// Sums over cells sorted by group move only each GPU's groups, and values
// split between the GPUs only each GPU's part (allgather), to every GPU or
// to one of them.
// Integer sums are exact in any order, so all GPUs hold the bits of a
// single-GPU run. The caller allocates all buffers.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "../nb_types.h"

namespace harmony_comm {

struct Comm {
    struct Rank {
        int device;
        long long *send[2], *recv[2];  // recv: capacity + n_ranks values
        // reduced: this rank's summed piece is in send; read: slot free again
        cudaEvent_t sent[2]{}, reduced[2]{}, read[2]{};
        cudaStream_t copies{};  // see pull
        cudaEvent_t queued{}, copied{};
        // The clustering sums a block's counts on `side` while the next block
        // is assigned: ready once the counts are in their slot, summed once
        // their sum is.
        cudaStream_t side{};
        cudaEvent_t ready[2]{}, summed[2]{};
        unsigned count = 0;
    };
    std::vector<Rank> ranks;
    size_t capacity;  // values per exchange (send slot)
    bool peer;
    std::vector<size_t> groups;  // per GPU: first and end group of its cells
    std::atomic<int> arrived{0};
    std::atomic<unsigned> generation{0};
    std::atomic<bool> aborted{false};  // a rank failed: release the others

    // buffers: per rank send[0], send[1] (device, or pinned host without
    // peer copies), recv[0], recv[1] (device). peer: the caller validated
    // peer copies between every two devices, which enabled peer access.
    Comm(const std::vector<int>& devices, const std::vector<uintptr_t>& buffers,
         size_t capacity, bool peer, const std::vector<size_t>& groups)
        : ranks(devices.size()),
          capacity(capacity),
          peer(peer),
          groups(groups) {
        if (buffers.size() != 4 * devices.size() ||
            groups.size() != 2 * devices.size())
            throw std::invalid_argument(
                "harmony comm: 4 buffers and 2 group bounds per GPU");
        int previous;
        cudaGetDevice(&previous);
        try {
            for (size_t r = 0; r < devices.size(); ++r) {
                Rank& me = ranks[r];
                me.device = devices[r];
                check(cudaSetDevice(me.device));
                for (int s = 0; s < 2; ++s) {
                    me.send[s] =
                        reinterpret_cast<long long*>(buffers[4 * r + s]);
                    me.recv[s] =
                        reinterpret_cast<long long*>(buffers[4 * r + 2 + s]);
                    for (cudaEvent_t* e :
                         {&me.sent[s], &me.reduced[s], &me.read[s],
                          &me.ready[s], &me.summed[s]})
                        create(e, cudaEventCreateWithFlags,
                               cudaEventDisableTiming);
                }
                for (cudaStream_t* s : {&me.copies, &me.side})
                    create(s, cudaStreamCreateWithFlags, cudaStreamNonBlocking);
                for (cudaEvent_t* e : {&me.queued, &me.copied})
                    create(e, cudaEventCreateWithFlags, cudaEventDisableTiming);
            }
        } catch (...) {  // no destructor runs on a half-built object
            release(previous);
            throw;
        }
        cudaSetDevice(previous);
    }
    ~Comm() {
        int previous;
        cudaGetDevice(&previous);
        release(previous);
    }
    // *handle = a new event or stream; left null when that fails (a failed
    // create can write an invalid handle).
    template <typename Handle>
    static void create(Handle* handle, cudaError_t (*make)(Handle*, unsigned),
                       unsigned flags) {
        Handle made;
        check(make(&made, flags));
        *handle = made;
    }
    // Destroys the streams and events created, each on its GPU, then makes
    // `device` current again.
    void release(int device) {
        for (Rank& me : ranks) {
            if (!me.sent[0]) continue;  // none created on this GPU
            cudaSetDevice(me.device);
            for (cudaStream_t s : {me.copies, me.side})
                if (s) cudaStreamDestroy(s);
            for (cudaEvent_t e :
                 {me.sent[0], me.reduced[0], me.read[0], me.ready[0],
                  me.summed[0], me.sent[1], me.reduced[1], me.read[1],
                  me.ready[1], me.summed[1], me.queued, me.copied})
                if (e) cudaEventDestroy(e);
        }
        cudaSetDevice(device);
    }
    // The events every exchange of a rank precedes: all its work precedes its
    // sent, reduced or read event (its pulls the copied event), and a slot's
    // reuse waits for its previous use, so the latest of them cover every
    // exchange.
    template <typename F>
    void each_event(F&& f) {
        for (Rank& me : ranks)
            for (cudaEvent_t e :
                 {me.sent[0], me.reduced[0], me.read[0], me.sent[1],
                  me.reduced[1], me.read[1], me.copied})
                if (e) f(e);
    }
    // After a failure, other ranks may still have exchanges in flight on the
    // buffers: wait for them before the buffers are released. A failed GPU
    // may never complete them (nor the streams waiting on them): give up
    // after `timeout` seconds and return false; wait() then blocks until it
    // does, for a thread that holds the buffers meanwhile.
    bool synchronize(double timeout) {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout);
        while (true) {
            bool pending = false;
            each_event([&](cudaEvent_t e) {
                pending |= cudaEventQuery(e) == cudaErrorNotReady;
            });
            if (!pending) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    void wait() {
        each_event([](cudaEvent_t e) { cudaEventSynchronize(e); });
    }
    static void check(cudaError_t status) {
        cuda_check(status, "harmony comm");
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
    // Stages n values at data in the next send slot of `rank`, once every
    // rank (this one too, possibly on another stream) is done with that
    // slot's previous exchange, and waits for the other ranks to have staged
    // theirs. Returns the slot.
    int stage(int rank, const long long* data, size_t n, cudaStream_t stream) {
        Rank& me = ranks[rank];
        int slot = me.count++ & 1;
        for (Rank& peer : ranks)
            check(cudaStreamWaitEvent(stream, peer.read[slot]));
        check(cudaMemcpyAsync(me.send[slot], data, n * sizeof(long long),
                              cudaMemcpyDefault, stream));
        check(cudaEventRecord(me.sent[slot], stream));
        barrier();
        return slot;
    }
    // n values at src, in rank `from`'s send slot, to dst on rank `to`'s GPU,
    // in order on `stream` but copied on the rank's own stream: on a legacy
    // default stream, a peer copy would also wait for the other GPU's work.
    void pull(int to, int from, long long* dst, const long long* src, size_t n,
              cudaStream_t stream) {
        if (n == 0) return;
        Rank& me = ranks[to];
        check(cudaEventRecord(me.queued, stream));
        check(cudaStreamWaitEvent(me.copies, me.queued));
        // Peer copies: stream-ordered pool memory needs the peer API.
        check(peer ? cudaMemcpyPeerAsync(dst, me.device, src,
                                         ranks[from].device, n * 8, me.copies)
                   : cudaMemcpyAsync(dst, src, n * 8, cudaMemcpyHostToDevice,
                                     me.copies));
        check(cudaEventRecord(me.copied, me.copies));
        check(cudaStreamWaitEvent(stream, me.copied));
    }
};

// cudaStreamSynchronize(stream) on a rank of comm (none: plain). A failed GPU
// may never complete the events the stream waits on, so this polls and gives
// up once a rank aborted, leaving the stream's work behind.
inline void synchronize(Comm* comm, cudaStream_t stream, const char* what) {
    if (comm)
        while (cudaStreamQuery(stream) == cudaErrorNotReady) {
            if (comm->aborted.load())
                throw std::runtime_error("harmony comm: another GPU failed");
            std::this_thread::yield();
        }
    cuda_check(cudaStreamSynchronize(stream), what);
}

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
    int n_ranks = (int)comm->ranks.size();
    int slot = comm->stage(rank, data, n, stream);
    // The same on every rank: n is a global size.
    bool pieces = (size_t)(n_ranks - 1) * n > comm->capacity;
    auto lo = [&](int r) { return pieces ? n * r / n_ranks : 0; };
    auto hi = [&](int r) { return pieces ? n * (r + 1) / n_ranks : n; };
    size_t first = lo(rank), m = hi(rank) - first;
    for (int r = 0, p = 0; r < n_ranks; ++r) {
        if (r == rank) continue;
        Comm::check(cudaStreamWaitEvent(stream, comm->ranks[r].sent[slot]));
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
            Comm::check(
                cudaStreamWaitEvent(stream, comm->ranks[r].reduced[slot]));
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

// Every GPU's data (stream-ordered), or only GPU `to`'s (-1: every one's),
// gets values range(r) = {lo, hi} of GPU r (disjoint ranges): per exchange
// the other GPUs stage the next `capacity` values of their range and the
// receiving ones copy them through their receive slot. The peer copy lands
// in the slot, not in data: the caller may free data (scratch) before the
// stream drains, and cudaFree with a peer copy pending into the buffer
// faults both GPUs.
template <typename Range>
inline void allgather(Comm* comm, int rank, long long* data, Range range,
                      cudaStream_t stream, int to = -1) {
    if (!comm || comm->ranks.size() < 2) return;
    auto& me = comm->ranks[rank];
    size_t c = comm->capacity, n_rounds = 0;
    for (int r = 0; r < (int)comm->ranks.size(); ++r)
        n_rounds =
            std::max(n_rounds, (range(r).second - range(r).first + c - 1) / c);
    for (size_t w = 0; w < n_rounds; ++w) {
        auto piece = [&](int r) {
            auto [lo, hi] = range(r);
            lo = std::min(hi, lo + w * c);
            return std::pair{lo, std::min(hi, lo + c)};
        };
        auto [lo, hi] = piece(rank);
        int slot =
            comm->stage(rank, data + lo, rank == to ? 0 : hi - lo, stream);
        if (to < 0 || to == rank)
            for (int r = 0; r < (int)comm->ranks.size(); ++r) {
                if (r == rank) continue;
                auto [a, b] = piece(r);
                Comm::check(
                    cudaStreamWaitEvent(stream, comm->ranks[r].sent[slot]));
                comm->pull(rank, r, me.recv[slot], comm->ranks[r].send[slot],
                           b - a, stream);
                Comm::check(cudaMemcpyAsync(data + a, me.recv[slot],
                                            (b - a) * sizeof(long long),
                                            cudaMemcpyDeviceToDevice, stream));
            }
        Comm::check(cudaEventRecord(me.read[slot], stream));
    }
}

// Sums of group-major rows of n values over cells sorted by group, zero
// outside each GPU's groups (Comm::groups), on every GPU or only on GPU `to`:
// the groups straddling GPUs are summed (allreduce), then the rows from each
// GPU's first group to the next one's are gathered (about (P-1)/P of the
// rows instead of P-1 times).
inline void allgather_groups(Comm* comm, int rank, long long* data, size_t n,
                             cudaStream_t stream, int to = -1) {
    if (!comm || comm->ranks.size() < 2) return;
    const auto& g = comm->groups;
    size_t n_ranks = comm->ranks.size(), summed = SIZE_MAX;
    for (size_t r = 1; r < n_ranks; ++r)
        if (g[2 * r] < g[2 * r - 1] && g[2 * r] != summed) {
            summed = g[2 * r];
            allreduce(comm, rank, data + summed * n, n, stream);
        }
    allgather(
        comm, rank, data,
        [&](size_t r) {
            size_t end = r + 1 < n_ranks ? g[2 * r + 2] : g[2 * r + 1];
            return std::pair{g[2 * r] * n, end * n};
        },
        stream, to);
}

}  // namespace harmony_comm
