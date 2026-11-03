#pragma once

// rapids-singlecell's multi-GPU entry point (`leiden_mg`): every partition in
// this process with its own stream; PeerExchange moves their data with peer /
// same-device copies, or synchronously through pinned host memory where peer
// copies are not validated (then every copy between partitions does, also on
// one GPU).

#include "partition.cuh"

namespace leiden {

class PeerExchange final : public Exchange {
   public:
    PeerExchange(std::vector<int> devices, std::vector<cudaStream_t> streams,
                 bool peer)
        : dev_(std::move(devices)), s_(std::move(streams)), peer_(peer) {
        ev_.assign(dev_.size(), nullptr);
        try {
            for (std::size_t i = 0; i < dev_.size(); ++i) {
                set(static_cast<int>(i));
                cuda_check(
                    cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming),
                    "leiden: exchange event");
            }
            if (!peer_)
                cuda_check(cudaMallocHost(&stage_, kStageBytes),
                           "leiden: exchange staging buffer");
        } catch (...) {  // no destructor runs for a failed constructor
            release();
            throw;
        }
    }
    ~PeerExchange() override {
        release();
    }
    PeerExchange(const PeerExchange&) = delete;
    PeerExchange& operator=(const PeerExchange&) = delete;

    int parts() const override {
        return static_cast<int>(dev_.size());
    }
    int n_local() const override {
        return parts();
    }
    int global(int i) const override {
        return i;
    }
    void allgather_slices(void* const* bufs, int elem,
                          const i64* bounds) override {
        barrier();
        for (int d = 0; d < parts(); ++d)
            for (int q = 0; q < parts(); ++q) {
                if (q == d) continue;
                const i64 lo = bounds[q] * elem;
                copy(d, at(bufs[d], lo), q, at(bufs[q], lo),
                     (bounds[q + 1] - bounds[q]) * elem);
            }
        barrier();
    }
    void allgatherv(void* const* bufs, int elem, const i64* counts) override {
        barrier();
        i64 displ = 0;
        for (int q = 0; q < parts(); ++q) {
            for (int d = 0; d < parts(); ++d)
                if (d != q)
                    copy(d, at(bufs[d], displ * elem), q,
                         at(bufs[q], displ * elem), counts[q] * elem);
            displ += counts[q];
        }
        barrier();
    }
    void allgather_host(const i64* in, int k, i64* out) override {
        std::copy(in, in + static_cast<std::size_t>(parts()) * k, out);
    }
    void allreduce_or(u32* const* in, u32* const* out, u32* const* scratch,
                      i64 words) override {
        barrier();
        for (int d = 0; d < parts(); ++d) {
            int j = 0;
            for (int q = 0; q < parts(); ++q)
                if (q != d)
                    copy(d, scratch[d] + j++ * words, q, in[q],
                         words * static_cast<i64>(sizeof(u32)));
        }
        barrier();  // every pull has landed before an in-place OR
        for (int d = 0; d < parts(); ++d) {
            set(d);
            part_or_kernel<<<grid_items(words), kBlock, 0, s_[d]>>>(
                in[d], out[d], scratch[d], parts() - 1, words);
            CUDA_CHECK_LAST_ERROR(part_or_kernel);
        }
        barrier();
    }
    void gatherv(int root, const void* const* send, int elem, const i64* counts,
                 void* recv) override {
        barrier();
        i64 displ = 0;
        for (int q = 0; q < parts(); ++q) {
            copy(root, at(recv, displ * elem), q, send[q], counts[q] * elem);
            displ += counts[q];
        }
        barrier();
    }
    void bcast(int root, void* const* bufs, i64 bytes) override {
        barrier();
        for (int d = 0; d < parts(); ++d)
            if (d != root) copy(d, bufs[d], root, bufs[root], bytes);
        barrier();
    }

   private:
    static constexpr std::size_t kStageBytes = std::size_t{1} << 23;
    std::vector<int> dev_;
    std::vector<cudaStream_t> s_;
    std::vector<cudaEvent_t> ev_;
    bool peer_;
    void* stage_ = nullptr;

    static void* at(const void* p, i64 off) {
        return static_cast<char*>(const_cast<void*>(p)) + off;
    }
    void set(int i) {
        cuda_check(cudaSetDevice(dev_[i]), "leiden: cudaSetDevice");
    }
    void release() {
        for (std::size_t i = 0; i < dev_.size(); ++i) {
            if (!ev_[i]) continue;
            cudaSetDevice(dev_[i]);
            cudaEventDestroy(ev_[i]);
            ev_[i] = nullptr;
        }
        if (stage_) cudaFreeHost(stage_);
        stage_ = nullptr;
    }
    // Every stream waits for the work queued so far on every other one.
    void barrier() {
        for (int i = 0; i < parts(); ++i) {
            set(i);
            cuda_check(cudaEventRecord(ev_[i], s_[i]), "leiden: exchange");
        }
        for (int i = 0; i < parts(); ++i) {
            set(i);
            for (int j = 0; j < parts(); ++j)
                if (j != i)
                    cuda_check(cudaStreamWaitEvent(s_[i], ev_[j], 0),
                               "leiden: exchange wait");
        }
    }
    // `bytes` of partition q at src to partition d at dst, on d's stream.
    void copy(int d, void* dst, int q, const void* src, i64 bytes) {
        if (bytes <= 0 || (d == q && dst == src)) return;
        if (peer_ || d == q) {
            set(d);
            if (dev_[d] == dev_[q]) {
                cuda_check(cudaMemcpyAsync(dst, src, bytes,
                                           cudaMemcpyDeviceToDevice, s_[d]),
                           "leiden: exchange copy");
            } else {
                cuda_check(cudaMemcpyPeerAsync(dst, dev_[d], src, dev_[q],
                                               bytes, s_[d]),
                           "leiden: exchange peer copy");
            }
            return;
        }
        for (i64 off = 0; off < bytes; off += static_cast<i64>(kStageBytes)) {
            const std::size_t m = static_cast<std::size_t>(
                std::min<i64>(static_cast<i64>(kStageBytes), bytes - off));
            set(q);
            cuda_check(cudaMemcpyAsync(stage_, at(src, off), m,
                                       cudaMemcpyDeviceToHost, s_[q]),
                       "leiden: exchange staging");
            cuda_check(cudaStreamSynchronize(s_[q]), "leiden: exchange sync");
            set(d);
            cuda_check(cudaMemcpyAsync(at(dst, off), stage_, m,
                                       cudaMemcpyHostToDevice, s_[d]),
                       "leiden: exchange staging");
            cuda_check(cudaStreamSynchronize(s_[d]), "leiden: exchange sync");
        }
    }
};

// Streams and pinned readback buffers of a call (one per partition).
struct PartResources {
    std::vector<int> dev;
    std::vector<cudaStream_t> s;
    void* pinned = nullptr;
    int previous = 0;

    explicit PartResources(const std::vector<int>& devices) : dev(devices) {
        cuda_check(cudaGetDevice(&previous), "leiden: cudaGetDevice");
        s.assign(dev.size(), nullptr);
        try {
            for (std::size_t i = 0; i < dev.size(); ++i) {
                cuda_check(cudaSetDevice(dev[i]), "leiden: cudaSetDevice");
                cuda_check(
                    cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking),
                    "leiden: partition stream");
            }
            cuda_check(cudaMallocHost(&pinned, dev.size() * kPinnedBytes),
                       "leiden: pinned readback buffer");
        } catch (...) {  // no destructor runs for a failed constructor
            release();
            throw;
        }
    }
    void synchronize() {
        for (std::size_t i = 0; i < dev.size(); ++i) {
            if (!s[i]) continue;
            cudaSetDevice(dev[i]);
            cudaStreamSynchronize(s[i]);
        }
    }
    ~PartResources() {
        release();
    }
    void release() {
        synchronize();
        for (std::size_t i = 0; i < dev.size(); ++i) {
            if (!s[i]) continue;
            cudaSetDevice(dev[i]);
            cudaStreamDestroy(s[i]);
            s[i] = nullptr;
        }
        if (pinned) cudaFreeHost(pinned);
        pinned = nullptr;
        cudaSetDevice(previous);
        cudaGetLastError();
    }
    PartResources(const PartResources&) = delete;
    PartResources& operator=(const PartResources&) = delete;
};

template <typename WI, typename Device>
nb::dict call_leiden_mg(
    const std::vector<gpu_array_c<const i64, Device>>& indptrs,
    const std::vector<gpu_array_c<const int, Device>>& indices,
    const std::vector<gpu_array_c<const WI, Device>>& data,
    const std::vector<int>& devices, const std::vector<i64>& bounds,
    bool use_weights, const nb::dict& check,
    std::vector<gpu_array_c<std::uint8_t, Device>>& workspaces, bool peer,
    optional_array<int, Device>& labels_out,
    const std::vector<double>& resolutions,
    const std::vector<std::uint32_t>& seeds, int n_iterations,
    const optional_array<const int, Device>& init,
    const std::vector<double>& arena_factors) {
    const std::size_t P = indptrs.size();
    if (P < 1 || P > static_cast<std::size_t>(kMaxParts))
        throw std::invalid_argument("leiden_mg: 1 to 32 partitions");
    if (indices.size() != P || data.size() != P || devices.size() != P ||
        bounds.size() != P + 1)
        throw std::invalid_argument(
            "leiden_mg: indptrs, indices, data, devices and P + 1 bounds");
    if (indptrs[0].ndim() != 1 || indptrs[0].size() < 1)
        throw std::invalid_argument("indptr must be 1-D with n + 1 entries");
    const i64 n = static_cast<i64>(indptrs[0].size()) - 1;
    if (n >= kMaxVertices - 1)
        throw std::invalid_argument("leiden: n must be < 2^30 - 1");
    if (bounds[0] != 0 || bounds[P] != n)
        throw std::invalid_argument("leiden_mg: bounds must run from 0 to n");
    std::vector<i64> nnz(P);
    i64 nnz_total = 0;
    for (std::size_t p = 0; p < P; ++p) {
        if (bounds[p + 1] < bounds[p] ||
            (p > 0 && bounds[p] % kWarp != 0 && bounds[p] != n))
            throw std::invalid_argument(
                "leiden_mg: bounds must be non-decreasing multiples of 32 "
                "(or n)");
        if (static_cast<i64>(indptrs[p].size()) != n + 1)
            throw std::invalid_argument("leiden_mg: indptrs of n + 1 entries");
        nnz[p] = static_cast<i64>(indices[p].size());
        if (static_cast<i64>(data[p].size()) != nnz[p])
            throw std::invalid_argument(
                "data and indices must have equal length");
        nnz_total += nnz[p];
    }
    const std::size_t n_res = resolutions.size();
    if (n_res == 0 || seeds.size() != n_res)
        throw std::invalid_argument(
            "one seed per resolution (at least one) is required");
    for (double g : resolutions) check_gamma(g);
    if (!(n_iterations == -1 ||
          (n_iterations >= 1 && n_iterations < (1 << 16))))
        throw std::invalid_argument(
            "n_iterations must be a positive integer or -1");
    if (arena_factors.size() != P)
        throw std::invalid_argument(
            "leiden_mg: one arena factor per partition");
    for (double f : arena_factors)
        if (!(f > 0.0))
            throw std::invalid_argument("arena factors must be > 0");
    const bool run = !workspaces.empty();
    if (run && workspaces.size() != P)
        throw std::invalid_argument("leiden_mg: one workspace per partition");
    if (run != labels_out.has_value())
        throw std::invalid_argument("workspaces and labels_out go together");
    if (labels_out &&
        (labels_out->ndim() != 2 || labels_out->shape(0) != n_res ||
         static_cast<i64>(labels_out->shape(1)) != n))
        throw std::invalid_argument(
            "labels_out must have shape (len(resolutions), n)");
    if (init && static_cast<i64>(init->size()) != n)
        throw std::invalid_argument("initial_membership must have n entries");
    DriverInput in;
    in.n = n;
    in.nnz = nnz_total;
    in.use_weights = use_weights;
    in.info = info_from_dict(check);
    in.gammas = resolutions;
    in.seeds.assign(seeds.begin(), seeds.end());
    in.init = init ? init->data() : nullptr;
    in.labels_out = labels_out ? labels_out->data() : nullptr;
    DriverParams dp;
    dp.n_iterations = n_iterations;
    dp.max_replicas = 1;
    std::vector<LayoutParams> lp(P);
    for (std::size_t p = 0; p < P; ++p) {
        lp[p].n = n;
        lp[p].nnz = nnz[p];
        lp[p].wkind = call_wkind(in.info, use_weights,
                                 std::is_same_v<WI, float>, resolutions);
        lp[p].arena_factor = arena_factors[p];
        lp[p].arena_entries = part_arena_entries(
            static_cast<int>(p), static_cast<int>(P), nnz[p], nnz_total);
        lp[p].subrounds = dp.subrounds;
    }
    nb::dict d;
    nb::list need;
    for (std::size_t p = 0; p < P; ++p)
        need.append(make_layout(lp[p], nullptr).total);
    d["required_bytes"] = need;
    if (!run) {
        d["status"] = "size";
        return d;
    }
    std::vector<const i64*> ip(P);
    std::vector<const int*> ix(P);
    std::vector<const WI*> wd(P);
    for (std::size_t p = 0; p < P; ++p) {
        ip[p] = indptrs[p].data();
        ix[p] = indices[p].data();
        wd[p] = data[p].data();
    }
    std::vector<Part> parts(P);
    DriverResult out;
    bool ok = false;
    {
        nb::gil_scoped_release release;
        PartResources res(devices);
        try {
            for (std::size_t p = 0; p < P; ++p) {
                Part& part = parts[p];
                part.global = static_cast<int>(p);
                part.device = devices[p];
                part.s = res.s[p];
                part.pinned = static_cast<char*>(res.pinned) + p * kPinnedBytes;
                part.v0 = bounds[p];
                part.v1 = bounds[p + 1];
                part.nnz = nnz[p];
                part.L = make_layout(lp[p], workspaces[p].data());
                if (part.L.total > workspaces[p].size())
                    throw std::invalid_argument(
                        "leiden_mg: workspace too small for its layout");
                part.L.pinned = part.pinned;
            }
            PeerExchange ex(devices, res.s, peer);
            const PartitionedDriver::Quantize quantize = [&](Part& part,
                                                             double gamma) {
                const std::size_t p = static_cast<std::size_t>(part.global);
                return run_quantize<i64, int, WI>(
                    ip[p], ix[p], wd[p], n, part.nnz, use_weights, gamma,
                    in.info, part.L, dp.th, part.s);
            };
            PartitionedDriver drv(parts, ex, dp, n, bounds);
            ok = drv.run(in, quantize, out);
            res.synchronize();
        } catch (...) {
            res.synchronize();  // no kernel may touch a returned workspace
            throw;
        }
    }
    if (!ok) {  // the factor each overflowed arena needs (0: no overflow)
        nb::list f;
        for (const Part& part : parts)
            f.append(
                part.L.arena.required > 0
                    ? arena_required_factor(part.L.p, part.L.arena.required)
                    : 0.0);
        d["status"] = "workspace_too_small";
        d["required_arena_factor"] = f;
        return d;
    }
    d["status"] = "ok";
    nb::list q, k;
    for (const ResolutionResult& x : out.res) {
        q.append(x.modularity);
        k.append(x.n_clusters);
    }
    d["modularity"] = q;
    d["n_clusters"] = k;
    return d;
}

// Level 0 split into P = len(indptrs) row ranges [bounds[p], bounds[p + 1]),
// partition p on devices[p] (several may share one): indptrs[p] the
// full-length int64 offsets of its rows (other rows empty), indices[p] /
// data[p] its entries. `check`: ingest_check of the whole graph. Without
// workspaces the call only sizes them (one per partition); labels_out lives
// on devices[0]; arena_factors: one per partition (see part_arena_entries).
// Status "ok" or "workspace_too_small" with the arena factor a rerun needs
// for each partition (0: no overflow).
template <typename WI, typename Device>
void register_mg_typed(nb::module_& m) {
    m.def(
        "leiden_mg",
        [](std::vector<gpu_array_c<const i64, Device>> indptrs,
           std::vector<gpu_array_c<const int, Device>> indices,
           std::vector<gpu_array_c<const WI, Device>> data,
           std::vector<int> devices, std::vector<long long> bounds,
           bool use_weights, nb::dict check,
           std::vector<gpu_array_c<std::uint8_t, Device>> workspaces, bool peer,
           optional_array<int, Device> labels_out,
           std::vector<double> resolutions, std::vector<std::uint32_t> seeds,
           int n_iterations,
           optional_array<const int, Device> initial_membership,
           std::vector<double> arena_factors) {
            return call_leiden_mg<WI, Device>(
                indptrs, indices, data, devices,
                std::vector<i64>(bounds.begin(), bounds.end()), use_weights,
                check, workspaces, peer, labels_out, resolutions, seeds,
                n_iterations, initial_membership, arena_factors);
        },
        "indptrs"_a, nb::kw_only(), "indices"_a, "data"_a, "devices"_a,
        "bounds"_a, "use_weights"_a, "check"_a, "workspaces"_a, "peer"_a,
        "labels_out"_a.noconvert().none() = nb::none(), "resolutions"_a,
        "seeds"_a, "n_iterations"_a,
        "initial_membership"_a.noconvert().none() = nb::none(),
        "arena_factors"_a);
}

template <typename Device>
void register_mg_bindings(nb::module_& m) {
    register_mg_typed<float, Device>(m);
    register_mg_typed<double, Device>(m);
}

inline void register_mg(nb::module_& m) {
    REGISTER_GPU_BINDINGS(register_mg_bindings, m);
}

}  // namespace leiden
