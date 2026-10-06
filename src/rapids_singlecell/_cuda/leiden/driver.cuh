#pragma once

// Host driver and the module's API. Python checks and fixes up the input
// (ingest_check) and sizes the workspace; one `leiden()` call then runs:
//   I2 (+ I4) when s(gamma) changes                                    -> S2
//   per iteration, per level: LOCAL_MOVE (DOWN) + COMPACT              -> SL1
//     (C == n: all-active TOP sweeps certify or re-enter), PLEIDEN_R +
//     AGGREGATE stage 1 -> SL2 (nothing merged: re-enter TOP, then the
//     CC-piece last resort), AGGREGATE stage 2; projection to level 0 with
//     V-cycle UP sweeps, the CC split and, with replicas, the best slice
//   exact Q, size-ordered labels

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <optional>
#include <utility>

#include "kernels_aggregate.cuh"
#include "kernels_final.cuh"
#include "kernels_ingest.cuh"
#include "kernels_move.cuh"

namespace leiden {

using namespace nb::literals;

constexpr double kDeltaQStop = 1e-6;  // n_iterations = -1, rule (b) (§4.14)

// Knobs of one call (the shipped binding sets n_iterations and arena_factor).
struct DriverParams {
    int n_iterations = 2;  // >= 1, or -1 (Python maps > 20 to -1)
    int move_sweeps = 4;   // DOWN cap
    int subrounds = kNumSubrounds;
    int vcycle_sweeps = 4;  // UP cap; 0 = no V-cycle
    bool vcycle_last = false;
    int top_sweeps = 32;  // TOP cap (§4.10)
    int max_levels = kMaxLevels;
    int max_reentry = 4;  // TOP re-entries per level (Lemma R)
    int max_iter_until_stable = 20;
    int max_replicas = kMaxReplicas;
    i64 replica_nnz_budget = 1ll << 19;
    double arena_factor = 1.9;
    int graph_mode = 2;  // 0: launches, 1: sweep graph replays, 2: WHILE node
    ClassThresholds th{};
    GatherClasses gather{};
    TreeClasses trees{};
};

inline int n_replicas(i64 nnz_counted, int max_replicas, i64 budget) {
    if (nnz_counted <= 0) return 1;
    const i64 r = budget / nnz_counted;
    return static_cast<int>(std::min<i64>(max_replicas, std::max<i64>(1, r)));
}

inline WKind call_wkind(const IngestInfo& info, bool weighted, bool f32,
                        const std::vector<double>& gammas) {
    WKind k = WKind::UNIT;
    bool first = true;
    for (double g : gammas) {
        const int s = scale_for_gamma(scale_s0(info.n_counted, info.wmax), g);
        const WKind kg = level0_kind(info.flags, weighted, f32, s);
        k = first ? kg : (kg == k ? k : WKind::I64);
        first = false;
    }
    return k;
}

__global__ void drv_iota_kernel(int* __restrict__ P, i64 n) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n;
         v += stride)
        P[v] = static_cast<int>(v);
}

__global__ void drv_replicate_partition_kernel(int* __restrict__ P, i64 n,
                                               int R) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n;
         v += stride) {
        const int p = P[v];
        for (int r = 1; r < R; ++r)
            P[static_cast<i64>(r) * n + v] = p + static_cast<int>(r * n);
    }
}

// COMPACT of one replica slice: flags of its ids (< C), then their ranks.
__global__ void drv_slice_flags_kernel(const int* __restrict__ P, i64 n, i64 C,
                                       int* __restrict__ flags, Control* ctl) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n;
         v += stride) {
        const int c = P[v];
        if (c < 0 || c >= C)
            atomicOr(&ctl->flags, kFlagBadLabel);
        else
            flags[c] = 1;  // idempotent store
    }
}

__global__ void drv_slice_relabel_kernel(const int* __restrict__ P_in, i64 n,
                                         i64 C, const int* __restrict__ rank,
                                         int* __restrict__ P_out,
                                         Control* ctl) {
    const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
    for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n;
         v += stride) {
        const int c = P_in[v];
        P_out[v] = (c >= 0 && c < C) ? rank[c] : 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_components = rank[C];
}

// One iteration: a vertex moved (stop rule), and the dev module's counters.
struct IterationStat {
    bool moved = false;
    int levels = 0;
    i64 cc_pieces = 0;
    int reentries = 0;
    int fallbacks = 0;
    bool uncertified = false;
    bool top_cap_hit = false;
};

struct ResolutionResult {
    double modularity = 0.0;
    i64 n_clusters = 0;
    std::vector<IterationStat> iters;
};

struct DriverResult {
    std::string status = "ok";
    double required_arena_factor = 0.0;
    std::size_t required_bytes = 0;
    std::vector<ResolutionResult> res;
};

struct LevelGraph {
    const i64* indptr = nullptr;
    const int* indices = nullptr;
    const float* wf = nullptr;  // fp32: level 0 scaled by 2^s on the fly,
                                // coarse levels exact integers (scale 1)
    const i64* wq = nullptr;    // int64 W0q (level 0)
    i64 unit = 1;               // UNIT kind: the quantised unit weight
    double scale = 1.0;
    const i64* khat = nullptr;
    i64 n = 0, nnz = 0;
    i64 cls[kNumClasses] = {};  // vertices per degree class
    int* cmap = nullptr;        // to level l + 1 (arena bottom)

    template <typename Fn>
    void with_w(Fn&& fn) const {
        dispatch_weights(wf, wq, scale, unit, std::forward<Fn>(fn));
    }
};

class LeidenDriver {
   public:
    LeidenDriver(Layout& L, const DriverParams& dp, cudaStream_t s)
        : L_(L), dp_(dp), s_(s) {
    }

    void set_level0(const QuantizeResult& q, i64 n, i64 nnz) {
        q_ = q;
        n_ = n;
        nnz_ = nnz;
        g0_ = LevelGraph{};
        g0_.indptr = q.indptr;
        g0_.indices = q.indices;
        g0_.wf = q.wkind == WKind::F32 ? q.wf32 : nullptr;
        g0_.wq = q.wkind == WKind::I64 ? q.wq : nullptr;
        g0_.unit = q.wkind == WKind::UNIT ? q.unit_q : 1;
        g0_.scale = std::ldexp(1.0, q.s);
        g0_.khat = L_.l0.khat;
        g0_.n = n;
        g0_.nnz = nnz;
        for (int c = 0; c < kNumClasses; ++c) g0_.cls[c] = q.class_count[c];
    }

    // One resolution; false if the arena overflowed.
    bool run_resolution(double gamma, u32 seed, const int* init, int R,
                        int* labels_out, ResolutionResult& out) {
        const i64 n = n_;
        const i64 two_m = q_.two_m_hat;
        if (two_m <= 0) {  // no counted entry: every vertex its own cluster
            out.n_clusters = n;
            drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(labels_out, n);
            CUDA_CHECK_LAST_ERROR(drv_iota_kernel);
            return true;
        }
        lam_ = gamma / static_cast<double>(two_m);  // one IEEE division
        gamma_ = gamma;
        seed_ = seed;
        s64_ = seed64(seed);
        int* P0 = L_.persist.P0;
        if (init) {
            cuda_check(cudaMemcpyAsync(P0, init, n * sizeof(int),
                                       cudaMemcpyDeviceToDevice, s_),
                       "leiden: initial membership");
        } else {
            drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n);
            CUDA_CHECK_LAST_ERROR(drv_iota_kernel);
        }
        i64 C = n;  // id bound of P0
        const int T = dp_.n_iterations;
        const int V = dp_.vcycle_sweeps;
        int it = 0;
        double q_prev = -std::numeric_limits<double>::infinity();
        bool stop = false;
        while (true) {
            const bool final = (T >= 1 && it == T - 1) || (T == -1 && stop);
            const bool vcycle = V > 0 && (!final || T == 1 || dp_.vcycle_last);
            IterationStat stat;
            if (it == 0 && R > 1) {
                drv_replicate_partition_kernel<<<grid_items(n), kBlock, 0,
                                                 s_>>>(P0, n, R);
                CUDA_CHECK_LAST_ERROR(drv_replicate_partition_kernel);
                const LevelGraph gu = union_level(R);
                i64 Cb = 0;
                if (!leiden_iteration(gu, P0, gu.n, it, vcycle, stat, Cb))
                    return false;
                const i64 Cu = cc_split(gu, P0);  // union ranks [0, Cu)
                stat.cc_pieces = Cu - Cb;
                // Q_r of each replica with one copy's 2m_hat
                g0_.with_w([&](auto w) {
                    run_quality(g0_.indptr, g0_.indices, w, g0_.khat, P0, n, R,
                                Cu, two_m, gamma, L_.final_, L_.ctl, s_);
                });
                const Control c = readback();
                int best = 0;
                for (int r = 1; r < R; ++r)
                    if (c.q[r] > c.q[best]) best = r;  // ties -> lowest r
                C = compact_slice(P0 + static_cast<i64>(best) * n, n, Cu, P0);
            } else {
                i64 Cb = 0;
                if (!leiden_iteration(g0_, P0, C, it, vcycle, stat, Cb))
                    return false;
                C = cc_split(g0_, P0);
                stat.cc_pieces = C - Cb;
            }
            ++it;
            if (!final && T == -1) {  // §4.14: is the next iteration final?
                const double q = quality(P0, C).q[0];
                stop = !stat.moved || q - q_prev < kDeltaQStop ||
                       it == dp_.max_iter_until_stable;
                q_prev = q;
            }
            out.iters.push_back(std::move(stat));
            if (final) break;
        }
        launch_quality(P0, C);
        run_rank_labels(P0, n, C, labels_out, L_.final_, L_.cub, L_.cub_bytes,
                        L_.ctl, s_);
        const Control c = readback();
        out.modularity = c.q[0];
        out.n_clusters = C;
        return true;
    }

   private:
    Layout& L_;
    DriverParams dp_;
    cudaStream_t s_;
    QuantizeResult q_;
    LevelGraph g0_;
    i64 n_ = 0, nnz_ = 0;
    double lam_ = 0.0, gamma_ = 0.0;
    u32 seed_ = 0;
    u64 s64_ = 0;
    MoveLaunch ml_;                // sweep calls, reused by every LOCAL_MOVE
    GraphChain graphs_[3];         // one sweep graph per weight kind (WKind)
    GraphChain refine_graphs_[3];  // PLEIDEN_R R1 - R8, per weight kind
    GraphChain stage1_graph_;      // R9 + AGGREGATE stage 1
    GraphChain compact_graph_;     // COMPACT when refinement is not launched

    // Control readback (one sync) with the device invariant checks.
    Control readback() {
        const Control c = read_control(L_.ctl, s_, L_.pinned);
        if (c.flags & kFlagForestInvariant)
            throw std::runtime_error(
                "leiden: refinement forest invariant violated");
        if (c.flags & kFlagGatherInvariant)
            throw std::runtime_error("leiden: multi-pass gather failed");
        if (c.flags & kFlagBadLabel)
            throw std::runtime_error(
                "leiden: internal error: community id out of range");
        return c;
    }

    int* level_partition(int l) const {
        if (l == 0) return L_.persist.P0;
        return (l & 1) ? L_.persist.Pa : L_.persist.Pb;
    }

    LevelGraph union_level(int R) const {
        LevelGraph g;
        g.indptr = L_.uni.indptr;
        g.indices = L_.uni.indices;
        g.wf = q_.wkind == WKind::F32 ? L_.uni.wf32 : nullptr;
        g.wq = q_.wkind == WKind::I64 ? L_.uni.wq : nullptr;
        g.unit = g0_.unit;
        g.scale = g0_.scale;
        g.khat = L_.uni.khat;
        g.n = static_cast<i64>(R) * n_;
        g.nnz = static_cast<i64>(R) * nnz_;
        for (int c = 0; c < kNumClasses; ++c) g.cls[c] = R * g0_.cls[c];
        return g;
    }

    // LOCAL_MOVE + COMPACT on level l, launch only; a TOP phase launches its
    // first chunk and a gated COMPACT (continue_move runs the rest).
    MoveRun launch_move(const LevelGraph& g, int* P, int l, u32 phase, int cap,
                        u32 sweep0, i64 id_bound, u32 it,
                        std::vector<KernelCall>& compact) {
        MoveParams prm;
        prm.lam = lam_;
        prm.seed = seed_;
        prm.it = it;
        prm.level = static_cast<u32>(l);
        prm.phase = phase;
        prm.sweep0 = sweep0;
        prm.cap = cap;
        prm.S = dp_.subrounds;
        prm.th = dp_.th;
        prm.id_bound = id_bound;
        MoveRun mr;
        g.with_w([&](auto w) {
            constexpr WKind KW = MoveWKindOf<decltype(w)>::value;
            ml_.graph =
                dp_.graph_mode > 0 ? &graphs_[static_cast<int>(KW)] : nullptr;
            ml_.loop = dp_.graph_mode == 2 && graph_loop_supported();
            if (phase == kPhaseTop) {
                // replays would run no-op sweeps after the certifying one
                prm.first_chunk = ml_.graph && ml_.loop ? kSweepChunk : 1;
                prm.defer_rest = true;
            }
            mr = run_local_move(g.indptr, g.indices, w, g.khat, P, g.n, g.cls,
                                prm, L_.move, L_.ctl, ml_, L_.pinned, s_);
        });
        launch_compact(P, g.n, L_.move, L_.persist.KS, L_.cub, L_.cub_bytes,
                       L_.ctl, s_, &compact, mr.gate());
        return mr;
    }

    // The rest of a deferred TOP phase: its chunks, COMPACT, the SL1 readback.
    Control continue_move(const LevelGraph& g, int* P, MoveRun& mr) {
        continue_local_move(mr, ml_, L_.ctl, L_.pinned, s_);
        std::vector<KernelCall> compact;
        launch_compact(P, g.n, L_.move, L_.persist.KS, L_.cub, L_.cub_bytes,
                       L_.ctl, s_, &compact);
        launch_chain(compact, dp_.graph_mode > 0 ? &compact_graph_ : nullptr,
                     s_);
        const Control c = readback();  // SL1
        finish_local_move(mr, c);
        return c;
    }

    // PLEIDEN_R + AGGREGATE stage 1 of level l (C on the device; SL2 next).
    void launch_refine_begin(const LevelGraph& g, const int* P, int l, u32 it,
                             int* cmap, const AggregateOptions& opt,
                             AggregatePlan& plan,
                             std::vector<KernelCall> pre = {}) {
        RefineParams rp;
        rp.lam = lam_;
        rp.ctx_order = ctx(s64_, kTagOrder, it, static_cast<u32>(l), 0, 0);
        rp.trees = dp_.trees;
        g.with_w([&](auto w) {
            constexpr WKind KW = MoveWKindOf<decltype(w)>::value;
            GraphChain* rg = dp_.graph_mode > 0
                                 ? &refine_graphs_[static_cast<int>(KW)]
                                 : nullptr;
            std::vector<KernelCall> r9;  // runs with AGGREGATE stage 1
            run_refine(g.indptr, g.indices, w, g.khat, P, L_.persist.KS, g.n,
                       g.n, rp, L_.refine, cmap, L_.cub, L_.cub_bytes, L_.ctl,
                       s_, &L_.ctl->move_C, rg, &r9, std::move(pre));
            plan = aggregate_begin(g.indptr, g.indices, w, g.khat, g.n, g.nnz,
                                   cmap, opt, L_.aggregate, L_.arena, L_.cub,
                                   L_.cub_bytes, L_.ctl, s_, std::move(r9));
        });
    }

    i64 cc_split(const LevelGraph& g, int* P) {
        g.with_w([&](auto w) {
            run_cc_split(g.indptr, g.indices, w, P, g.n, L_.final_, L_.cub,
                         L_.cub_bytes, L_.ctl, s_);
        });
        return readback().n_components;
    }

    void launch_quality(const int* P, i64 C) {
        g0_.with_w([&](auto w) {
            run_quality(g0_.indptr, g0_.indices, w, g0_.khat, P, n_, 1, C,
                        q_.two_m_hat, gamma_, L_.final_, L_.ctl, s_);
        });
    }
    Control quality(const int* P, i64 C) {
        launch_quality(P, C);
        return readback();
    }

    // Ranks P_in[0, n) among the ids its slice uses into P_out; returns C.
    i64 compact_slice(const int* P_in, i64 n, i64 C, int* P_out) {
        const FinalBufs& fb = L_.final_;
        cuda_check(cudaMemsetAsync(fb.flags, 0, (C + 1) * sizeof(int), s_),
                   "leiden: clear slice flags");
        drv_slice_flags_kernel<<<grid_items(n), kBlock, 0, s_>>>(
            P_in, n, C, fb.flags, L_.ctl);
        CUDA_CHECK_LAST_ERROR(drv_slice_flags_kernel);
        std::size_t tb = L_.cub_bytes;
        cuda_check(cub::DeviceScan::ExclusiveSum(
                       L_.cub, tb, fb.flags, fb.rank,
                       cub_items(C + 1, "slice compact"), s_),
                   "leiden: slice compact scan");
        drv_slice_relabel_kernel<<<grid_items(n), kBlock, 0, s_>>>(
            P_in, n, C, fb.rank, P_out, L_.ctl);
        CUDA_CHECK_LAST_ERROR(drv_slice_relabel_kernel);
        return readback().n_components;
    }

    // One iteration on the level-0 graph g0 with partition P0 (ids < id_bound).
    // False on arena overflow; C_out: communities before the CC split.
    bool leiden_iteration(const LevelGraph& g0, int* P0, i64 id_bound, int it,
                          bool vcycle, IterationStat& stat, i64& C_out) {
        LevelArena& arena = L_.arena;
        arena.reset();
        std::vector<LevelGraph> G;
        G.reserve(static_cast<std::size_t>(dp_.max_levels) + 1);
        G.push_back(g0);
        const u32 uit = static_cast<u32>(it);
        bool moved = false;
        int lvl = 0;
        u32 phase = kPhaseDown;
        int reentry = 0;
        u32 top_done = 0;  // TOP sweeps run at this level (cumulative, §4.4)
        i64 idb = id_bound;
        AggregateOptions opt;
        opt.gather = dp_.gather;
        opt.move = dp_.th;
        opt.stage1 = dp_.graph_mode > 0 ? &stage1_graph_ : nullptr;
        while (true) {
            LevelGraph& g = G[lvl];
            int* P = level_partition(lvl);
            const bool top = phase == kPhaseTop;
            std::vector<KernelCall> compact;
            MoveRun mr = launch_move(g, P, lvl, phase,
                                     top ? dp_.top_sweeps : dp_.move_sweeps,
                                     top ? top_done : 0, idb, uit, compact);
            // Refinement and AGGREGATE stage 1 launch before the host knows C
            // (one readback delivers C and SL2), except where C == n is likely
            // (TOP, singleton levels); C == n or no merge restores the arena.
            const bool speculate = !top && !(lvl > 0 && idb == g.n);
            const std::size_t mark = arena.bottom;
            int* cmap =
                speculate
                    ? arena.push_bottom<int>(static_cast<std::size_t>(g.n))
                    : nullptr;
            AggregatePlan plan;
            if (cmap) {  // COMPACT runs in refinement's graph
                launch_refine_begin(g, P, lvl, uit, cmap, opt, plan,
                                    std::move(compact));
            } else {
                launch_chain(compact,
                             dp_.graph_mode > 0 ? &compact_graph_ : nullptr,
                             s_);
            }
            Control c = readback();  // SL1 (+ SL2)
            finish_local_move(mr, c);
            if (mr.more_sweeps()) {  // TOP only (never speculated)
                if (cmap)
                    throw std::logic_error(
                        "leiden: refinement speculated on a deferred move");
                c = continue_move(g, P, mr);
            }
            if (c.move_C < 0)
                throw std::logic_error("leiden: COMPACT gated off at SL1");
            const MoveStats& st = mr.st;
            const i64 C = c.move_C;
            if (top) {
                top_done += static_cast<u32>(st.sweeps);
                if (st.last > 0) stat.top_cap_hit = true;
            }
            moved = moved || st.moves > 0;
            idb = C;
            if (C == g.n) {
                if (cmap) {  // drop the speculative refinement / level
                    arena.release_top();
                    arena.bottom = mark;
                }
                if (!top) {  // all-active verification (§4.10)
                    phase = kPhaseTop;
                    continue;
                }
                ++stat.levels;  // certified gamma-separated top
                break;
            }
            if (!cmap) {  // refinement was not launched: launch it now
                cmap = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
                if (!cmap) return false;  // the arena cannot hold this level
                launch_refine_begin(g, P, lvl, uit, cmap, opt, plan);
                c = readback();  // SL2
            }
            if (c.n_next == g.n) {  // refinement merged nothing (C < n)
                arena.release_top();
                arena.bottom = mark;              // drop the speculative level
                if (reentry < dp_.max_reentry) {  // Lemma R (§4.10)
                    ++reentry;
                    stat.reentries += 1;
                    phase = kPhaseTop;
                    continue;
                }
                // last resort (never expected): the connected pieces of P
                stat.fallbacks += 1;
                stat.uncertified = true;
                cmap = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
                if (!cmap) return false;
                cuda_check(cudaMemcpyAsync(cmap, P, g.n * sizeof(int),
                                           cudaMemcpyDeviceToDevice, s_),
                           "leiden: fallback pieces");
                const i64 pieces = cc_split(g, cmap);
                if (pieces == g.n) {  // output the pieces (uncertified)
                    cuda_check(cudaMemcpyAsync(P, cmap, g.n * sizeof(int),
                                               cudaMemcpyDeviceToDevice, s_),
                               "leiden: fallback partition");
                    arena.bottom = mark;
                    idb = g.n;
                    ++stat.levels;
                    break;
                }
                agg_set_n_next_kernel<<<1, 1, 0, s_>>>(L_.ctl, pieces);
                CUDA_CHECK_LAST_ERROR(agg_set_n_next_kernel);
                g.with_w([&](auto w) {
                    plan = aggregate_begin(
                        g.indptr, g.indices, w, g.khat, g.n, g.nnz, cmap, opt,
                        L_.aggregate, arena, L_.cub, L_.cub_bytes, L_.ctl, s_);
                });
                c = readback();
            }
            const CoarseLevel lv = aggregate_finish(
                g.n, cmap, plan, c.n_next, c.nnz_next, L_.aggregate, arena, P,
                level_partition(lvl + 1), s_);
            if (!lv.ok) return false;
            g.cmap = cmap;
            ++stat.levels;
            LevelGraph ng;
            ng.indptr = lv.indptr;
            ng.indices = lv.indices;
            ng.wf = lv.weights;
            ng.khat = lv.khat;
            ng.n = lv.n;
            ng.nnz = lv.nnz;
            for (int k = 0; k < kNumClasses; ++k)
                ng.cls[k] = c.coarse_class_count[k];
            G.push_back(ng);  // (invalidates g)
            ++lvl;
            phase = kPhaseDown;
            reentry = 0;
            top_done = 0;
            if (lvl == dp_.max_levels) {  // guard; kNN graphs need 5-12
                ++stat.levels;
                break;
            }
        }
        // projection to level 0, with the V-cycle's UP sweeps
        i64 Ccur = idb;
        for (int lp = lvl - 1; lp >= 0; --lp) {
            const LevelGraph& g = G[lp];
            int* P = level_partition(lp);
            run_project(level_partition(lp + 1), g.cmap, g.n, P, s_);
            if (vcycle) {
                std::vector<KernelCall> compact;
                MoveRun mr = launch_move(g, P, lp, kPhaseUp, dp_.vcycle_sweeps,
                                         0, Ccur, uit, compact);
                launch_chain(compact,
                             dp_.graph_mode > 0 ? &compact_graph_ : nullptr,
                             s_);
                const Control c = readback();  // SL1
                finish_local_move(mr, c);
                Ccur = c.move_C;
                moved = moved || mr.st.moves > 0;
            }
        }
        stat.moved = moved;
        C_out = Ccur;
        return true;
    }
};

struct DriverInput {
    i64 n = 0, nnz = 0;
    bool use_weights = true;
    IngestInfo info;
    std::vector<double> gammas;
    std::vector<u32> seeds;
    const int* init = nullptr;  // compacted initial membership or null
    int* labels_out = nullptr;  // [n_res][n]
    void* workspace = nullptr;  // null: size the workspace only
    std::size_t workspace_bytes = 0;
    void* pinned = nullptr;
};

// The layout follows from the input; a call without a workspace sizes it.
// `quantize` runs I2 (+ I4).
template <typename Quantize>
DriverResult run_leiden(const DriverInput& in, const DriverParams& dp,
                        bool idx64_input, bool data_f32, bool host_input,
                        Quantize&& quantize, cudaStream_t s) {
    DriverResult out;
    LayoutParams p;
    p.n = in.n;
    p.nnz = in.nnz;
    p.wkind = call_wkind(in.info, in.use_weights, data_f32, in.gammas);
    p.idx64_input = idx64_input;
    p.host_input = host_input;
    const int R =
        n_replicas(in.info.n_counted, dp.max_replicas, dp.replica_nnz_budget);
    p.replicas = R;
    p.arena_factor = dp.arena_factor;
    p.subrounds = dp.subrounds;
    Layout L = make_layout(p, in.workspace);
    out.required_bytes = L.total;
    out.required_arena_factor = dp.arena_factor;
    if (L.total > in.workspace_bytes) {
        out.status = "workspace_too_small";
        return out;
    }
    L.pinned = in.pinned;
    LeidenDriver drv(L, dp, s);
    int s_prev = INT_MIN;
    for (std::size_t i = 0; i < in.gammas.size(); ++i) {
        const double gamma = in.gammas[i];
        int* labels = in.labels_out + static_cast<i64>(i) * in.n;
        ResolutionResult r;
        if (in.info.n_counted == 0) {  // no counted entry: trivial result
            r.n_clusters = in.n;
            drv_iota_kernel<<<grid_items(in.n), kBlock, 0, s>>>(labels, in.n);
            CUDA_CHECK_LAST_ERROR(drv_iota_kernel);
            out.res.push_back(std::move(r));
            continue;
        }
        const int sg =
            scale_for_gamma(scale_s0(in.info.n_counted, in.info.wmax), gamma);
        if (sg != s_prev) {  // I2 (+ I4) for the new scale (§4.2, RA7.4)
            drv.set_level0(quantize(gamma, L), in.n, in.nnz);
            s_prev = sg;
        }
        if (!drv.run_resolution(gamma, in.seeds[i], in.init, R, labels, r)) {
            out.status = "workspace_too_small";
            out.required_arena_factor =
                arena_required_factor(p, L.arena.required);
            return out;
        }
        out.res.push_back(std::move(r));
    }
    return out;
}

template <typename T, typename Device>
using optional_array = std::optional<gpu_array_c<T, Device>>;

// One `leiden` call with driver parameters `dp` (also the dev knob binding).
template <typename IP, typename IX, typename WI, typename InDevice,
          typename Device>
DriverResult call_leiden(const input_array<const IP, InDevice>& indptr,
                         const input_array<const IX, InDevice>& indices,
                         const input_array<const WI, InDevice>& data,
                         bool use_weights, const nb::dict& check,
                         optional_array<std::uint8_t, Device>& workspace,
                         std::uintptr_t pinned,
                         optional_array<int, Device>& labels_out,
                         const std::vector<double>& resolutions,
                         const std::vector<std::uint32_t>& seeds,
                         const optional_array<const int, Device>& init,
                         const DriverParams& dp, std::uintptr_t stream) {
    if (indptr.ndim() != 1 || indptr.size() < 1)
        throw std::invalid_argument("indptr must be 1-D with n + 1 entries");
    const i64 n = static_cast<i64>(indptr.size()) - 1;
    const i64 nnz = static_cast<i64>(indices.size());
    if (static_cast<i64>(data.size()) != nnz)
        throw std::invalid_argument("data and indices must have equal length");
    if (n >= kMaxVertices)
        throw std::invalid_argument("leiden: n must be < 2^30");
    if constexpr (host_input_v<InDevice>) require_pageable_access();
    const std::size_t n_res = resolutions.size();
    if (n_res == 0 || seeds.size() != n_res)
        throw std::invalid_argument(
            "one seed per resolution (at least one) is required");
    for (double g : resolutions) check_gamma(g);
    if (workspace.has_value() != labels_out.has_value())
        throw std::invalid_argument("workspace and labels_out go together");
    if (labels_out &&
        (labels_out->ndim() != 2 || labels_out->shape(0) != n_res ||
         static_cast<i64>(labels_out->shape(1)) != n))
        throw std::invalid_argument(
            "labels_out must have shape (len(resolutions), n)");
    if (init && static_cast<i64>(init->size()) != n)
        throw std::invalid_argument("initial_membership must have n entries");
    if (!(dp.n_iterations == -1 ||
          (dp.n_iterations >= 1 && dp.n_iterations < (1 << 16))))
        throw std::invalid_argument(
            "n_iterations must be a positive integer or -1");
    if (!(dp.arena_factor > 0.0))
        throw std::invalid_argument("arena_factor must be > 0");
    DriverInput in;
    in.n = n;
    in.nnz = nnz;
    in.use_weights = use_weights;
    in.info = info_from_dict(check);
    in.gammas = resolutions;
    in.seeds.assign(seeds.begin(), seeds.end());
    in.init = init ? init->data() : nullptr;
    in.labels_out = labels_out ? labels_out->data() : nullptr;
    in.workspace = workspace ? workspace->data() : nullptr;
    in.workspace_bytes = workspace ? workspace->size() : 0;
    in.pinned = reinterpret_cast<void*>(pinned);
    const IP* ip = indptr.data();
    const IX* ix = indices.data();
    const WI* wd = data.data();
    nb::gil_scoped_release release;
    const auto s = (cudaStream_t)stream;
    return run_synced(s, [&] {
        auto quantize = [&](double gamma, Layout& L) {
            return run_quantize<IP, IX, WI>(ip, ix, wd, n, nnz, use_weights,
                                            gamma, in.info, L, dp.th, s);
        };
        DriverResult r = run_leiden(in, dp, !std::is_same_v<IX, int>,
                                    std::is_same_v<WI, float>,
                                    host_input_v<InDevice>, quantize, s);
        cuda_check(cudaStreamSynchronize(s), "leiden: final sync");
        return r;
    });
}

inline nb::dict result_dict(const DriverResult& r) {
    nb::dict d;
    d["status"] = r.status;
    if (r.status != "ok") {
        d["required_bytes"] = r.required_bytes;
        d["required_arena_factor"] = r.required_arena_factor;
        return d;
    }
    nb::list q, k;
    for (const ResolutionResult& x : r.res) {
        q.append(x.modularity);
        k.append(x.n_clusters);
    }
    d["modularity"] = q;
    d["n_clusters"] = k;
    return d;
}

// Every resolution of the call. `check`: ingest_check's result for exactly
// these arrays; `labels_out`: (len(resolutions), n) int32. Without `workspace`
// the call only sizes it. Status "ok" or "workspace_too_small" (arena).
template <typename IP, typename IX, typename WI, typename InDevice,
          typename Device>
void register_driver_typed(nb::module_& m) {
    m.def(
        "leiden",
        [](input_array<const IP, InDevice> indptr,
           input_array<const IX, InDevice> indices,
           input_array<const WI, InDevice> data, bool use_weights,
           nb::dict check, optional_array<std::uint8_t, Device> workspace,
           std::uintptr_t pinned, optional_array<int, Device> labels_out,
           std::vector<double> resolutions, std::vector<std::uint32_t> seeds,
           int n_iterations,
           optional_array<const int, Device> initial_membership,
           double arena_factor, std::uintptr_t stream) {
            DriverParams dp;
            dp.n_iterations = n_iterations;
            dp.arena_factor = arena_factor;
            return result_dict(call_leiden<IP, IX, WI, InDevice, Device>(
                indptr, indices, data, use_weights, check, workspace, pinned,
                labels_out, resolutions, seeds, initial_membership, dp,
                stream));
        },
        "indptr"_a.noconvert(), nb::kw_only(), "indices"_a.noconvert(),
        "data"_a.noconvert(), "use_weights"_a, "check"_a,
        "workspace"_a.noconvert().none() = nb::none(), "pinned"_a = 0,
        "labels_out"_a.noconvert().none() = nb::none(), "resolutions"_a,
        "seeds"_a, "n_iterations"_a,
        "initial_membership"_a.noconvert().none() = nb::none(),
        "arena_factor"_a = 1.9, "stream"_a = 0);
}

template <typename InDevice, typename Device>
void register_driver_inputs(nb::module_& m) {
    register_driver_typed<int, int, float, InDevice, Device>(m);
    register_driver_typed<int, int, double, InDevice, Device>(m);
    register_driver_typed<long long, int, float, InDevice, Device>(m);
    register_driver_typed<long long, int, double, InDevice, Device>(m);
    register_driver_typed<int, long long, float, InDevice, Device>(m);
    register_driver_typed<int, long long, double, InDevice, Device>(m);
    register_driver_typed<long long, long long, float, InDevice, Device>(m);
    register_driver_typed<long long, long long, double, InDevice, Device>(m);
}

// Device input first (overloads are tried in order), then host input.
template <typename Device>
void register_driver_bindings(nb::module_& m) {
    register_driver_inputs<Device, Device>(m);
    register_driver_inputs<nb::device::cpu, Device>(m);
}

inline void register_api(nb::module_& m) {
    register_ingest(m);

    m.attr("PINNED_BYTES") = kPinnedBytes;

    REGISTER_GPU_BINDINGS(register_driver_bindings, m);
}

}  // namespace leiden
