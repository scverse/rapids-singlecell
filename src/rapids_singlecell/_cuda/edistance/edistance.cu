#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_edistance.cuh"

using namespace nb::literals;

template <typename T, typename Device>
using arr = gpu_array_c<T, Device>;
// A policy's smem element as nanobind sees it (fp16 bits as uint16)
template <class M>
using elem_t =
    std::conditional_t<sizeof(typename M::E) == 2, uint16_t, typename M::E>;

// The loaders' conversion type
template <class M, typename In>
using W = std::common_type_t<In, typename M::Acc>;

template <class K>
static void set_smem(K* kernel, int bytes) {
    cuda_check(cudaFuncSetAttribute(
                   kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
               "cudaFuncSetAttribute");
}

// Dynamic shared memory: A rows stay resident over all feature windows only
// when that keeps the blocks per SM of the launch that streams them with B.
struct EdPlan {
    bool a_resident;
    int smem, streamed;
};
template <class M>
static EdPlan plan(int kp) {
    constexpr int TA = M::TA, KW = ED_WB / sizeof(typename M::E);
    const int streamed = (TA + ED_TB) * ED_PITCH;
    const long long nwin = (kp + KW - 1) / KW;
    const long long resident = TA * (nwin * ED_WB + 16) + ED_TB * ED_PITCH;
    const int fixed = (TA + ED_TB) * (4 + (int)sizeof(typename M::Acc)) +
                      WARP_SIZE * 8 + 1024;  // static + reserved
    const int opt_in =
        device_attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    const int per_sm =
        device_attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
    auto blocks = [&](long long dyn) {
        return dyn + fixed > opt_in + 1024
                   ? 0LL
                   : std::min<long long>(M::min_blocks, per_sm / (dyn + fixed));
    };
    const bool a_resident = blocks(resident) == blocks(streamed);
    return {a_resident, a_resident ? (int)resident : streamed, streamed};
}

#define ED_LAUNCH_ARGS                                                         \
    arr<const int, Device> slots, int kp, arr<typename M::Acc, Device> norms,  \
        arr<const int, Device> cat_offsets,                                    \
        arr<const int, Device> cell_indices, arr<const int, Device> pair_left, \
        arr<const int, Device> pair_right,                                     \
        arr<const int, Device> block_offsets, int grid,                        \
        arr<double, Device> pairwise_sums, bool norms_only,                    \
        std::uintptr_t stream

// norms_only: norms[row] of the rows cell_indices lists, of every row without
// it, on grid = block_offsets[splits] blocks. Otherwise the pair sums of one
// chunk of pairs on grid = block_offsets[num_pairs] blocks. slots are the
// first cells of the splits (one per row of nm) and the number of cells.
template <class M, class L, typename Device>
static void launch(const L& ld, size_t centers, ED_LAUNCH_ARGS) {
    nb_require(centers > 0 && slots.size() == centers + 1,
               "edistance: nm, sc and slots do not match");
    const int n_splits = static_cast<int>(centers);
    const EdPlan p = plan<M>(kp);
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (grid == 0) return;
    if (norms_only) {
        nb_require(block_offsets.size() == centers + 1,
                   "edistance: block_offsets and slots do not match");
        set_smem(edistance_norms_kernel<M, L>, p.streamed);
        edistance_norms_kernel<M, L><<<grid, ED_THREADS, p.streamed, s>>>(
            ld, cell_indices.is_valid() ? cell_indices.data() : nullptr,
            slots.data(), block_offsets.data(), n_splits, kp, norms.data());
        CUDA_CHECK_LAST_ERROR(edistance_norms_kernel);
        return;
    }
    set_smem(edistance_kernel<M, L>, p.smem);
    edistance_kernel<M, L><<<grid, ED_THREADS, p.smem, s>>>(
        ld, slots.data(), n_splits, norms.data(), cat_offsets.data(),
        cell_indices.data(), pair_left.data(), pair_right.data(),
        block_offsets.data(), static_cast<int>(pair_left.size()), kp,
        p.a_resident, pairwise_sums.data());
    CUDA_CHECK_LAST_ERROR(edistance_kernel);
}

#define ED_LAUNCH_PASS                                                  \
    slots, kp, norms, cat_offsets, cell_indices, pair_left, pair_right, \
        block_offsets, grid, pairwise_sums, norms_only, stream
#define ED_LAUNCH_NAMES                                                   \
    "slots"_a.noconvert(), "kp"_a, "norms"_a.noconvert(),                 \
        "cat_offsets"_a.noconvert(), "cell_indices"_a.noconvert().none(), \
        "pair_left"_a.noconvert(), "pair_right"_a.noconvert(),            \
        "block_offsets"_a.noconvert(), "grid"_a,                          \
        "pairwise_sums"_a.noconvert(), "norms_only"_a, "stream"_a = 0

template <class M, typename In, typename IndptrT, typename Device>
static void def_sparse(nb::module_& m, const char* name) {
    m.def(
        name,
        [](arr<const IndptrT, Device> indptr, arr<const int, Device> indices,
           arr<const In, Device> data, arr<const W<M, In>, Device> nm,
           arr<const W<M, In>, Device> sc, arr<const elem_t<M>, Device> zero,
           bool duplicates, ED_LAUNCH_ARGS) {
            nb_require(nm.ndim() == 2 && nm.shape(0) == sc.size() &&
                           zero.size() == nm.size(),
                       "edistance: nm, sc and zero do not match");
            const EdCsrRows<M, In, IndptrT> ld{
                data.data(),
                indices.data(),
                indptr.data(),
                nm.data(),
                sc.data(),
                reinterpret_cast<const uint4*>(zero.data()),
                static_cast<int>(nm.shape(1)),
                duplicates};
            launch<M>(ld, sc.size(), ED_LAUNCH_PASS);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "nm"_a.noconvert(), "sc"_a.noconvert(), "zero"_a.noconvert(),
        "duplicates"_a, ED_LAUNCH_NAMES);
}

// One entry point per math policy, overloaded on the layout and dtypes: dense
// float64 / float32 rows, CSR with int32 / int64 indptr. f64 before f32. nm
// has a row and sc a value per split.
template <class M, typename Device>
static void def_math(nb::module_& m, const char* name) {
    auto dense = [&](auto tag) {
        using In = decltype(tag);
        m.def(
            name,
            [](arr<const In, Device> X, int align,
               arr<const W<M, In>, Device> nm, arr<const W<M, In>, Device> sc,
               ED_LAUNCH_ARGS) {
                nb_require(nm.size() == sc.size() * X.shape(1),
                           "edistance: nm and sc do not match X");
                const EdDenseRows<M, In> ld{X.data(),
                                            static_cast<long long>(X.shape(1)),
                                            static_cast<int>(X.shape(1)),
                                            align,
                                            nm.data(),
                                            sc.data()};
                launch<M>(ld, sc.size(), ED_LAUNCH_PASS);
            },
            "X"_a.noconvert(), "align"_a, "nm"_a.noconvert(),
            "sc"_a.noconvert(), ED_LAUNCH_NAMES);
    };
    dense(double{});
    dense(float{});
    def_sparse<M, double, int, Device>(m, name);
    def_sparse<M, float, int, Device>(m, name);
    def_sparse<M, double, long long, Device>(m, name);
    def_sparse<M, float, long long, Device>(m, name);
}

#define ED_CENTER_ARGS                                                      \
    arr<const int, Device> sample, int rows_read, int pass,                 \
        arr<In, Device> out, arr<double, Device> sums, arr<In, Device> top, \
        std::uintptr_t stream
#define ED_CENTER_NAMES                                                   \
    "sample"_a.noconvert(), "rows_read"_a, "pass"_a, "out"_a.noconvert(), \
        "sums"_a.noconvert(), "top"_a.noconvert(), "stream"_a = 0

// One pass of the center kernel over the sampled rows i < out.size() of the
// splits (the rows of sums), in windows of R::CW columns
template <typename In, class R, typename Device>
static void center(const R& rd, ED_CENTER_ARGS) {
    const int m = pass == 3 ? rows_read : static_cast<int>(out.size());
    nb_require(sums.ndim() == 2 && sample.size() == 3 * (sums.shape(0) + 1) &&
                   top.size() == sums.shape(0),
               "center: sample / top do not match sums");
    const int d = static_cast<int>(sums.shape(1));
    if (m == 0 || d == 0) return;
    if (pass == 1) {
        const int n = static_cast<int>(sums.shape(0));
        const int warps = n * ((d - 1) / 8 + 1);
        constexpr int per_block = ED_THREADS / WARP_SIZE;
        edistance_mean_kernel<In, R>
            <<<(warps + per_block - 1) / per_block, ED_THREADS, 0,
               reinterpret_cast<cudaStream_t>(stream)>>>(
                rd, sample.data(), n, d, out.data(), sums.data());
        CUDA_CHECK_LAST_ERROR(edistance_mean_kernel);
        return;
    }
    const long long units = (long long)m * ((d - 1) / R::CW + 1);
    edistance_center_kernel<In, R>
        <<<(unsigned)((units + ED_THREADS - 1) / ED_THREADS), ED_THREADS, 0,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            rd, sample.data(), static_cast<int>(sums.shape(0)), m, d, R::CW,
            pass, out.data(), sums.data(), top.data());
    CUDA_CHECK_LAST_ERROR(edistance_center_kernel);
}

// Dense rows (those of the index array rows, if given) or sorted CSR with
// int32 / int64 indptr and int32 / int64 indices; and the trim of the sampled
// rows between pass 0 and pass 1
template <typename In, typename Device>
static void def_center(nb::module_& m) {
    m.def(
        "center",
        [](arr<const In, Device> X, arr<const int, Device> rows,
           ED_CENTER_ARGS) {
            center<In>(
                EdCenterDense<In>{X.data(), (long long)X.shape(1), rows.data()},
                sample, rows_read, pass, out, sums, top, stream);
        },
        "X"_a.noconvert(), "rows"_a.noconvert().none(), ED_CENTER_NAMES);
    auto csr = [&](auto ptag, auto itag) {
        using P = decltype(ptag);
        using I = decltype(itag);
        m.def(
            "center",
            [](arr<const P, Device> indptr, arr<const I, Device> indices,
               arr<const In, Device> data, ED_CENTER_ARGS) {
                center<In>(EdCenterCsr<In, P, I>{data.data(), indices.data(),
                                                 indptr.data()},
                           sample, rows_read, pass, out, sums, top, stream);
            },
            "indptr"_a.noconvert(), "indices"_a.noconvert(),
            "data"_a.noconvert(), ED_CENTER_NAMES);
    };
    csr(0, 0);
    csr(0LL, 0);
    csr(0, 0LL);
    csr(0LL, 0LL);
    m.def(
        "trim",
        [](arr<const int, Device> first, In trim, arr<In, Device> out,
           arr<In, Device> top, std::uintptr_t stream) {
            nb_require(first.size() == top.size() + 1,
                       "trim: first and top do not match");
            if (top.size() == 0) return;
            edistance_trim_kernel<In>
                <<<(unsigned)top.size(), ED_THREADS, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
                    first.data(), trim, out.data(), top.data());
            CUDA_CHECK_LAST_ERROR(edistance_trim_kernel);
        },
        "first"_a.noconvert(), "trim"_a, "out"_a.noconvert(),
        "top"_a.noconvert(), "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_center<double, Device>(m);
    def_center<float, Device>(m);
    def_math<EdFma<double>, Device>(m, "pair_sums_f64");
    def_math<EdFma<float>, Device>(m, "pair_sums_f32");
    def_math<EdMma16, Device>(m, "pair_sums_f16");
}

NB_MODULE(_edistance_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
