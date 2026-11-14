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
    int kp, arr<typename M::Acc, Device> norms,                                \
        arr<const int, Device> cat_offsets,                                    \
        arr<const int, Device> cell_indices, arr<const int, Device> pair_left, \
        arr<const int, Device> pair_right,                                     \
        arr<const int, Device> block_offsets, int grid,                        \
        arr<double, Device> pairwise_sums, bool norms_only,                    \
        std::uintptr_t stream

// norms_only: norms[row] of every row. Otherwise the pair sums of one chunk of
// pairs on grid = block_offsets[num_pairs] blocks.
template <class M, class L, typename Device>
static void launch(const L& ld, ED_LAUNCH_ARGS) {
    constexpr int TA = M::TA;
    const EdPlan p = plan<M>(kp);
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (norms_only) {
        const int n = static_cast<int>(norms.size());
        if (n == 0) return;
        set_smem(edistance_norms_kernel<M, L>, p.streamed);
        edistance_norms_kernel<M, L>
            <<<(n + TA - 1) / TA, ED_THREADS, p.streamed, s>>>(ld, n, kp,
                                                               norms.data());
        CUDA_CHECK_LAST_ERROR(edistance_norms_kernel);
        return;
    }
    if (grid == 0) return;
    set_smem(edistance_kernel<M, L>, p.smem);
    edistance_kernel<M, L><<<grid, ED_THREADS, p.smem, s>>>(
        ld, norms.data(), cat_offsets.data(), cell_indices.data(),
        pair_left.data(), pair_right.data(), block_offsets.data(),
        static_cast<int>(pair_left.size()), kp, p.a_resident,
        pairwise_sums.data());
    CUDA_CHECK_LAST_ERROR(edistance_kernel);
}

#define ED_LAUNCH_PASS                                           \
    kp, norms, cat_offsets, cell_indices, pair_left, pair_right, \
        block_offsets, grid, pairwise_sums, norms_only, stream
#define ED_LAUNCH_NAMES                                                      \
    "kp"_a, "norms"_a.noconvert(), "cat_offsets"_a.noconvert(),              \
        "cell_indices"_a.noconvert(), "pair_left"_a.noconvert(),             \
        "pair_right"_a.noconvert(), "block_offsets"_a.noconvert(), "grid"_a, \
        "pairwise_sums"_a.noconvert(), "norms_only"_a, "stream"_a = 0

template <class M, typename In, typename IndptrT, typename Device>
static void def_sparse(nb::module_& m, const char* name) {
    m.def(
        name,
        [](arr<const IndptrT, Device> indptr, arr<const int, Device> indices,
           arr<const In, Device> data, arr<const W<M, In>, Device> nm,
           W<M, In> sc, arr<const elem_t<M>, Device> zero, bool duplicates,
           ED_LAUNCH_ARGS) {
            const EdCsrRows<M, In, IndptrT> ld{
                data.data(),
                indices.data(),
                indptr.data(),
                nm.data(),
                sc,
                reinterpret_cast<const uint4*>(zero.data()),
                duplicates};
            launch<M>(ld, ED_LAUNCH_PASS);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "nm"_a.noconvert(), "sc"_a, "zero"_a.noconvert(), "duplicates"_a,
        ED_LAUNCH_NAMES);
}

// One entry point per math policy, overloaded on the layout and dtypes: dense
// float64 / float32 rows, CSR with int32 / int64 indptr. f64 before f32.
template <class M, typename Device>
static void def_math(nb::module_& m, const char* name) {
    auto dense = [&](auto tag) {
        using In = decltype(tag);
        m.def(
            name,
            [](arr<const In, Device> X, int align,
               arr<const W<M, In>, Device> nm, W<M, In> sc, ED_LAUNCH_ARGS) {
                const EdDenseRows<M, In> ld{X.data(),
                                            static_cast<long long>(X.shape(1)),
                                            static_cast<int>(X.shape(1)),
                                            align,
                                            nm.data(),
                                            sc};
                launch<M>(ld, ED_LAUNCH_PASS);
            },
            "X"_a.noconvert(), "align"_a, "nm"_a.noconvert(), "sc"_a,
            ED_LAUNCH_NAMES);
    };
    dense(double{});
    dense(float{});
    def_sparse<M, double, int, Device>(m, name);
    def_sparse<M, float, int, Device>(m, name);
    def_sparse<M, double, long long, Device>(m, name);
    def_sparse<M, float, long long, Device>(m, name);
}

#define ED_CENTER_ARGS                                                      \
    long long step, int pass, int shift, arr<const bool, Device> keep,      \
        arr<In, Device> ext, arr<In, Device> out, arr<double, Device> sums, \
        std::uintptr_t stream
#define ED_CENTER_NAMES                                                       \
    "step"_a, "pass"_a, "shift"_a, "keep"_a.noconvert(), "ext"_a.noconvert(), \
        "out"_a.noconvert(), "sums"_a.noconvert(), "stream"_a = 0

// One pass of the center kernel over the rows i * step, i < out.size(), in
// windows of R::CW columns
template <typename In, class R, typename Device>
static void center(const R& rd, ED_CENTER_ARGS) {
    const int m = static_cast<int>(out.size());
    const int d = static_cast<int>(ext.size() / 2);
    nb_require(!pass || (keep.size() == out.size() && sums.size() == (size_t)d),
               "center: keep / sums do not match out / ext");
    if (m == 0 || d == 0) return;
    const long long units = (long long)m * ((d - 1) / R::CW + 1);
    edistance_center_kernel<In, R>
        <<<(unsigned)((units + ED_THREADS - 1) / ED_THREADS), ED_THREADS, 0,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            rd, step, m, d, R::CW, pass, shift, keep.data(), ext.data(),
            out.data(), sums.data());
    CUDA_CHECK_LAST_ERROR(edistance_center_kernel);
}

// Dense rows (those of the index array rows, if given), the column extremes
// less mu, or sorted CSR with int32 / int64 indptr and int32 / int64 indices
template <typename In, typename Device>
static void def_center(nb::module_& m) {
    m.def(
        "center",
        [](arr<const In, Device> X, arr<const int, Device> rows,
           ED_CENTER_ARGS) {
            center<In>(
                EdCenterDense<In>{X.data(), (long long)X.shape(1), rows.data()},
                step, pass, shift, keep, ext, out, sums, stream);
        },
        "X"_a.noconvert(), "rows"_a.noconvert().none(), ED_CENTER_NAMES);
    m.def(
        "center",
        [](arr<const In, Device> keys, arr<const In, Device> mu,
           ED_CENTER_ARGS) {
            center<In>(
                EdCenterReach<In>{keys.data(), mu.data(), (long long)mu.size()},
                step, pass, shift, keep, ext, out, sums, stream);
        },
        "keys"_a.noconvert(), "mu"_a.noconvert(), ED_CENTER_NAMES);
    auto csr = [&](auto ptag, auto itag) {
        using P = decltype(ptag);
        using I = decltype(itag);
        m.def(
            "center",
            [](arr<const P, Device> indptr, arr<const I, Device> indices,
               arr<const In, Device> data, ED_CENTER_ARGS) {
                center<In>(EdCenterCsr<In, P, I>{data.data(), indices.data(),
                                                 indptr.data()},
                           step, pass, shift, keep, ext, out, sums, stream);
            },
            "indptr"_a.noconvert(), "indices"_a.noconvert(),
            "data"_a.noconvert(), ED_CENTER_NAMES);
    };
    csr(0, 0);
    csr(0LL, 0);
    csr(0, 0LL);
    csr(0LL, 0LL);
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
