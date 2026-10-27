#include <cuda_runtime.h>
#include <cstdint>
#include "../nb_types.h"

#include "kernels_sepal.cuh"

using namespace nb::literals;

struct SepalLaunch {
    void* kernel;
    size_t smem;
    int chunk;
};

// The kernel instance and its shared memory for a configuration.
template <typename V>
static SepalLaunch sepal_launch(int max_neighs, int store, int cluster_size,
                                int n_cells, int block_size) {
    const int chunk = (n_cells + cluster_size - 1) / cluster_size;
    size_t smem = sepal_scratch_bytes(block_size / 32);
    if (store == SEPAL_SMEM2) smem += 2 * chunk * sizeof(V);
    if (store == SEPAL_SMEM1) smem += chunk * sizeof(V);
    const bool cluster = cluster_size > 1;
    void* k = nullptr;
#define SEPAL_PICK(K)                                      \
    if (cluster && store == SEPAL_SMEM2)                   \
        k = (void*)sepal_kernel<V, K, true, SEPAL_SMEM2>;  \
    else if (cluster && store == SEPAL_SMEM1)              \
        k = (void*)sepal_kernel<V, K, true, SEPAL_SMEM1>;  \
    else if (store == SEPAL_SMEM2)                         \
        k = (void*)sepal_kernel<V, K, false, SEPAL_SMEM2>; \
    else if (store == SEPAL_SMEM1)                         \
        k = (void*)sepal_kernel<V, K, false, SEPAL_SMEM1>; \
    else if (!cluster)                                     \
        k = (void*)sepal_kernel<V, K, false, SEPAL_GLOBAL>;
    if (max_neighs == 4) {
        SEPAL_PICK(4)
    } else {
        SEPAL_PICK(6)
    }
#undef SEPAL_PICK
    return {k, smem, chunk};
}

static cudaLaunchConfig_t sepal_config(const SepalLaunch& l, int n_groups,
                                       int cluster_size, int block_size,
                                       cudaStream_t stream,
                                       cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_groups * cluster_size);
    cfg.blockDim = dim3(block_size);
    cfg.dynamicSmemBytes = l.smem;
    cfg.stream = stream;
    if (cluster_size > 1) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cluster_size;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    return cfg;
}

static bool sepal_prepare(const SepalLaunch& l, int cluster_size) {
    if (l.kernel == nullptr) return false;
    if (cudaFuncSetAttribute(l.kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(l.smem)) != cudaSuccess)
        return false;
    if (cluster_size > 8 &&
        cudaFuncSetAttribute(l.kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed,
                             1) != cudaSuccess)
        return false;
    return true;
}

// Groups (genes in flight) that can be resident at once; 0 if the
// configuration cannot run.
static int sepal_occupancy(bool f32, int max_neighs, int store,
                           int cluster_size, int n_cells, int block_size) {
    const SepalLaunch l =
        f32 ? sepal_launch<float>(max_neighs, store, cluster_size, n_cells,
                                  block_size)
            : sepal_launch<double>(max_neighs, store, cluster_size, n_cells,
                                   block_size);
    int n = 0;
    if (!sepal_prepare(l, cluster_size)) {
        cudaGetLastError();
        return 0;
    }
    if (cluster_size > 1) {
        cudaLaunchAttribute attr[1];
        cudaLaunchConfig_t cfg =
            sepal_config(l, 1, cluster_size, block_size, 0, attr);
        if (cudaOccupancyMaxActiveClusters(&n, l.kernel, &cfg) != cudaSuccess)
            n = 0;
    } else {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &n, l.kernel, block_size, l.smem) != cudaSuccess)
            n = 0;
        n *= sms;
    }
    cudaGetLastError();
    return n;
}

template <typename V, typename Device>
void def_diffusion(nb::module_& m) {
    // conc: (n_genes, n_cells); buf: (n_groups, 2, n_cells) global state or
    // staging; lo: (n_groups, n_cells) float32 compensation
    m.def(
        "diffusion",
        [](gpu_array_c<const V, Device> conc, gpu_array_c<V, Device> buf,
           gpu_array_c<V, Device> lo, gpu_array_c<const int, Device> nbrs,
           gpu_array_c<const int, Device> ctr, gpu_array_c<int, Device> results,
           gpu_array_c<int, Device> counter, int n_sat, int max_neighs,
           int n_iter, double dt, double thresh, int store, int cluster_size,
           int block_size, std::uintptr_t stream) {
            nb_require(conc.ndim() == 2,
                       "sepal: conc must be (n_genes, n_cells)");
            const int n_genes = static_cast<int>(conc.shape(0));
            const int n_cells = static_cast<int>(conc.shape(1));
            const int n_groups = static_cast<int>(buf.shape(0));
            nb_require(max_neighs == 4 || max_neighs == 6,
                       "sepal: max_neighs must be 4 or 6");
            nb_require(nbrs.size() == (size_t)n_cells * max_neighs &&
                           ctr.shape(0) == (size_t)n_cells,
                       "sepal: nbrs and ctr must cover every cell");
            // unused scratch may be empty
            nb_require(
                buf.ndim() == 3 && buf.shape(1) == 2 &&
                    (buf.shape(2) == (size_t)n_cells || buf.shape(2) == 0) &&
                    lo.ndim() == 2 && lo.shape(0) == (size_t)n_groups &&
                    (lo.shape(1) == (size_t)n_cells || lo.shape(1) == 0),
                "sepal: buf (n_groups, 2, n_cells), lo (n_groups, n_cells)");
            nb_require(
                results.shape(0) == (size_t)n_genes && counter.size() == 1,
                "sepal: results must have n_genes entries");
            if (n_genes == 0) return;
            const SepalLaunch l = sepal_launch<V>(
                max_neighs, store, cluster_size, n_cells, block_size);
            nb_require(sepal_prepare(l, cluster_size),
                       "sepal: unsupported configuration");
            cudaLaunchAttribute attr[1];
            cudaLaunchConfig_t cfg =
                sepal_config(l, n_groups, cluster_size, block_size,
                             (cudaStream_t)stream, attr);
            const V* conc_p = conc.data();
            V *buf_p = buf.data(), *lo_p = lo.data();
            const int *nbrs_p = nbrs.data(), *ctr_p = ctr.data();
            int *res_p = results.data(), *cnt_p = counter.data();
            int n_g = n_genes, n_c = n_cells, chunk = l.chunk;
            void* args[] = {&conc_p, &buf_p,  &lo_p, &nbrs_p, &ctr_p,
                            &res_p,  &cnt_p,  &n_g,  &n_c,    &chunk,
                            &n_sat,  &n_iter, &dt,   &thresh};
            cuda_check(cudaLaunchKernelExC(&cfg, l.kernel, args),
                       "sepal_kernel launch");
        },
        "conc"_a, nb::kw_only(), "buf"_a, "lo"_a, "nbrs"_a, "ctr"_a,
        "results"_a, "counter"_a, "n_sat"_a, "max_neighs"_a, "n_iter"_a, "dt"_a,
        "thresh"_a, "store"_a, "cluster_size"_a, "block_size"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_diffusion<double, Device>(m);
    def_diffusion<float, Device>(m);

    m.def(
        "first_sat_neighbor",
        [](gpu_array_c<const int, Device> unsat,
           gpu_array_c<const int, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const bool, Device> sat_mask,
           gpu_array_c<int, Device> nearest, std::uintptr_t stream) {
            const int n = static_cast<int>(unsat.shape(0));
            nb_require(nearest.shape(0) == (size_t)n,
                       "sepal: nearest must match unsat");
            if (n == 0) return;
            sepal_first_sat_neighbor_kernel<<<(n + 255) / 256, 256, 0,
                                              (cudaStream_t)stream>>>(
                unsat.data(), indptr.data(), indices.data(), sat_mask.data(),
                nearest.data(), n);
            CUDA_CHECK_LAST_ERROR(sepal_first_sat_neighbor_kernel);
        },
        "unsat"_a, "indptr"_a, "indices"_a, "sat_mask"_a, nb::kw_only(),
        "nearest"_a, "stream"_a = 0);

    m.def(
        "nearest_sat_l1",
        [](gpu_array_c<const double, Device> spatial,
           gpu_array_c<const int, Device> query,
           gpu_array_c<const int, Device> sat, gpu_array_c<int, Device> nearest,
           std::uintptr_t stream) {
            const int n = static_cast<int>(query.shape(0));
            const int n_sat = static_cast<int>(sat.shape(0));
            nb_require(spatial.ndim() == 2 && spatial.shape(1) == 2,
                       "sepal: spatial must have shape (n_cells, 2)");
            nb_require(nearest.shape(0) == (size_t)n && n_sat > 0,
                       "sepal: nearest must match query");
            if (n == 0) return;
            sepal_nearest_sat_l1_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
                spatial.data(), query.data(), sat.data(), nearest.data(),
                n_sat);
            CUDA_CHECK_LAST_ERROR(sepal_nearest_sat_l1_kernel);
        },
        "spatial"_a, "query"_a, "sat"_a, nb::kw_only(), "nearest"_a,
        "stream"_a = 0);
}

NB_MODULE(_sepal_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
    m.def("occupancy", &sepal_occupancy, "f32"_a, "max_neighs"_a, "store"_a,
          "cluster_size"_a, "n_cells"_a, "block_size"_a);
}
