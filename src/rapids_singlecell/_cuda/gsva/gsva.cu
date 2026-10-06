#include "kernels_gsva_density.cuh"
#include "kernels_gsva_rank.cuh"
#include "kernels_gsva_score.cuh"

template <typename Device>
void register_bindings(nb::module_& m) {
    gsva_density::register_bindings<Device>(m);
    gsva_rank::register_bindings<Device>(m);
    gsva_score::register_bindings<Device>(m);
}

NB_MODULE(_gsva_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
