// Native Leiden community detection (`_leiden_cuda`), bitwise reproducible.

// CUB launches from the host only: its device-side launch paths (relocatable
// device code) would make compute-sanitizer initcheck skip the module.
#ifndef CUB_DISABLE_CDP
#define CUB_DISABLE_CDP
#endif

#include "driver.cuh"

NB_MODULE(_leiden_cuda, m) {
    m.doc() = "Native bitwise-reproducible Leiden (rapids-singlecell)";
    leiden::register_api(m);
}
