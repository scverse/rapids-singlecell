//! Shared PyO3 bindings for the Rust CUDA backend.

mod array;
mod blas;
mod device_sparse;
mod domain;
mod elementwise;
mod gmm;
mod harmony;
mod harmony_clustering;
mod harmony_correction;
mod host_buffer;
mod host_parallel;
mod host_sparse;
mod jaccard;
mod preprocessing;
mod rank_sort;
mod rank_stats;
mod rank_stream;
mod rank_support;
mod runtime;
mod sparse2dense;
mod sparse_ovo;
mod sparse_ovr;
mod staging;
mod wilcoxon;
mod wilcoxon_binned;
mod wilcoxon_sparse;

use pyo3::prelude::*;

#[pymodule]
fn _rust_cuda(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.setattr("__backend__", "rust")?;
    harmony::register(module)?;
    preprocessing::register(module)?;
    domain::register(module)?;
    harmony_correction::register(module)?;
    harmony_clustering::register(module)?;
    gmm::register(module)?;
    elementwise::register(module)?;
    rank_stats::register(module)?;
    rank_stream::register(module)?;
    wilcoxon_binned::register(module)?;
    wilcoxon::register(module)?;
    wilcoxon_sparse::register(module)?;
    let jaccard = PyModule::new(module.py(), "rapids_singlecell._cuda._jaccard_cuda")?;
    jaccard.setattr("__backend__", "rust")?;
    jaccard.add_function(wrap_pyfunction!(jaccard::jaccard_shared_counts, &jaccard)?)?;
    module.add_submodule(&jaccard)?;
    let sparse = PyModule::new(module.py(), "rapids_singlecell._cuda._sparse2dense_cuda")?;
    sparse.setattr("__backend__", "rust")?;
    sparse.add_function(wrap_pyfunction!(sparse2dense::sparse2dense, &sparse)?)?;
    module.add_submodule(&sparse)?;
    Ok(())
}
