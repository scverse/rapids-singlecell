//! Harmony regression orchestration over native kernels and cuBLAS.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Dtype, current_device},
    harmony::*,
};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyfunction]
#[pyo3(signature=(O,*,lambda_kb,n_batches,n_clusters,cluster_k,inv_mat,g_factor,g_P_row0,stream=0))]
pub fn compute_inv_mat(
    py: Python<'_>,
    O: &Bound<'_, PyAny>,
    lambda_kb: &Bound<'_, PyAny>,
    n_batches: u64,
    n_clusters: u64,
    cluster_k: i32,
    inv_mat: &Bound<'_, PyAny>,
    g_factor: &Bound<'_, PyAny>,
    g_P_row0: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let O = floating(O, &cp, "O", 0)?;
    let dtype = O.dtype;
    if O.len < product(&[n_batches, n_clusters])? {
        return Err(PyValueError::new_err("O buffer is too small"));
    }
    let lambda_kb = read(
        lambda_kb,
        &cp,
        "lambda_kb",
        Some(dtype),
        product(&[n_batches, n_clusters])?,
    )?;
    let inv_mat = read(
        inv_mat,
        &cp,
        "inv_mat",
        Some(dtype),
        product(&[n_batches + 1, n_batches + 1])?,
    )?;
    let g_factor = read(
        g_factor,
        &cp,
        "g_factor",
        Some(dtype),
        product(&[n_batches])?,
    )?;
    let g_P_row0 = read(
        g_P_row0,
        &cp,
        "g_P_row0",
        Some(dtype),
        product(&[n_batches])?,
    )?;
    let device = current_device(&cp, &[&O, &lambda_kb, &inv_mat, &g_factor, &g_P_row0])?;
    disjoint(&[&inv_mat, &g_factor, &g_P_row0], &[&O, &lambda_kb])?;
    if cluster_k < 0 || cluster_k as u64 >= n_clusters {
        return Err(PyValueError::new_err("cluster_k is out of range"));
    }
    launch(
        device,
        "harmony_inverse",
        dtype,
        32,
        stream,
        &mut [
            Arg::Ptr(O.pointer),
            Arg::Ptr(lambda_kb.pointer),
            Arg::Ptr(inv_mat.pointer),
            Arg::Ptr(g_factor.pointer),
            Arg::Ptr(g_P_row0.pointer),
            Arg::U64(n_batches),
            Arg::U64(n_clusters),
            Arg::I32(cluster_k),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X,*,R,O,cats,cat_offsets,cell_indices,lambda_kb,n_cells,n_pcs,n_clusters,n_batches,Z,inv_mat,R_col,Phi_t_diag_R_X,W,g_factor,g_P_row0,stream=0,handle))]
pub fn correction_fast(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    R: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    cat_offsets: &Bound<'_, PyAny>,
    cell_indices: &Bound<'_, PyAny>,
    lambda_kb: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_clusters: u64,
    n_batches: u64,
    Z: &Bound<'_, PyAny>,
    inv_mat: &Bound<'_, PyAny>,
    R_col: &Bound<'_, PyAny>,
    Phi_t_diag_R_X: &Bound<'_, PyAny>,
    W: &Bound<'_, PyAny>,
    g_factor: &Bound<'_, PyAny>,
    g_P_row0: &Bound<'_, PyAny>,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n_cells, n_pcs])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let R = read(R, &cp, "R", Some(dtype), product(&[n_cells, n_clusters])?)?;
    let O = read(O, &cp, "O", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let lambda_kb = read(
        lambda_kb,
        &cp,
        "lambda_kb",
        Some(dtype),
        product(&[n_batches, n_clusters])?,
    )?;
    let Z = read(Z, &cp, "Z", Some(dtype), product(&[n_cells, n_pcs])?)?;
    let cats = read(cats, &cp, "cats", Some(Dtype::I32), product(&[n_cells])?)?;
    let cat_offsets = read(
        cat_offsets,
        &cp,
        "cat_offsets",
        Some(Dtype::I32),
        product(&[n_batches + 1])?,
    )?;
    let cell_indices = read(
        cell_indices,
        &cp,
        "cell_indices",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &X,
            &R,
            &O,
            &lambda_kb,
            &Z,
            &cats,
            &cat_offsets,
            &cell_indices,
        ],
    )?;
    let inv_mat = read(
        inv_mat,
        &cp,
        "inv_mat",
        Some(dtype),
        product(&[n_batches + 1, n_batches + 1])?,
    )?;
    let R_col = read(R_col, &cp, "R_col", Some(dtype), n_cells)?;
    let Phi_t_diag_R_X = read(
        Phi_t_diag_R_X,
        &cp,
        "Phi_t_diag_R_X",
        Some(dtype),
        product(&[n_batches + 1, n_pcs])?,
    )?;
    let W = read(W, &cp, "W", Some(dtype), product(&[n_batches + 1, n_pcs])?)?;
    let g_factor = read(g_factor, &cp, "g_factor", Some(dtype), n_batches)?;
    let g_P_row0 = read(g_P_row0, &cp, "g_P_row0", Some(dtype), n_batches)?;
    current_device(
        &cp,
        &[&inv_mat, &R_col, &Phi_t_diag_R_X, &W, &g_factor, &g_P_row0],
    )?;
    disjoint(
        &[
            &Z,
            &inv_mat,
            &R_col,
            &Phi_t_diag_R_X,
            &W,
            &g_factor,
            &g_P_row0,
        ],
        &[&X, &R, &O, &lambda_kb, &cats, &cat_offsets, &cell_indices],
    )?;
    copy(
        &cp,
        Z.pointer,
        X.pointer,
        n_cells * n_pcs * itemsize(dtype),
        stream,
    )?;
    for k in 0..n_clusters {
        launch(
            device,
            "harmony_inverse",
            dtype,
            32,
            stream,
            &mut [
                Arg::Ptr(O.pointer),
                Arg::Ptr(lambda_kb.pointer),
                Arg::Ptr(inv_mat.pointer),
                Arg::Ptr(g_factor.pointer),
                Arg::Ptr(g_P_row0.pointer),
                Arg::U64(n_batches),
                Arg::U64(n_clusters),
                Arg::I32(k as i32),
            ],
        )?;
        launch(
            device,
            "harmony_column",
            dtype,
            n_cells,
            stream,
            &mut [
                Arg::Ptr(R.pointer),
                Arg::Ptr(R_col.pointer),
                Arg::U64(n_cells),
                Arg::U64(n_clusters),
                Arg::U64(k),
            ],
        )?;
        if n_cells < 300_000 {
            // Eight independent intercept partitions accumulate into row 0;
            // category rows are owned and overwritten by their paired blocks.
            zero(&cp, Phi_t_diag_R_X.pointer, n_pcs * itemsize(dtype), stream)?;
        }
        launch_grid(
            device,
            "harmony_weighted_rhs",
            dtype,
            (n_batches + if n_cells < 300_000 { 8 } else { 0 }) * n_pcs.div_ceil(2),
            1024,
            stream,
            &mut [
                Arg::Ptr(X.pointer),
                Arg::Ptr(R_col.pointer),
                Arg::Ptr(cat_offsets.pointer),
                Arg::Ptr(cell_indices.pointer),
                Arg::Ptr(Phi_t_diag_R_X.pointer),
                Arg::U64(n_cells),
                Arg::U64(n_pcs),
                Arg::U64(n_batches),
            ],
        )?;
        if n_cells >= 300_000 {
            // Preserve the large-cell intercept BLAS route; the native
            // segmented kernel computes only category rows in this regime.
            gemm(
                py,
                dtype,
                handle,
                stream,
                0,
                0,
                n_pcs,
                1,
                n_cells,
                X.pointer,
                n_pcs,
                R_col.pointer,
                n_cells,
                Phi_t_diag_R_X.pointer,
                n_pcs,
                0.,
            )?;
        }
        gemm(
            py,
            dtype,
            handle,
            stream,
            0,
            0,
            n_pcs,
            n_batches + 1,
            n_batches + 1,
            Phi_t_diag_R_X.pointer,
            n_pcs,
            inv_mat.pointer,
            n_batches + 1,
            W.pointer,
            n_pcs,
            0.,
        )?;
        zero(&cp, W.pointer, n_pcs * itemsize(dtype), stream)?;
        launch(
            device,
            "harmony_apply",
            dtype,
            n_cells * n_pcs,
            stream,
            &mut [
                Arg::Ptr(X.pointer),
                Arg::Ptr(R.pointer),
                Arg::Ptr(W.pointer),
                Arg::Ptr(cats.pointer),
                Arg::Ptr(Z.pointer),
                Arg::U64(n_cells),
                Arg::U64(n_pcs),
                Arg::U64(n_clusters),
                Arg::U64(n_batches),
                Arg::U64(1),
                Arg::U32(0),
                Arg::I32(k as i32),
            ],
        )?;
    }
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X,*,R,O,cats,cat_offsets,cell_indices,lambda_kb,n_cells,n_pcs,n_clusters,n_batches,Z,inv_mats,Phi_t_diag_R_X_all,W_all,g_factor,g_P_row0,X_batch,R_batch,batch_chunk_size,stream=0,handle))]
pub fn correction_batched(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    R: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    cat_offsets: &Bound<'_, PyAny>,
    cell_indices: &Bound<'_, PyAny>,
    lambda_kb: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_clusters: u64,
    n_batches: u64,
    Z: &Bound<'_, PyAny>,
    inv_mats: &Bound<'_, PyAny>,
    Phi_t_diag_R_X_all: &Bound<'_, PyAny>,
    W_all: &Bound<'_, PyAny>,
    g_factor: &Bound<'_, PyAny>,
    g_P_row0: &Bound<'_, PyAny>,
    X_batch: &Bound<'_, PyAny>,
    R_batch: &Bound<'_, PyAny>,
    batch_chunk_size: u64,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n_cells, n_pcs])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let R = read(R, &cp, "R", Some(dtype), product(&[n_cells, n_clusters])?)?;
    let O = read(O, &cp, "O", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let lambda_kb = read(
        lambda_kb,
        &cp,
        "lambda_kb",
        Some(dtype),
        product(&[n_batches, n_clusters])?,
    )?;
    let Z = read(Z, &cp, "Z", Some(dtype), product(&[n_cells, n_pcs])?)?;
    let cats = read(cats, &cp, "cats", Some(Dtype::I32), product(&[n_cells])?)?;
    let cat_offsets = read(
        cat_offsets,
        &cp,
        "cat_offsets",
        Some(Dtype::I32),
        product(&[n_batches + 1])?,
    )?;
    let cell_indices = read(
        cell_indices,
        &cp,
        "cell_indices",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &X,
            &R,
            &O,
            &lambda_kb,
            &Z,
            &cats,
            &cat_offsets,
            &cell_indices,
        ],
    )?;
    if batch_chunk_size == 0 {
        return Err(PyValueError::new_err(
            "correction_batched requires positive batch scratch capacity",
        ));
    }
    let inv_mats = read(
        inv_mats,
        &cp,
        "inv_mats",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_batches + 1])?,
    )?;
    let Phi_t_diag_R_X_all = read(
        Phi_t_diag_R_X_all,
        &cp,
        "Phi_t_diag_R_X_all",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_pcs])?,
    )?;
    let W_all = read(
        W_all,
        &cp,
        "W_all",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_pcs])?,
    )?;
    let g_factor = read(
        g_factor,
        &cp,
        "g_factor",
        Some(dtype),
        product(&[n_clusters, n_batches])?,
    )?;
    let g_P_row0 = read(
        g_P_row0,
        &cp,
        "g_P_row0",
        Some(dtype),
        product(&[n_clusters, n_batches])?,
    )?;
    let X_batch = read(
        X_batch,
        &cp,
        "X_batch",
        Some(dtype),
        product(&[batch_chunk_size, n_pcs])?,
    )?;
    let R_batch = read(
        R_batch,
        &cp,
        "R_batch",
        Some(dtype),
        product(&[batch_chunk_size, n_clusters])?,
    )?;
    current_device(
        &cp,
        &[
            &inv_mats,
            &Phi_t_diag_R_X_all,
            &W_all,
            &g_factor,
            &g_P_row0,
            &X_batch,
            &R_batch,
        ],
    )?;
    disjoint(
        &[
            &Z,
            &inv_mats,
            &Phi_t_diag_R_X_all,
            &W_all,
            &g_factor,
            &g_P_row0,
            &X_batch,
            &R_batch,
        ],
        &[&X, &R, &O, &lambda_kb, &cats, &cat_offsets, &cell_indices],
    )?;
    launch(
        device,
        "harmony_inverse",
        dtype,
        n_clusters * 32,
        stream,
        &mut [
            Arg::Ptr(O.pointer),
            Arg::Ptr(lambda_kb.pointer),
            Arg::Ptr(inv_mats.pointer),
            Arg::Ptr(g_factor.pointer),
            Arg::Ptr(g_P_row0.pointer),
            Arg::U64(n_batches),
            Arg::U64(n_clusters),
            Arg::I32(-1),
        ],
    )?;
    // Keep the bounded per-category GEMM path: it retains dense BLAS
    // throughput without materializing sorted copies of the entire embedding.
    zero(
        &cp,
        Phi_t_diag_R_X_all.pointer,
        n_clusters * (n_batches + 1) * n_pcs * itemsize(dtype),
        stream,
    )?;
    gemm(
        py,
        dtype,
        handle,
        stream,
        0,
        1,
        n_pcs,
        n_clusters,
        n_cells,
        X.pointer,
        n_pcs,
        R.pointer,
        n_clusters,
        Phi_t_diag_R_X_all.pointer,
        (n_batches + 1) * n_pcs,
        0.,
    )?;
    let mut offsets = vec![0i32; (n_batches + 1) as usize];
    let rt = cp.getattr("cuda")?.getattr("runtime")?;
    rt.call_method1(
        "memcpyAsync",
        (
            offsets.as_mut_ptr() as usize,
            cat_offsets.pointer,
            (n_batches + 1) * 4,
            2,
            stream,
        ),
    )?;
    rt.call_method1("streamSynchronize", (stream,))?;
    for batch in 0..n_batches as usize {
        let begin = offsets[batch];
        let end = offsets[batch + 1];
        if begin < 0 || end < begin || end as u64 > n_cells {
            return Err(PyValueError::new_err("invalid category offsets"));
        }
        let output = Phi_t_diag_R_X_all.pointer + (batch as u64 + 1) * n_pcs * itemsize(dtype);
        let mut position = begin as u64;
        let mut beta = 0.;
        while position < end as u64 {
            let count = batch_chunk_size.min(end as u64 - position);
            let indices = cell_indices.pointer + position * 4;
            for (src, dst, cols) in [(&X, &X_batch, n_pcs), (&R, &R_batch, n_clusters)] {
                launch(
                    device,
                    "harmony_rows",
                    dtype,
                    count * cols,
                    stream,
                    &mut [
                        Arg::Ptr(src.pointer),
                        Arg::Ptr(indices),
                        Arg::Ptr(dst.pointer),
                        Arg::U64(count),
                        Arg::U64(cols),
                        Arg::U64(n_cells),
                        Arg::U32(0),
                    ],
                )?;
            }
            gemm(
                py,
                dtype,
                handle,
                stream,
                0,
                1,
                n_pcs,
                n_clusters,
                count,
                X_batch.pointer,
                n_pcs,
                R_batch.pointer,
                n_clusters,
                output,
                (n_batches + 1) * n_pcs,
                beta,
            )?;
            beta = 1.;
            position += count;
        }
    }
    for k in 0..n_clusters {
        let offset = k * (n_batches + 1) * n_pcs * itemsize(dtype);
        gemm(
            py,
            dtype,
            handle,
            stream,
            0,
            0,
            n_pcs,
            n_batches + 1,
            n_batches + 1,
            Phi_t_diag_R_X_all.pointer + offset,
            n_pcs,
            inv_mats.pointer + k * (n_batches + 1) * (n_batches + 1) * itemsize(dtype),
            n_batches + 1,
            W_all.pointer + offset,
            n_pcs,
            0.,
        )?;
        zero(&cp, W_all.pointer + offset, n_pcs * itemsize(dtype), stream)?;
    }
    launch(
        device,
        "harmony_apply",
        dtype,
        n_cells * n_pcs,
        stream,
        &mut [
            Arg::Ptr(X.pointer),
            Arg::Ptr(R.pointer),
            Arg::Ptr(W_all.pointer),
            Arg::Ptr(cats.pointer),
            Arg::Ptr(Z.pointer),
            Arg::U64(n_cells),
            Arg::U64(n_pcs),
            Arg::U64(n_clusters),
            Arg::U64(n_batches),
            Arg::U64(1),
            Arg::U32(1),
            Arg::I32(-1),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X,*,R,W_all,cats,n_cells,n_pcs,n_clusters,n_batches,n_covariates,initialize_output,Z,stream=0))]
pub fn apply_multi(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    R: &Bound<'_, PyAny>,
    W_all: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_clusters: u64,
    n_batches: u64,
    n_covariates: u64,
    initialize_output: bool,
    Z: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n_cells, n_pcs])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let R = read(R, &cp, "R", Some(dtype), product(&[n_cells, n_clusters])?)?;
    let W_all = read(
        W_all,
        &cp,
        "W_all",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_pcs])?,
    )?;
    let Z = read(Z, &cp, "Z", Some(dtype), product(&[n_cells, n_pcs])?)?;
    let cats = read(
        cats,
        &cp,
        "cats",
        Some(Dtype::I32),
        product(&[n_cells, n_covariates])?,
    )?;
    let device = current_device(&cp, &[&X, &R, &W_all, &Z, &cats])?;
    disjoint(&[&Z], &[&R, &W_all, &cats])?;
    if Z.pointer != X.pointer {
        Z.require_disjoint(&X)?;
    }
    if n_covariates < 2 {
        return Err(PyValueError::new_err(
            "apply_multi requires at least two covariates",
        ));
    }
    launch(
        device,
        "harmony_apply",
        dtype,
        n_cells * n_pcs,
        stream,
        &mut [
            Arg::Ptr(X.pointer),
            Arg::Ptr(R.pointer),
            Arg::Ptr(W_all.pointer),
            Arg::Ptr(cats.pointer),
            Arg::Ptr(Z.pointer),
            Arg::U64(n_cells),
            Arg::U64(n_pcs),
            Arg::U64(n_clusters),
            Arg::U64(n_batches),
            Arg::U64(n_covariates),
            Arg::U32(initialize_output as u32),
            Arg::I32(-1),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X,*,R,O,joint_codes,joint_cats,joint_offsets,joint_cell_indices,marginal_joint_offsets,marginal_joint_indices,lambda_kb,active_mask,n_cells,n_pcs,n_clusters,n_batches,n_covariates,n_joint_categories,gram,rhs,joint_O,joint_rhs,stream=0,handle))]
pub fn prepare_multi(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    R: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    joint_codes: &Bound<'_, PyAny>,
    joint_cats: &Bound<'_, PyAny>,
    joint_offsets: &Bound<'_, PyAny>,
    joint_cell_indices: &Bound<'_, PyAny>,
    marginal_joint_offsets: &Bound<'_, PyAny>,
    marginal_joint_indices: &Bound<'_, PyAny>,
    lambda_kb: &Bound<'_, PyAny>,
    active_mask: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_clusters: u64,
    n_batches: u64,
    n_covariates: u64,
    n_joint_categories: u64,
    gram: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    joint_O: &Bound<'_, PyAny>,
    joint_rhs: &Bound<'_, PyAny>,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n_cells, n_pcs])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let R = read(R, &cp, "R", Some(dtype), product(&[n_cells, n_clusters])?)?;
    let O = read(O, &cp, "O", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let lambda_kb = read(
        lambda_kb,
        &cp,
        "lambda_kb",
        Some(dtype),
        product(&[n_batches, n_clusters])?,
    )?;
    let gram = read(
        gram,
        &cp,
        "gram",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_batches + 1])?,
    )?;
    let rhs = read(
        rhs,
        &cp,
        "rhs",
        Some(dtype),
        product(&[n_clusters, n_batches + 1, n_pcs])?,
    )?;
    let joint_O = read(
        joint_O,
        &cp,
        "joint_O",
        Some(dtype),
        product(&[n_joint_categories, n_clusters])?,
    )?;
    let joint_rhs = read(
        joint_rhs,
        &cp,
        "joint_rhs",
        Some(dtype),
        product(&[n_joint_categories, n_clusters, n_pcs])?,
    )?;
    let joint_codes = read(
        joint_codes,
        &cp,
        "joint_codes",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let joint_cats = read(
        joint_cats,
        &cp,
        "joint_cats",
        Some(Dtype::I32),
        product(&[n_joint_categories, n_covariates])?,
    )?;
    let joint_offsets = read(
        joint_offsets,
        &cp,
        "joint_offsets",
        Some(Dtype::I32),
        product(&[n_joint_categories + 1])?,
    )?;
    let joint_cell_indices = read(
        joint_cell_indices,
        &cp,
        "joint_cell_indices",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let marginal_joint_offsets = read(
        marginal_joint_offsets,
        &cp,
        "marginal_joint_offsets",
        Some(Dtype::I32),
        product(&[n_batches + 1])?,
    )?;
    let marginal_joint_indices = read(
        marginal_joint_indices,
        &cp,
        "marginal_joint_indices",
        Some(Dtype::I32),
        product(&[0])?,
    )?;
    let active_mask = read(
        active_mask,
        &cp,
        "active_mask",
        Some(Dtype::U8),
        product(&[n_batches, n_clusters])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &X,
            &R,
            &O,
            &lambda_kb,
            &gram,
            &rhs,
            &joint_O,
            &joint_rhs,
            &joint_codes,
            &joint_cats,
            &joint_offsets,
            &joint_cell_indices,
            &marginal_joint_offsets,
            &marginal_joint_indices,
            &active_mask,
        ],
    )?;
    disjoint(
        &[&gram, &rhs, &joint_O, &joint_rhs],
        &[
            &X,
            &R,
            &O,
            &joint_codes,
            &joint_cats,
            &joint_offsets,
            &joint_cell_indices,
            &marginal_joint_offsets,
            &marginal_joint_indices,
            &lambda_kb,
            &active_mask,
        ],
    )?;
    if n_covariates < 2 {
        return Err(PyValueError::new_err(
            "prepare_multi requires at least two covariates",
        ));
    }
    zero(
        &cp,
        joint_O.pointer,
        n_joint_categories * n_clusters * itemsize(dtype),
        stream,
    )?;
    zero(
        &cp,
        rhs.pointer,
        n_clusters * (n_batches + 1) * n_pcs * itemsize(dtype),
        stream,
    )?;
    launch(
        device,
        "harmony_scatter",
        dtype,
        n_cells * n_clusters,
        stream,
        &mut [
            Arg::Ptr(R.pointer),
            Arg::Ptr(joint_codes.pointer),
            Arg::Ptr(joint_O.pointer),
            Arg::U64(n_cells),
            Arg::U64(n_clusters),
            Arg::U64(1),
            Arg::U64(n_joint_categories),
            Arg::I32(1),
        ],
    )?;
    launch(
        device,
        "harmony_gram",
        dtype,
        n_clusters * (n_batches + 1) * (n_batches + 1),
        stream,
        &mut [
            Arg::Ptr(O.pointer),
            Arg::Ptr(lambda_kb.pointer),
            Arg::Ptr(active_mask.pointer),
            Arg::Ptr(joint_O.pointer),
            Arg::Ptr(gram.pointer),
            Arg::U64(n_batches),
            Arg::U64(n_clusters),
            Arg::U64(n_joint_categories),
        ],
    )?;
    launch(
        device,
        "harmony_cross",
        dtype,
        n_joint_categories * n_clusters,
        stream,
        &mut [
            Arg::Ptr(joint_O.pointer),
            Arg::Ptr(joint_cats.pointer),
            Arg::Ptr(active_mask.pointer),
            Arg::Ptr(gram.pointer),
            Arg::U64(n_batches),
            Arg::U64(n_clusters),
            Arg::U64(n_joint_categories),
            Arg::U64(n_covariates),
        ],
    )?;
    gemm(
        py,
        dtype,
        handle,
        stream,
        0,
        1,
        n_pcs,
        n_clusters,
        n_cells,
        X.pointer,
        n_pcs,
        R.pointer,
        n_clusters,
        rhs.pointer,
        (n_batches + 1) * n_pcs,
        0.,
    )?;
    launch(
        device,
        "harmony_joint_rhs",
        dtype,
        n_joint_categories * n_clusters * n_pcs * 32,
        stream,
        &mut [
            Arg::Ptr(X.pointer),
            Arg::Ptr(R.pointer),
            Arg::Ptr(joint_offsets.pointer),
            Arg::Ptr(joint_cell_indices.pointer),
            Arg::Ptr(joint_rhs.pointer),
            Arg::U64(n_cells),
            Arg::U64(n_pcs),
            Arg::U64(n_clusters),
            Arg::U64(n_joint_categories),
        ],
    )?;
    launch(
        device,
        "harmony_marginal_rhs",
        dtype,
        n_clusters * n_batches * n_pcs,
        stream,
        &mut [
            Arg::Ptr(joint_rhs.pointer),
            Arg::Ptr(marginal_joint_offsets.pointer),
            Arg::Ptr(marginal_joint_indices.pointer),
            Arg::Ptr(active_mask.pointer),
            Arg::Ptr(rhs.pointer),
            Arg::U64(n_pcs),
            Arg::U64(n_clusters),
            Arg::U64(n_batches),
            Arg::U64(n_joint_categories),
            Arg::U64(marginal_joint_indices.len),
        ],
    )?;
    Ok(())
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(
        parent.py(),
        "rapids_singlecell._cuda._harmony_correction_cuda",
    )?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(correction_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_inv_mat, &m)?)?;
    parent.add_submodule(&m)?;
    let m = PyModule::new(
        parent.py(),
        "rapids_singlecell._cuda._harmony_correction_batched_cuda",
    )?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(correction_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(prepare_multi, &m)?)?;
    m.add_function(wrap_pyfunction!(apply_multi, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
