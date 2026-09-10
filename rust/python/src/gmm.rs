//! Gaussian-mixture bindings using native Rust kernels and CUDA math libraries.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Dtype, current_device},
    harmony::*,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyTuple},
};
#[pyfunction]
#[pyo3(signature=(X,weights,means,prec_chol,log_det_half,log_prob,resp,ll_per_cell,*,n,d,K,stream=0))]
pub fn e_step(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    prec_chol: &Bound<'_, PyAny>,
    log_det_half: &Bound<'_, PyAny>,
    log_prob: &Bound<'_, PyAny>,
    resp: &Bound<'_, PyAny>,
    ll_per_cell: &Bound<'_, PyAny>,
    n: u64,
    d: u64,
    K: u64,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n, d])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let weights = read(weights, &cp, "weights", Some(dtype), product(&[K])?)?;
    let means = read(means, &cp, "means", Some(dtype), product(&[K, d])?)?;
    let prec_chol = read(
        prec_chol,
        &cp,
        "prec_chol",
        Some(dtype),
        product(&[K, d, d])?,
    )?;
    let log_det_half = read(
        log_det_half,
        &cp,
        "log_det_half",
        Some(dtype),
        product(&[K])?,
    )?;
    let log_prob = read(log_prob, &cp, "log_prob", Some(dtype), product(&[n, K])?)?;
    let resp = read(resp, &cp, "resp", Some(dtype), product(&[n, K])?)?;
    let ll_per_cell = read(ll_per_cell, &cp, "ll_per_cell", Some(dtype), product(&[n])?)?;
    let device = current_device(
        &cp,
        &[
            &X,
            &weights,
            &means,
            &prec_chol,
            &log_det_half,
            &log_prob,
            &resp,
            &ll_per_cell,
        ],
    )?;
    disjoint(
        &[&log_prob, &resp, &ll_per_cell],
        &[&X, &weights, &means, &prec_chol, &log_det_half],
    )?;
    if n == 0 || d == 0 || K == 0 {
        return Ok(());
    }
    let mut args = [
        Arg::Ptr(X.pointer),
        Arg::Ptr(weights.pointer),
        Arg::Ptr(means.pointer),
        Arg::Ptr(prec_chol.pointer),
        Arg::Ptr(log_det_half.pointer),
        Arg::Ptr(log_prob.pointer),
        Arg::U64(n),
        Arg::U64(d),
        Arg::U64(K),
    ];
    if n > 0 && K > 0 && d > 0 && d <= 64 && K <= 65_535 && n.div_ceil(64) <= i32::MAX as u64 {
        let dimension = match d {
            16 => "16",
            32 => "32",
            50 => "50",
            64 => "64",
            _ => "dynamic",
        };
        let suffix = if dtype == Dtype::F32 { "f32" } else { "f64" };
        let name = format!("gmm_small_{dimension}_{suffix}");
        let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
        // Larger float32 blocks amortize component loading; float64 retains
        // the smaller block because its register and arithmetic costs differ.
        let threads = if dtype == Dtype::F32 { 256 } else { 64 };
        // SAFETY: each block caches one component and owns distinct cells.
        unsafe {
            crate::runtime::launch(
                device,
                &name,
                (n.div_ceil(threads) as u32, K as u32, 1),
                (threads as u32, 1, 1),
                stream,
                &mut pointers,
            )?;
        }
    } else if n > 0 && K > 0 && d > 0 && K <= 65_535 && n.div_ceil(256) <= i32::MAX as u64 {
        let name = if dtype == Dtype::F32 {
            "gmm_tiled_f32"
        } else {
            "gmm_tiled_f64"
        };
        let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
        // SAFETY: uniform block barriers cover shared tiles, and each valid
        // cell/component output has a unique writer.
        unsafe {
            crate::runtime::launch(
                device,
                name,
                (n.div_ceil(256) as u32, K as u32, 1),
                (256, 1, 1),
                stream,
                &mut pointers,
            )?;
        }
    } else {
        launch(device, "gmm_logprob", dtype, n * K * 32, stream, &mut args)?;
    }
    launch(
        device,
        "gmm_normalize",
        dtype,
        n * 32,
        stream,
        &mut [
            Arg::Ptr(log_prob.pointer),
            Arg::Ptr(resp.pointer),
            Arg::Ptr(ll_per_cell.pointer),
            Arg::U64(n),
            Arg::U64(K),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X,weights,means,prec_chol,log_det_half,centered_workspace,y_workspace,log_prob,resp,ll_per_cell,*,n,d,K,stream=0,handle))]
pub fn e_step_cublas(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    prec_chol: &Bound<'_, PyAny>,
    log_det_half: &Bound<'_, PyAny>,
    centered_workspace: &Bound<'_, PyAny>,
    y_workspace: &Bound<'_, PyAny>,
    log_prob: &Bound<'_, PyAny>,
    resp: &Bound<'_, PyAny>,
    ll_per_cell: &Bound<'_, PyAny>,
    n: u64,
    d: u64,
    K: u64,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n, d])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let weights = read(weights, &cp, "weights", Some(dtype), product(&[K])?)?;
    let means = read(means, &cp, "means", Some(dtype), product(&[K, d])?)?;
    let prec_chol = read(
        prec_chol,
        &cp,
        "prec_chol",
        Some(dtype),
        product(&[K, d, d])?,
    )?;
    let log_det_half = read(
        log_det_half,
        &cp,
        "log_det_half",
        Some(dtype),
        product(&[K])?,
    )?;
    let log_prob = read(log_prob, &cp, "log_prob", Some(dtype), product(&[n, K])?)?;
    let resp = read(resp, &cp, "resp", Some(dtype), product(&[n, K])?)?;
    let ll_per_cell = read(ll_per_cell, &cp, "ll_per_cell", Some(dtype), product(&[n])?)?;
    let device = current_device(
        &cp,
        &[
            &X,
            &weights,
            &means,
            &prec_chol,
            &log_det_half,
            &log_prob,
            &resp,
            &ll_per_cell,
        ],
    )?;
    let centered_workspace = read(
        centered_workspace,
        &cp,
        "centered_workspace",
        Some(dtype),
        product(&[n, d])?,
    )?;
    let y_workspace = read(
        y_workspace,
        &cp,
        "y_workspace",
        Some(dtype),
        product(&[n, d])?,
    )?;
    current_device(&cp, &[&centered_workspace, &y_workspace])?;
    disjoint(
        &[
            &centered_workspace,
            &y_workspace,
            &log_prob,
            &resp,
            &ll_per_cell,
        ],
        &[&X, &weights, &means, &prec_chol, &log_det_half],
    )?;
    if n == 0 || d == 0 || K == 0 {
        return Ok(());
    }
    for cl in 0..K {
        launch(
            device,
            "gmm_center",
            dtype,
            n * d,
            stream,
            &mut [
                Arg::Ptr(X.pointer),
                Arg::Ptr(means.pointer),
                Arg::Ptr(resp.pointer),
                Arg::Ptr(centered_workspace.pointer),
                Arg::U64(n),
                Arg::U64(d),
                Arg::U64(K),
                Arg::U64(cl),
                Arg::U32(0),
            ],
        )?;
        gemm(
            py,
            dtype,
            handle,
            stream,
            0,
            0,
            d,
            n,
            d,
            prec_chol.pointer + cl * d * d * itemsize(dtype),
            d,
            centered_workspace.pointer,
            d,
            y_workspace.pointer,
            d,
            0.,
        )?;
        launch(
            device,
            "gmm_logprob_y",
            dtype,
            n * 32,
            stream,
            &mut [
                Arg::Ptr(y_workspace.pointer),
                Arg::Ptr(weights.pointer),
                Arg::Ptr(log_det_half.pointer),
                Arg::Ptr(log_prob.pointer),
                Arg::U64(n),
                Arg::U64(d),
                Arg::U64(K),
                Arg::U64(cl),
            ],
        )?;
    }
    launch(
        device,
        "gmm_normalize",
        dtype,
        n * 32,
        stream,
        &mut [
            Arg::Ptr(log_prob.pointer),
            Arg::Ptr(resp.pointer),
            Arg::Ptr(ll_per_cell.pointer),
            Arg::U64(n),
            Arg::U64(K),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(resp,X,ones,weights,means,covariances,N_k_workspace,num_workspace,centered_workspace,*,n,d,K,reg_covar,stream=0,handle))]
pub fn m_step(
    py: Python<'_>,
    resp: &Bound<'_, PyAny>,
    X: &Bound<'_, PyAny>,
    ones: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    covariances: &Bound<'_, PyAny>,
    N_k_workspace: &Bound<'_, PyAny>,
    num_workspace: &Bound<'_, PyAny>,
    centered_workspace: &Bound<'_, PyAny>,
    n: u64,
    d: u64,
    K: u64,
    reg_covar: f64,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let X = floating(X, &cp, "X", 0)?;
    let dtype = X.dtype;
    if X.len < product(&[n, d])? {
        return Err(PyValueError::new_err("X buffer is too small"));
    }
    let resp = read(resp, &cp, "resp", Some(dtype), product(&[n, K])?)?;
    let ones = read(ones, &cp, "ones", Some(dtype), product(&[n])?)?;
    let weights = read(weights, &cp, "weights", Some(dtype), product(&[K])?)?;
    let means = read(means, &cp, "means", Some(dtype), product(&[K, d])?)?;
    let covariances = read(
        covariances,
        &cp,
        "covariances",
        Some(dtype),
        product(&[K, d, d])?,
    )?;
    let N_k_workspace = read(
        N_k_workspace,
        &cp,
        "N_k_workspace",
        Some(dtype),
        product(&[K])?,
    )?;
    let num_workspace = read(
        num_workspace,
        &cp,
        "num_workspace",
        Some(dtype),
        product(&[K, d])?,
    )?;
    let centered_workspace = read(
        centered_workspace,
        &cp,
        "centered_workspace",
        Some(dtype),
        product(&[n, d])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &X,
            &resp,
            &ones,
            &weights,
            &means,
            &covariances,
            &N_k_workspace,
            &num_workspace,
            &centered_workspace,
        ],
    )?;
    disjoint(
        &[
            &weights,
            &means,
            &covariances,
            &N_k_workspace,
            &num_workspace,
            &centered_workspace,
        ],
        &[&resp, &X, &ones],
    )?;
    if n == 0 || d == 0 || K == 0 {
        return Ok(());
    }
    gemm(
        py,
        dtype,
        handle,
        stream,
        0,
        0,
        K,
        1,
        n,
        resp.pointer,
        K,
        ones.pointer,
        n,
        N_k_workspace.pointer,
        K,
        0.,
    )?;
    gemm(
        py,
        dtype,
        handle,
        stream,
        0,
        1,
        d,
        K,
        n,
        X.pointer,
        d,
        resp.pointer,
        K,
        num_workspace.pointer,
        d,
        0.,
    )?;
    launch(
        device,
        "gmm_means",
        dtype,
        K * d,
        stream,
        &mut [
            Arg::Ptr(N_k_workspace.pointer),
            Arg::Ptr(num_workspace.pointer),
            Arg::Ptr(weights.pointer),
            Arg::Ptr(means.pointer),
            Arg::U64(n),
            Arg::U64(d),
            Arg::U64(K),
        ],
    )?;
    for cl in 0..K {
        launch(
            device,
            "gmm_center",
            dtype,
            n * d,
            stream,
            &mut [
                Arg::Ptr(X.pointer),
                Arg::Ptr(means.pointer),
                Arg::Ptr(resp.pointer),
                Arg::Ptr(centered_workspace.pointer),
                Arg::U64(n),
                Arg::U64(d),
                Arg::U64(K),
                Arg::U64(cl),
                Arg::U32(1),
            ],
        )?;
        gemm(
            py,
            dtype,
            handle,
            stream,
            0,
            1,
            d,
            d,
            n,
            centered_workspace.pointer,
            d,
            centered_workspace.pointer,
            d,
            covariances.pointer + cl * d * d * itemsize(dtype),
            d,
            0.,
        )?;
    }
    launch(
        device,
        "gmm_cov",
        dtype,
        K * d * d,
        stream,
        &mut [
            Arg::Ptr(N_k_workspace.pointer),
            Arg::Ptr(covariances.pointer),
            Arg::U64(d),
            Arg::U64(K),
            scalar(dtype, reg_covar),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(cov_work,prec_chol,log_det,dev_info,*,dA,dB,d,K,stream=0,cublas_handle,cusolver_handle))]
pub fn precision_cholesky_full(
    py: Python<'_>,
    cov_work: &Bound<'_, PyAny>,
    prec_chol: &Bound<'_, PyAny>,
    log_det: &Bound<'_, PyAny>,
    dev_info: &Bound<'_, PyAny>,
    dA: usize,
    dB: usize,
    d: u64,
    K: u64,
    stream: usize,
    cublas_handle: usize,
    cusolver_handle: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let cov_work = floating(cov_work, &cp, "cov_work", 0)?;
    let dtype = cov_work.dtype;
    if cov_work.len < product(&[K, d, d])? {
        return Err(PyValueError::new_err("cov_work buffer is too small"));
    }
    let prec_chol = read(
        prec_chol,
        &cp,
        "prec_chol",
        Some(dtype),
        product(&[K, d, d])?,
    )?;
    let log_det = read(log_det, &cp, "log_det", Some(dtype), product(&[K])?)?;
    let dev_info = read(dev_info, &cp, "dev_info", Some(Dtype::I32), product(&[K])?)?;
    let device = current_device(&cp, &[&cov_work, &prec_chol, &log_det, &dev_info])?;
    disjoint(&[&cov_work, &prec_chol, &log_det, &dev_info], &[])?;
    if d == 0 || K == 0 {
        return Ok(());
    }
    if d > i32::MAX as u64 || K > i32::MAX as u64 {
        return Err(PyValueError::new_err(
            "batched solver dimensions exceed int32",
        ));
    }
    // Pointer tables are part of the established low-level API. Confirm their
    // CUDA allocation kind before passing them to the CUDA libraries.
    let rt = cp.getattr("cuda")?.getattr("runtime")?;
    rt.call_method1("pointerGetAttributes", (dA,))?;
    rt.call_method1("pointerGetAttributes", (dB,))?;
    let blas = py.import("cupy_backends.cuda.libs.cublas")?;
    let solver = py.import("cupy_backends.cuda.libs.cusolver")?;
    blas.call_method1("setStream", (cublas_handle, stream))?;
    solver.call_method1("setStream", (cusolver_handle, stream))?;
    solver
        .getattr(if dtype == Dtype::F32 {
            "spotrfBatched"
        } else {
            "dpotrfBatched"
        })?
        .call1((cusolver_handle, 0, d, dA, d, dev_info.pointer, K))?;
    launch(
        device,
        "gmm_identity",
        dtype,
        K * d * d,
        stream,
        &mut [Arg::Ptr(prec_chol.pointer), Arg::U64(d), Arg::U64(K)],
    )?;
    let one32 = 1f32;
    let one64 = 1f64;
    let alpha = if dtype == Dtype::F32 {
        (&one32 as *const f32) as usize
    } else {
        (&one64 as *const f64) as usize
    };
    let args = PyTuple::new(
        py,
        [
            cublas_handle.into_pyobject(py)?.into_any(),
            0i32.into_pyobject(py)?.into_any(),
            0i32.into_pyobject(py)?.into_any(),
            0i32.into_pyobject(py)?.into_any(),
            0i32.into_pyobject(py)?.into_any(),
            d.into_pyobject(py)?.into_any(),
            d.into_pyobject(py)?.into_any(),
            alpha.into_pyobject(py)?.into_any(),
            dA.into_pyobject(py)?.into_any(),
            d.into_pyobject(py)?.into_any(),
            dB.into_pyobject(py)?.into_any(),
            d.into_pyobject(py)?.into_any(),
            K.into_pyobject(py)?.into_any(),
        ],
    )?;
    blas.getattr(if dtype == Dtype::F32 {
        "strsmBatched"
    } else {
        "dtrsmBatched"
    })?
    .call1(args)?;
    launch(
        device,
        "gmm_logdet",
        dtype,
        K * 32,
        stream,
        &mut [
            Arg::Ptr(prec_chol.pointer),
            Arg::Ptr(log_det.pointer),
            Arg::U64(d),
            Arg::U64(K),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(pvec,offsets,m0,v0,m1_init,v1_init,resp1,m1_out,v1_out,w1_out,*,n_genes,max_iter,tol,reg_covar,stream=0))]
pub fn spherical_gmm_fit_batched(
    py: Python<'_>,
    pvec: &Bound<'_, PyAny>,
    offsets: &Bound<'_, PyAny>,
    m0: &Bound<'_, PyAny>,
    v0: &Bound<'_, PyAny>,
    m1_init: &Bound<'_, PyAny>,
    v1_init: &Bound<'_, PyAny>,
    resp1: &Bound<'_, PyAny>,
    m1_out: &Bound<'_, PyAny>,
    v1_out: &Bound<'_, PyAny>,
    w1_out: &Bound<'_, PyAny>,
    n_genes: u64,
    max_iter: u32,
    tol: f64,
    reg_covar: f64,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let pvec = floating(pvec, &cp, "pvec", 0)?;
    let dtype = pvec.dtype;
    if pvec.len < product(&[0])? {
        return Err(PyValueError::new_err("pvec buffer is too small"));
    }
    let m0 = read(m0, &cp, "m0", Some(dtype), product(&[n_genes])?)?;
    let v0 = read(v0, &cp, "v0", Some(dtype), product(&[n_genes])?)?;
    let m1_init = read(m1_init, &cp, "m1_init", Some(dtype), product(&[n_genes])?)?;
    let v1_init = read(v1_init, &cp, "v1_init", Some(dtype), product(&[n_genes])?)?;
    let resp1 = read(resp1, &cp, "resp1", Some(dtype), product(&[pvec.len])?)?;
    let m1_out = read(m1_out, &cp, "m1_out", Some(dtype), product(&[n_genes])?)?;
    let v1_out = read(v1_out, &cp, "v1_out", Some(dtype), product(&[n_genes])?)?;
    let w1_out = read(w1_out, &cp, "w1_out", Some(dtype), product(&[n_genes])?)?;
    let offsets = read(
        offsets,
        &cp,
        "offsets",
        Some(Dtype::I32),
        product(&[n_genes + 1])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &pvec, &m0, &v0, &m1_init, &v1_init, &resp1, &m1_out, &v1_out, &w1_out, &offsets,
        ],
    )?;
    disjoint(
        &[&resp1, &m1_out, &v1_out, &w1_out],
        &[&pvec, &offsets, &m0, &v0, &m1_init, &v1_init],
    )?;
    let block_em = pvec.dtype == Dtype::F64 || pvec.len / n_genes.max(1) > 1024;
    launch(
        device,
        if block_em {
            "gmm_spherical_block"
        } else {
            "gmm_spherical"
        },
        dtype,
        n_genes * if block_em { 256 } else { 32 },
        stream,
        &mut [
            Arg::Ptr(pvec.pointer),
            Arg::Ptr(offsets.pointer),
            Arg::Ptr(m0.pointer),
            Arg::Ptr(v0.pointer),
            Arg::Ptr(m1_init.pointer),
            Arg::Ptr(v1_init.pointer),
            Arg::Ptr(resp1.pointer),
            Arg::Ptr(m1_out.pointer),
            Arg::Ptr(v1_out.pointer),
            Arg::Ptr(w1_out.pointer),
            Arg::U64(n_genes),
            Arg::U64(pvec.len),
            Arg::U32(max_iter),
            scalar(dtype, tol),
            scalar(dtype, reg_covar),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(dat,dat_offsets,n_per_gene,k_per_gene,cell_offsets,feat_offsets,nt_cells_mean,guide_sel,nt_in_all,active_genes,pvec_scratch,resp1,*,n_active,max_k,max_iter,tol,reg_covar,stream=0))]
pub fn mixscape_project_em(
    py: Python<'_>,
    dat: &Bound<'_, PyAny>,
    dat_offsets: &Bound<'_, PyAny>,
    n_per_gene: &Bound<'_, PyAny>,
    k_per_gene: &Bound<'_, PyAny>,
    cell_offsets: &Bound<'_, PyAny>,
    feat_offsets: &Bound<'_, PyAny>,
    nt_cells_mean: &Bound<'_, PyAny>,
    guide_sel: &Bound<'_, PyAny>,
    nt_in_all: &Bound<'_, PyAny>,
    active_genes: &Bound<'_, PyAny>,
    pvec_scratch: &Bound<'_, PyAny>,
    resp1: &Bound<'_, PyAny>,
    n_active: u64,
    max_k: u64,
    max_iter: u32,
    tol: f64,
    reg_covar: f64,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let dat = floating(dat, &cp, "dat", 0)?;
    let dtype = dat.dtype;
    if dat.len < product(&[0])? {
        return Err(PyValueError::new_err("dat buffer is too small"));
    }
    let nt_cells_mean = read(
        nt_cells_mean,
        &cp,
        "nt_cells_mean",
        Some(dtype),
        product(&[0])?,
    )?;
    let pvec_scratch = read(
        pvec_scratch,
        &cp,
        "pvec_scratch",
        Some(dtype),
        product(&[0])?,
    )?;
    let resp1 = read(
        resp1,
        &cp,
        "resp1",
        Some(dtype),
        product(&[pvec_scratch.len])?,
    )?;
    let n_per_gene = read(
        n_per_gene,
        &cp,
        "n_per_gene",
        Some(Dtype::I32),
        product(&[0])?,
    )?;
    let k_per_gene = read(
        k_per_gene,
        &cp,
        "k_per_gene",
        Some(Dtype::I32),
        product(&[n_per_gene.len])?,
    )?;
    let cell_offsets = read(
        cell_offsets,
        &cp,
        "cell_offsets",
        Some(Dtype::I32),
        product(&[n_per_gene.len])?,
    )?;
    let feat_offsets = read(
        feat_offsets,
        &cp,
        "feat_offsets",
        Some(Dtype::I32),
        product(&[n_per_gene.len])?,
    )?;
    let active_genes = read(
        active_genes,
        &cp,
        "active_genes",
        Some(Dtype::I32),
        product(&[n_active])?,
    )?;
    let device = current_device(
        &cp,
        &[
            &dat,
            &nt_cells_mean,
            &pvec_scratch,
            &resp1,
            &n_per_gene,
            &k_per_gene,
            &cell_offsets,
            &feat_offsets,
            &active_genes,
        ],
    )?;
    let dat_offsets = read(
        dat_offsets,
        &cp,
        "dat_offsets",
        Some(Dtype::I64),
        n_per_gene.len,
    )?;
    let guide_sel = read(
        guide_sel,
        &cp,
        "guide_sel",
        Some(Dtype::Bool),
        pvec_scratch.len,
    )?;
    let nt_in_all = read(
        nt_in_all,
        &cp,
        "nt_in_all",
        Some(Dtype::Bool),
        pvec_scratch.len,
    )?;
    current_device(&cp, &[&dat_offsets, &guide_sel, &nt_in_all])?;
    disjoint(
        &[&pvec_scratch, &resp1],
        &[
            &dat,
            &dat_offsets,
            &n_per_gene,
            &k_per_gene,
            &cell_offsets,
            &feat_offsets,
            &nt_cells_mean,
            &guide_sel,
            &nt_in_all,
            &active_genes,
        ],
    )?;
    if max_k == 0 || n_active == 0 {
        return Ok(());
    }
    // A small number of genes needs the original full block even for wide
    // projections. The shared/global vector choice below budgets the actual
    // feature width; limiting this dispatch by width serializes that work in
    // one warp and can force eight unnecessary vectors into global memory.
    let block_em = pvec_scratch.len / n_per_gene.len.max(1) > 1024 || n_active <= 8;
    let vectors = if block_em { 1 } else { 8 };
    let requested_shared = max_k
        .checked_mul(dtype.size())
        .and_then(|n| n.checked_mul(vectors));
    // Leave room for static reduction storage within Turing's 48 KiB limit.
    // Wider vectors retain the existing global-workspace fallback.
    let shared_budget = 47 * 1024 - if block_em { 1024 * dtype.size() } else { 0 };
    let shared_bytes = requested_shared
        .filter(|&bytes| bytes <= shared_budget)
        .unwrap_or(0) as u32;
    let vec_obj = if shared_bytes == 0 {
        let kw = PyDict::new(py);
        kw.set_item("dtype", dtype.name())?;
        Some(cp.getattr("empty")?.call(((n_active, max_k),), Some(&kw))?)
    } else {
        None
    };
    let vec_pointer = if let Some(vec) = &vec_obj {
        read(
            vec,
            &cp,
            "projection workspace",
            Some(dtype),
            product(&[n_active, max_k])?,
        )?
        .pointer
    } else {
        0
    };
    let mut args = [
        Arg::Ptr(dat.pointer),
        Arg::Ptr(dat_offsets.pointer),
        Arg::Ptr(n_per_gene.pointer),
        Arg::Ptr(k_per_gene.pointer),
        Arg::Ptr(cell_offsets.pointer),
        Arg::Ptr(feat_offsets.pointer),
        Arg::Ptr(nt_cells_mean.pointer),
        Arg::Ptr(guide_sel.pointer),
        Arg::Ptr(nt_in_all.pointer),
        Arg::Ptr(active_genes.pointer),
        Arg::Ptr(pvec_scratch.pointer),
        Arg::Ptr(resp1.pointer),
        Arg::Ptr(vec_pointer),
        Arg::U64(n_active),
        Arg::U64(n_per_gene.len),
        Arg::U64(max_k),
        Arg::U64(dat.len),
        Arg::U64(pvec_scratch.len),
        Arg::U64(nt_cells_mean.len),
        Arg::U32(max_iter),
        scalar(dtype, tol),
        scalar(dtype, reg_covar),
    ];

    let stem = if block_em {
        "gmm_project_block"
    } else {
        "gmm_project"
    };
    let storage = if shared_bytes != 0 { "_shared" } else { "" };
    let suffix = if dtype == Dtype::F32 { "f32" } else { "f64" };
    let work = product(&[n_active, if block_em { 256 } else { 32 }])?;
    let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
    // SAFETY: validated extents and one shared direction vector per independent
    // gene/warp. The launch budget includes static reduction storage as well.
    unsafe {
        crate::runtime::launch_shared(
            device,
            &format!("{stem}{storage}_{suffix}"),
            (work.div_ceil(256).min(65535) as u32, 1, 1),
            (256, 1, 1),
            shared_bytes,
            stream,
            &mut pointers,
        )?;
    }
    // CuPy's memory pool associates temporary storage with this current stream,
    // keeping recycling ordered after the native kernel that consumes it.
    Ok(())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._gmm_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(e_step, &m)?)?;
    m.add_function(wrap_pyfunction!(e_step_cublas, &m)?)?;
    m.add_function(wrap_pyfunction!(m_step, &m)?)?;
    m.add_function(wrap_pyfunction!(precision_cholesky_full, &m)?)?;
    m.add_function(wrap_pyfunction!(spherical_gmm_fit_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(mixscape_project_em, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
