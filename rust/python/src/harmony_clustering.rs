//! Harmony's iterative clustering loop, with native device reductions and cuBLAS.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Dtype, current_device},
    harmony::*,
    runtime,
};
use pyo3::{exceptions::PyValueError, prelude::*};

fn objective(
    py: Python<'_>,
    cp: &Bound<'_, PyModule>,
    device: usize,
    dtype: Dtype,
    R: u64,
    sim: u64,
    O: u64,
    E: u64,
    theta: u64,
    out: u64,
    n: u64,
    k: u64,
    b: u64,
    sigma: f64,
    stabilized: bool,
    stream: usize,
) -> PyResult<f64> {
    zero(cp, out, itemsize(dtype), stream)?;
    launch(
        device,
        "harmony_kmeans",
        dtype,
        (n * k).min(colsum_multiprocessors(cp, device)? * 8 * 256),
        stream,
        &mut [Arg::Ptr(R), Arg::Ptr(sim), Arg::Ptr(out), Arg::U64(n * k)],
    )?;
    launch(
        device,
        "harmony_entropy",
        dtype,
        n * 32,
        stream,
        &mut [
            Arg::Ptr(R),
            Arg::Ptr(out),
            Arg::U64(n),
            Arg::U64(k),
            scalar(dtype, sigma),
        ],
    )?;
    launch(
        device,
        "harmony_diversity",
        dtype,
        b * k,
        stream,
        &mut [
            Arg::Ptr(O),
            Arg::Ptr(E),
            Arg::Ptr(theta),
            Arg::Ptr(out),
            Arg::U64(b),
            Arg::U64(k),
            Arg::U32(stabilized as u32),
            scalar(dtype, sigma),
        ],
    )?;
    let mut f = 0f32;
    let mut d = 0f64;
    let ptr = if dtype == Dtype::F32 {
        (&mut f as *mut f32) as usize
    } else {
        (&mut d as *mut f64) as usize
    };
    let rt = cp.getattr("cuda")?.getattr("runtime")?;
    rt.call_method1("memcpyAsync", (ptr, out, itemsize(dtype), 2, stream))?;
    rt.call_method1("streamSynchronize", (stream,))?;
    let _ = py;
    Ok(if dtype == Dtype::F32 { f as f64 } else { d })
}
#[pyfunction]
#[pyo3(signature=(R,*,similarities,O,E,theta,sigma,obj_scalar,n_cells,n_clusters,n_batches,stabilized,stream=0))]
pub fn compute_objective(
    py: Python<'_>,
    R: &Bound<'_, PyAny>,
    similarities: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    E: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
    sigma: f64,
    obj_scalar: &Bound<'_, PyAny>,
    n_cells: u64,
    n_clusters: u64,
    n_batches: u64,
    stabilized: bool,
    stream: usize,
) -> PyResult<f64> {
    let cp = py.import("cupy")?;
    let R = floating(R, &cp, "R", product(&[n_cells, n_clusters])?)?;
    let dtype = R.dtype;
    let s = read(
        similarities,
        &cp,
        "similarities",
        Some(dtype),
        n_cells * n_clusters,
    )?;
    let O = read(O, &cp, "O", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let E = read(E, &cp, "E", Some(dtype), n_batches * n_clusters)?;
    let t = read(theta, &cp, "theta", Some(dtype), n_batches)?;
    let out = read(obj_scalar, &cp, "obj_scalar", Some(dtype), 1)?;
    let device = current_device(&cp, &[&R, &s, &O, &E, &t, &out])?;
    for a in [&R, &s, &O, &E, &t] {
        out.require_disjoint(a)?;
    }
    objective(
        py,
        &cp,
        device,
        dtype,
        R.pointer,
        s.pointer,
        O.pointer,
        E.pointer,
        t.pointer,
        out.pointer,
        n_cells,
        n_clusters,
        n_batches,
        sigma,
        stabilized,
        stream,
    )
}
#[pyfunction]
#[pyo3(signature=(*,n_cells))]
pub fn get_cub_sort_temp_bytes(n_cells: u64) -> PyResult<u64> {
    if n_cells > i32::MAX as u64 {
        return Err(PyValueError::new_err(
            "n_cells exceeds int32 shuffle indices",
        ));
    }
    Ok(0)
}
#[pyfunction]
#[pyo3(signature=(Z_norm,*,R,E,O,Pr_b,cats,theta,joint_codes=None,marginal_joint_offsets=None,marginal_joint_indices=None,O_joint=None,Y,Y_norm,similarities,idx_list,idx_list_alt,sort_keys,sort_keys_alt,cub_temp,R_out_buffer,cats_in,joint_codes_in=None,R_in_sum,R_out_sum,penalty,obj_scalar,ones_vec,last_obj,n_cells,n_pcs,n_clusters,n_batches,n_covariates,n_joint_categories,block_size,colsum_algo,sigma,tol,max_iter,seed,stabilized,use_joint_scatter,stream=0,handle))]
pub fn clustering_loop(
    py: Python<'_>,
    Z_norm: &Bound<'_, PyAny>,
    R: &Bound<'_, PyAny>,
    E: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    Pr_b: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
    joint_codes: Option<&Bound<'_, PyAny>>,
    marginal_joint_offsets: Option<&Bound<'_, PyAny>>,
    marginal_joint_indices: Option<&Bound<'_, PyAny>>,
    O_joint: Option<&Bound<'_, PyAny>>,
    Y: &Bound<'_, PyAny>,
    Y_norm: &Bound<'_, PyAny>,
    similarities: &Bound<'_, PyAny>,
    idx_list: &Bound<'_, PyAny>,
    idx_list_alt: &Bound<'_, PyAny>,
    sort_keys: &Bound<'_, PyAny>,
    sort_keys_alt: &Bound<'_, PyAny>,
    cub_temp: &Bound<'_, PyAny>,
    R_out_buffer: &Bound<'_, PyAny>,
    cats_in: &Bound<'_, PyAny>,
    joint_codes_in: Option<&Bound<'_, PyAny>>,
    R_in_sum: &Bound<'_, PyAny>,
    R_out_sum: &Bound<'_, PyAny>,
    penalty: &Bound<'_, PyAny>,
    obj_scalar: &Bound<'_, PyAny>,
    ones_vec: &Bound<'_, PyAny>,
    last_obj: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_clusters: u64,
    n_batches: u64,
    n_covariates: u64,
    n_joint_categories: u64,
    block_size: u64,
    colsum_algo: i32,
    sigma: f64,
    tol: f64,
    max_iter: u32,
    seed: u32,
    stabilized: bool,
    use_joint_scatter: bool,
    stream: usize,
    handle: usize,
) -> PyResult<()> {
    if n_covariates == 0 || block_size == 0 || n_cells > i32::MAX as u64 {
        return Err(PyValueError::new_err(
            "covariates and block_size must be positive and cell count fit int32",
        ));
    }
    let cp = py.import("cupy")?;
    let dtype = floating(Z_norm, &cp, "Z_norm", product(&[n_cells, n_pcs])?)?.dtype;
    let sort_keys_obj = sort_keys;
    let Z_norm = read(
        Z_norm,
        &cp,
        "Z_norm",
        Some(dtype),
        product(&[n_cells, n_pcs])?,
    )?;
    let R = read(R, &cp, "R", Some(dtype), product(&[n_cells, n_clusters])?)?;
    let E = read(E, &cp, "E", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let O = read(O, &cp, "O", Some(dtype), product(&[n_batches, n_clusters])?)?;
    let Pr_b = read(Pr_b, &cp, "Pr_b", Some(dtype), product(&[n_batches])?)?;
    let theta = read(theta, &cp, "theta", Some(dtype), product(&[n_batches])?)?;
    let Y = read(Y, &cp, "Y", Some(dtype), product(&[n_clusters, n_pcs])?)?;
    let Y_norm = read(
        Y_norm,
        &cp,
        "Y_norm",
        Some(dtype),
        product(&[n_clusters, n_pcs])?,
    )?;
    let similarities = read(
        similarities,
        &cp,
        "similarities",
        Some(dtype),
        product(&[n_cells, n_clusters])?,
    )?;
    let R_out_buffer = read(
        R_out_buffer,
        &cp,
        "R_out_buffer",
        Some(dtype),
        product(&[block_size, n_clusters])?,
    )?;
    let R_in_sum = read(
        R_in_sum,
        &cp,
        "R_in_sum",
        Some(dtype),
        product(&[n_clusters])?,
    )?;
    let R_out_sum = read(
        R_out_sum,
        &cp,
        "R_out_sum",
        Some(dtype),
        product(&[n_clusters])?,
    )?;
    let penalty = read(
        penalty,
        &cp,
        "penalty",
        Some(dtype),
        product(&[n_batches, n_clusters])?,
    )?;
    let obj_scalar = read(obj_scalar, &cp, "obj_scalar", Some(dtype), product(&[1])?)?;
    let ones_vec = read(
        ones_vec,
        &cp,
        "ones_vec",
        Some(dtype),
        product(&[block_size])?,
    )?;
    let last_obj = read(last_obj, &cp, "last_obj", Some(dtype), product(&[1])?)?;
    let cats = read(
        cats,
        &cp,
        "cats",
        Some(Dtype::I32),
        product(&[n_cells, n_covariates])?,
    )?;
    let idx_list = read(
        idx_list,
        &cp,
        "idx_list",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let idx_list_alt = read(
        idx_list_alt,
        &cp,
        "idx_list_alt",
        Some(Dtype::I32),
        product(&[n_cells])?,
    )?;
    let cats_in = read(
        cats_in,
        &cp,
        "cats_in",
        Some(Dtype::I32),
        product(&[block_size, n_covariates])?,
    )?;
    let sort_keys = read(
        sort_keys,
        &cp,
        "sort_keys",
        Some(Dtype::U32),
        product(&[n_cells])?,
    )?;
    let sort_keys_alt = read(
        sort_keys_alt,
        &cp,
        "sort_keys_alt",
        Some(Dtype::U32),
        product(&[n_cells])?,
    )?;
    let cub_temp = read(cub_temp, &cp, "cub_temp", Some(Dtype::U8), product(&[0])?)?;
    let device = current_device(
        &cp,
        &[
            &Z_norm,
            &R,
            &E,
            &O,
            &Pr_b,
            &theta,
            &Y,
            &Y_norm,
            &similarities,
            &R_out_buffer,
            &R_in_sum,
            &R_out_sum,
            &penalty,
            &obj_scalar,
            &ones_vec,
            &last_obj,
            &cats,
            &idx_list,
            &idx_list_alt,
            &cats_in,
            &sort_keys,
            &sort_keys_alt,
            &cub_temp,
        ],
    )?;
    let joint_codes = if use_joint_scatter {
        Some(read(
            joint_codes
                .ok_or_else(|| PyValueError::new_err("joint scatter requires all joint arrays"))?,
            &cp,
            "joint_codes",
            Some(Dtype::I32),
            product(&[n_cells])?,
        )?)
    } else {
        None
    };
    let marginal_joint_offsets = if use_joint_scatter {
        Some(read(
            marginal_joint_offsets
                .ok_or_else(|| PyValueError::new_err("joint scatter requires all joint arrays"))?,
            &cp,
            "marginal_joint_offsets",
            Some(Dtype::I32),
            product(&[n_batches + 1])?,
        )?)
    } else {
        None
    };
    let marginal_joint_indices = if use_joint_scatter {
        Some(read(
            marginal_joint_indices
                .ok_or_else(|| PyValueError::new_err("joint scatter requires all joint arrays"))?,
            &cp,
            "marginal_joint_indices",
            Some(Dtype::I32),
            product(&[0])?,
        )?)
    } else {
        None
    };
    let O_joint = if use_joint_scatter {
        Some(read(
            O_joint
                .ok_or_else(|| PyValueError::new_err("joint scatter requires all joint arrays"))?,
            &cp,
            "O_joint",
            Some(dtype),
            product(&[n_joint_categories, n_clusters])?,
        )?)
    } else {
        None
    };
    let joint_codes_in = if use_joint_scatter {
        Some(read(
            joint_codes_in
                .ok_or_else(|| PyValueError::new_err("joint scatter requires all joint arrays"))?,
            &cp,
            "joint_codes_in",
            Some(Dtype::I32),
            product(&[block_size])?,
        )?)
    } else {
        None
    };
    if use_joint_scatter && n_joint_categories == 0 {
        return Err(PyValueError::new_err(
            "joint scatter requires at least one joint category",
        ));
    }
    for a in [
        joint_codes.as_ref(),
        marginal_joint_offsets.as_ref(),
        marginal_joint_indices.as_ref(),
        O_joint.as_ref(),
        joint_codes_in.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        current_device(&cp, &[a])?;
    }
    let mut writable = vec![
        &R,
        &E,
        &O,
        &Y,
        &Y_norm,
        &similarities,
        &idx_list,
        &idx_list_alt,
        &sort_keys,
        &sort_keys_alt,
        &R_out_buffer,
        &cats_in,
        &R_in_sum,
        &R_out_sum,
        &penalty,
        &obj_scalar,
        &last_obj,
    ];
    writable.extend(O_joint.iter());
    writable.extend(joint_codes_in.iter());
    let mut inputs = vec![&Z_norm, &Pr_b, &cats, &theta, &ones_vec];
    inputs.extend(joint_codes.iter());
    inputs.extend(marginal_joint_offsets.iter());
    inputs.extend(marginal_joint_indices.iter());
    disjoint(&writable, &inputs)?;
    let _stream_scope = crate::runtime::StreamScope::new(&cp, stream)?;
    let mut objectives: Vec<f64> = Vec::new();
    for iter in 0..max_iter {
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
            Z_norm.pointer,
            n_pcs,
            R.pointer,
            n_clusters,
            Y.pointer,
            n_pcs,
            0.,
        )?;
        launch_rows(
            device,
            "harmony_normalize",
            dtype,
            n_clusters,
            n_pcs,
            stream,
            &mut [
                Arg::Ptr(Y.pointer),
                Arg::Ptr(Y_norm.pointer),
                Arg::U64(n_clusters),
                Arg::U64(n_pcs),
                Arg::U32(1),
            ],
        )?;
        gemm(
            py,
            dtype,
            handle,
            stream,
            1,
            0,
            n_clusters,
            n_cells,
            n_pcs,
            Y_norm.pointer,
            n_pcs,
            Z_norm.pointer,
            n_pcs,
            similarities.pointer,
            n_clusters,
            0.,
        )?;
        let mut keys = sort_keys.pointer;
        let mut indices = idx_list.pointer;
        let mut n = n_cells;
        let mut hash_seed = seed.wrapping_add(iter);
        let mut args = [
            (&mut keys as *mut u64).cast(),
            (&mut indices as *mut u64).cast(),
            (&mut n as *mut u64).cast(),
            (&mut hash_seed as *mut u32).cast(),
        ];
        if n_cells > 0 {
            unsafe {
                runtime::launch(
                    device,
                    "harmony_shuffle",
                    (n_cells.div_ceil(256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    stream,
                    &mut args,
                )?;
            }
        }
        // CuPy's stable GPU sorting supplies the primitive formerly supplied by CUB.
        let order = sort_keys_obj
            .call_method0("argsort")?
            .call_method1("astype", ("int32",))?;
        let sorted_keys = sort_keys_obj.get_item(&order)?;
        let order_a = read(&order, &cp, "shuffle order", Some(Dtype::I32), n_cells)?;
        let sorted_a = read(&sorted_keys, &cp, "sorted keys", Some(Dtype::U32), n_cells)?;
        copy(
            &cp,
            idx_list_alt.pointer,
            order_a.pointer,
            n_cells * 4,
            stream,
        )?;
        copy(
            &cp,
            sort_keys_alt.pointer,
            sorted_a.pointer,
            n_cells * 4,
            stream,
        )?;
        let mut pos = 0;
        while pos < n_cells {
            let bs = block_size.min(n_cells - pos);
            let block_idx = idx_list_alt.pointer + pos * 4;
            launch(
                device,
                "harmony_rows",
                dtype,
                bs * n_clusters,
                stream,
                &mut [
                    Arg::Ptr(R.pointer),
                    Arg::Ptr(block_idx),
                    Arg::Ptr(R_out_buffer.pointer),
                    Arg::U64(bs),
                    Arg::U64(n_clusters),
                    Arg::U64(n_cells),
                    Arg::U32(0),
                ],
            )?;
            launch(
                device,
                "harmony_rows",
                Dtype::I32,
                bs * n_covariates,
                stream,
                &mut [
                    Arg::Ptr(cats.pointer),
                    Arg::Ptr(block_idx),
                    Arg::Ptr(cats_in.pointer),
                    Arg::U64(bs),
                    Arg::U64(n_covariates),
                    Arg::U64(n_cells),
                    Arg::U32(0),
                ],
            )?;
            if let (Some(j), Some(dst)) = (&joint_codes, &joint_codes_in) {
                launch(
                    device,
                    "harmony_rows",
                    Dtype::I32,
                    bs,
                    stream,
                    &mut [
                        Arg::Ptr(j.pointer),
                        Arg::Ptr(block_idx),
                        Arg::Ptr(dst.pointer),
                        Arg::U64(bs),
                        Arg::U64(1),
                        Arg::U64(n_cells),
                        Arg::U32(0),
                    ],
                )?;
            }
            launch_colsum(
                &cp,
                device,
                dtype,
                R_out_buffer.pointer,
                R_in_sum.pointer,
                bs,
                n_clusters,
                false,
                stream,
            )?;
            if let (Some(j), Some(jin), Some(off), Some(idx)) = (
                &O_joint,
                &joint_codes_in,
                &marginal_joint_offsets,
                &marginal_joint_indices,
            ) {
                launch(
                    device,
                    "harmony_scatter",
                    dtype,
                    bs * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(R_out_buffer.pointer),
                        Arg::Ptr(jin.pointer),
                        Arg::Ptr(j.pointer),
                        Arg::U64(bs),
                        Arg::U64(n_clusters),
                        Arg::U64(1),
                        Arg::U64(n_joint_categories),
                        Arg::I32(0),
                    ],
                )?;
                launch(
                    device,
                    "harmony_marginal",
                    dtype,
                    n_batches * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(j.pointer),
                        Arg::Ptr(off.pointer),
                        Arg::Ptr(idx.pointer),
                        Arg::Ptr(O.pointer),
                        Arg::U64(n_batches),
                        Arg::U64(n_clusters),
                        Arg::U64(n_joint_categories),
                        Arg::U64(idx.len),
                    ],
                )?;
            } else {
                launch(
                    device,
                    "harmony_scatter",
                    dtype,
                    bs * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(R_out_buffer.pointer),
                        Arg::Ptr(cats_in.pointer),
                        Arg::Ptr(O.pointer),
                        Arg::U64(bs),
                        Arg::U64(n_clusters),
                        Arg::U64(n_covariates),
                        Arg::U64(n_batches),
                        Arg::I32(0),
                    ],
                )?;
            }
            launch(
                device,
                "harmony_outer",
                dtype,
                n_batches * n_clusters,
                stream,
                &mut [
                    Arg::Ptr(E.pointer),
                    Arg::Ptr(Pr_b.pointer),
                    Arg::Ptr(R_in_sum.pointer),
                    Arg::U64(n_batches),
                    Arg::U64(n_clusters),
                    Arg::I32(0),
                ],
            )?;
            launch(
                device,
                "harmony_penalty",
                dtype,
                n_batches * n_clusters,
                stream,
                &mut [
                    Arg::Ptr(E.pointer),
                    Arg::Ptr(O.pointer),
                    Arg::Ptr(theta.pointer),
                    Arg::Ptr(penalty.pointer),
                    Arg::U64(n_batches),
                    Arg::U64(n_clusters),
                    Arg::U32(stabilized as u32),
                ],
            )?;
            launch_rows(
                device,
                crate::harmony::pen_kernel(n_covariates),
                dtype,
                bs,
                n_clusters,
                stream,
                &mut [
                    Arg::Ptr(similarities.pointer),
                    Arg::Ptr(penalty.pointer),
                    Arg::Ptr(cats_in.pointer),
                    Arg::Ptr(block_idx),
                    Arg::Ptr(R_out_buffer.pointer),
                    Arg::U64(bs),
                    Arg::U64(n_clusters),
                    Arg::U64(n_covariates),
                    Arg::U64(n_cells),
                    Arg::U64(n_batches),
                    scalar(dtype, -2. / sigma),
                ],
            )?;
            launch(
                device,
                "harmony_rows",
                dtype,
                bs * n_clusters,
                stream,
                &mut [
                    Arg::Ptr(R_out_buffer.pointer),
                    Arg::Ptr(block_idx),
                    Arg::Ptr(R.pointer),
                    Arg::U64(bs),
                    Arg::U64(n_clusters),
                    Arg::U64(n_cells),
                    Arg::U32(1),
                ],
            )?;
            launch_colsum(
                &cp,
                device,
                dtype,
                R_out_buffer.pointer,
                R_out_sum.pointer,
                bs,
                n_clusters,
                false,
                stream,
            )?;
            if let (Some(j), Some(jin), Some(off), Some(idx)) = (
                &O_joint,
                &joint_codes_in,
                &marginal_joint_offsets,
                &marginal_joint_indices,
            ) {
                launch(
                    device,
                    "harmony_scatter",
                    dtype,
                    bs * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(R_out_buffer.pointer),
                        Arg::Ptr(jin.pointer),
                        Arg::Ptr(j.pointer),
                        Arg::U64(bs),
                        Arg::U64(n_clusters),
                        Arg::U64(1),
                        Arg::U64(n_joint_categories),
                        Arg::I32(1),
                    ],
                )?;
                launch(
                    device,
                    "harmony_marginal",
                    dtype,
                    n_batches * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(j.pointer),
                        Arg::Ptr(off.pointer),
                        Arg::Ptr(idx.pointer),
                        Arg::Ptr(O.pointer),
                        Arg::U64(n_batches),
                        Arg::U64(n_clusters),
                        Arg::U64(n_joint_categories),
                        Arg::U64(idx.len),
                    ],
                )?;
            } else {
                launch(
                    device,
                    "harmony_scatter",
                    dtype,
                    bs * n_clusters,
                    stream,
                    &mut [
                        Arg::Ptr(R_out_buffer.pointer),
                        Arg::Ptr(cats_in.pointer),
                        Arg::Ptr(O.pointer),
                        Arg::U64(bs),
                        Arg::U64(n_clusters),
                        Arg::U64(n_covariates),
                        Arg::U64(n_batches),
                        Arg::I32(1),
                    ],
                )?;
            }
            launch(
                device,
                "harmony_outer",
                dtype,
                n_batches * n_clusters,
                stream,
                &mut [
                    Arg::Ptr(E.pointer),
                    Arg::Ptr(Pr_b.pointer),
                    Arg::Ptr(R_out_sum.pointer),
                    Arg::U64(n_batches),
                    Arg::U64(n_clusters),
                    Arg::I32(1),
                ],
            )?;
            pos += bs;
        }
        let obj = objective(
            py,
            &cp,
            device,
            dtype,
            R.pointer,
            similarities.pointer,
            O.pointer,
            E.pointer,
            theta.pointer,
            obj_scalar.pointer,
            n_cells,
            n_clusters,
            n_batches,
            sigma,
            stabilized,
            stream,
        )?;
        objectives.push(obj);
        if objectives.len() >= 4 {
            let len = objectives.len();
            let mut old = 0.;
            let mut new = 0.;
            for i in 0..3 {
                old += objectives[len - 2 - i];
                new += objectives[len - 1 - i];
                if dtype == Dtype::F32 {
                    old = (old as f32) as f64;
                    new = (new as f32) as f64;
                }
            }
            if old - new < tol * old.abs() {
                break;
            }
        }
    }
    if max_iter == 0 {
        zero(&cp, last_obj.pointer, itemsize(dtype), stream)?;
    } else {
        copy(
            &cp,
            last_obj.pointer,
            obj_scalar.pointer,
            itemsize(dtype),
            stream,
        )?;
    }
    let _ = colsum_algo;
    Ok(())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(
        parent.py(),
        "rapids_singlecell._cuda._harmony_clustering_cuda",
    )?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(clustering_loop, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_objective, &m)?)?;
    m.add_function(wrap_pyfunction!(get_cub_sort_temp_bytes, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
