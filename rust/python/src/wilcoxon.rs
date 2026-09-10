//! Exact Wilcoxon ranks using bounded native GPU batches and stable radix sorting.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    runtime::StreamScope,
    staging::BatchStreams,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

pub struct Stats<'a, 'py> {
    pub sums: &'a Bound<'py, PyAny>,
    pub nnz: Option<&'a Bound<'py, PyAny>>,
    pub total: Option<&'a Bound<'py, PyAny>>,
    pub total_nnz: Option<&'a Bound<'py, PyAny>>,
}
fn stats_outputs(
    cp: &Bound<'_, PyModule>,
    stats: &Stats<'_, '_>,
    groups: u64,
    cols: u64,
) -> PyResult<Vec<Array>> {
    let mut arrays = Vec::new();
    for (obj, n) in [
        (Some(stats.sums), groups * cols),
        (stats.nnz, groups * cols),
        (stats.total, cols),
        (stats.total_nnz, cols),
    ] {
        if let Some(obj) = obj {
            let a = read(obj, cp, "statistics output", Some(Dtype::F64), Layout::C)?;
            capacity(&a, n, "statistics output")?;
            current_device(cp, &[&a])?;
            arrays.push(a);
        }
    }
    Ok(arrays)
}
fn rank_outputs(
    cp: &Bound<'_, PyModule>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    compute: bool,
    matrix_ties: bool,
) -> PyResult<(Array, Option<Array>, usize, usize)> {
    let r = read(ranks, cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (g, c) = matrix(&r, "rank_sums")?;
    let t = if compute {
        let t = read(tie, cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        capacity(&t, if matrix_ties { r.len } else { c }, "tie_corr")?;
        t.require_disjoint(&r)?;
        Some(t)
    } else {
        None
    };
    current_device(cp, &[&r])?;
    if let Some(t) = &t {
        current_device(cp, &[t])?;
    }
    Ok((r, t, g as usize, c as usize))
}
/// Budget a small stream ring against the same total free-memory allowance.
/// Single-batch inputs retain a single workspace and avoid overlap overhead.
fn staging_plan(
    cp: &Bound<'_, PyModule>,
    rows: usize,
    requested: isize,
    cols: usize,
) -> PyResult<(usize, usize)> {
    let desired = if requested <= 0 {
        64
    } else {
        requested as usize
    };
    let slots = if cols > desired { 2 } else { 1 };
    let budget_rows = rows
        .max(64)
        .checked_mul(slots)
        .ok_or_else(|| PyValueError::new_err("staging size overflow"))?;
    Ok((slots, batch(cp, budget_rows, requested, cols)?))
}

/// A stream slot reuses its conversion storage after `BatchStreams::enter`
/// completes the previous batch. Keep this owner before the ring in drop order.
struct DenseCast<'py> {
    rows: usize,
    width: usize,
    dtype: Dtype,
    storage: Option<Bound<'py, PyAny>>,
}
impl<'py> DenseCast<'py> {
    fn new(rows: usize, width: usize, dtype: Dtype) -> Self {
        Self {
            rows,
            width,
            dtype,
            storage: None,
        }
    }

    fn convert(
        &mut self,
        cp: &Bound<'py, PyModule>,
        pipeline: &mut BatchStreams<'py>,
        slot: usize,
        source: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = floating(source, cp, "rank tile", Layout::Contiguous)?;
        let (rows, cols) = matrix(&input, "rank tile")?;
        if rows != self.rows as u64 || cols > self.width as u64 {
            return Err(PyValueError::new_err("rank tile exceeds cast workspace"));
        }
        current_device(cp, &[&input])?;
        // A contiguous matrix with at most one row/column has both layouts.
        if input.dtype == self.dtype && (!input.c_contiguous || rows <= 1 || cols <= 1) {
            return Ok(source.clone());
        }
        if self.storage.is_none() {
            self.storage =
                Some(pipeline.allocate(slot, &[self.rows, self.width], self.dtype, "F")?);
        }
        let destination = window(self.storage.as_ref().unwrap(), 0, cols as usize)?;
        // The caller has retained the source before the first asynchronous use;
        // the cache and allocation helper both own the target before copyto.
        cp.call_method1(pyo3::intern!(cp.py(), "copyto"), (&destination, source))?;
        Ok(destination)
    }
}

/// Accumulate ranks into full-width outputs while staging only a column batch.
pub fn ovr<'py>(
    cp: &Bound<'py, PyModule>,
    rows: usize,
    codes: &Bound<'py, PyAny>,
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    compute: bool,
    requested: isize,
    statistics: Option<Stats<'_, 'py>>,
    mut tile: impl FnMut(usize, usize) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let s = stream(cp)?;
    let (r, t, groups, cols) = rank_outputs(cp, ranks, tie, compute, false)?;
    let c = vector(codes, cp, "group_codes", Some(Dtype::I32))?;
    if c.len != rows as u64 {
        return Err(PyValueError::new_err(
            "group_codes length must equal input rows",
        ));
    }
    r.require_disjoint(&c)?;
    if let Some(t) = &t {
        t.require_disjoint(&c)?;
    }
    current_device(cp, &[&r, &c])?;
    let stats_arrays = if let Some(st) = &statistics {
        stats_outputs(cp, st, groups as u64, cols as u64)?
    } else {
        Vec::new()
    };
    let mut outputs = vec![&r];
    outputs.extend(t.iter());
    outputs.extend(stats_arrays.iter());
    crate::harmony::disjoint(&outputs, &[&c])?;
    // The native API treats an empty whole population as a validated no-op.
    if rows == 0 || cols == 0 || groups == 0 {
        return Ok(());
    }
    for output in &outputs {
        zero(cp, output, s)?;
    }
    let host = statistics.is_some();
    let (slots, width) = staging_plan(cp, rows, requested, cols)?;
    let mut sorters = (0..slots)
        .map(|_| crate::rank_sort::Workspace::with_compact_indices(cp, rows, width, Dtype::F32))
        .collect::<PyResult<Vec<_>>>()?;
    let mut casts = (0..slots)
        .map(|_| DenseCast::new(rows, width, Dtype::F32))
        .collect::<Vec<_>>();
    // Declare the ring after scratch so its Drop synchronizes before workspace
    // allocations can be released on either normal return or an error.
    let mut pipeline = if slots > 1 {
        BatchStreams::new(cp, slots)?
    } else {
        BatchStreams::caller(cp)?
    };
    for (batch_index, start) in (0..cols).step_by(width).enumerate() {
        let slot = batch_index % slots;
        let _scope = pipeline.enter(slot)?;
        let s = stream(cp)?;
        let sorter = &mut sorters[slot];
        let stop = (start + width).min(cols);
        let original = tile(start, stop)?;
        let original = if host {
            // Keep native layout for statistics; ranking combines any required
            // transpose with its float32 cast in the single conversion below.
            pipeline.upload_dense_order(slot, 0, &original, false)?
        } else {
            original
        };
        pipeline.retain(slot, original.clone());
        let x = if host {
            casts[slot].convert(cp, &mut pipeline, slot, &original)?
        } else {
            original.clone()
        };
        pipeline.retain(slot, x.clone());
        let xa = floating(&x, cp, "block", Layout::F)?;
        if xa.shape != [rows, stop - start] {
            return Err(PyValueError::new_err("rank tile has incorrect shape"));
        }
        let order = sorter.sort(cp, &x)?;
        let oa = read(&order, cp, "sorted indices", Some(Dtype::U32), Layout::F)?;
        let values = if rows <= 1 {
            x.clone()
        } else {
            sorter.values(cp, rows, stop - start)?
        };
        let va = read(&values, cp, "sorted values", Some(Dtype::F32), Layout::F)?;
        let rank_threads = rows.div_ceil(32).clamp(1, 16) as u32 * 32;
        current_device(cp, &[&xa, &oa, &c, &r])?;
        // Match the original shared reduction tier, reserving 32 f64 warp
        // partials within the portable 48 KiB limit; larger groups use globals.
        let rank_shared_bytes = if groups <= (48 * 1024 - 32 * 8) / 8 {
            groups as u32 * 8
        } else {
            0
        };
        crate::rank_sort::checked_launch_shared(
            xa.device,
            "rank_ovr_runs",
            ((stop - start) * rank_threads as usize) as u64,
            rank_threads,
            rank_shared_bytes,
            s,
            &mut [
                Arg::P(va.pointer),
                Arg::P(oa.pointer),
                Arg::P(c.pointer),
                Arg::P(r.pointer),
                Arg::P(t.as_ref().map_or(0, |a| a.pointer)),
                Arg::N(rows as u64),
                Arg::N((stop - start) as u64),
                Arg::N(groups as u64),
                Arg::N(cols as u64),
                Arg::N(start as u64),
            ],
        )?;
        if let Some(st) = &statistics {
            stats(
                cp,
                &original,
                codes,
                None,
                Some(st.sums),
                None,
                st.nnz,
                st.total,
                st.total_nnz,
                cols as u64,
                start as u64,
                s,
                true,
            )?;
        }
    }
    pipeline.finish()?;
    // Public streaming entry points historically complete before returning.
    sync(cp)
}

pub fn offsets(
    cp: &Bound<'_, PyModule>,
    object: &Bound<'_, PyAny>,
    rows: usize,
) -> PyResult<Vec<usize>> {
    let numpy = cp.py().import("numpy")?;
    let host = if object.is_instance(&cp.getattr("ndarray")?)? {
        cp.call_method1("asnumpy", (object,))?
    } else {
        object.clone()
    };
    if !host.is_instance(&numpy.getattr("ndarray")?)?
        || host.getattr("ndim")?.extract::<usize>()? != 1
    {
        return Err(PyTypeError::new_err(
            "grp_offsets must be a one-dimensional integer array",
        ));
    }
    let dtype: String = host.getattr("dtype")?.getattr("name")?.extract()?;
    if dtype != "int32" && dtype != "int64" {
        return Err(PyTypeError::new_err(
            "grp_offsets must have dtype int32 or int64",
        ));
    }
    let v = host.call_method0("tolist")?.extract::<Vec<usize>>()?;
    if v.is_empty() || v[0] != 0 || *v.last().unwrap() != rows || v.windows(2).any(|w| w[1] < w[0])
    {
        return Err(PyValueError::new_err(
            "grp_offsets must partition all group rows in sorted order",
        ));
    }
    Ok(v)
}

/// Tier selection is immutable across column batches and reusable with a
/// cached sorted reference. Row offsets are validated by the entry point.
pub(crate) struct OvoTierPlan {
    groups: usize,
    rows: usize,
    has_medium: bool,
    has_large: bool,
    pub(crate) huge_groups: Vec<usize>,
}
impl OvoTierPlan {
    pub(crate) fn new(offsets: &[usize], compute: bool) -> Self {
        Self {
            groups: offsets.len().saturating_sub(1),
            rows: offsets.last().copied().unwrap_or(0),
            has_medium: compute && offsets.windows(2).any(|o| o[1] - o[0] <= 512),
            has_large: compute
                && offsets
                    .windows(2)
                    .any(|o| (513..=2500).contains(&(o[1] - o[0]))),
            huge_groups: if compute {
                offsets
                    .windows(2)
                    .enumerate()
                    .filter_map(|(g, o)| (o[1] - o[0] > 2500).then_some(g))
                    .collect()
            } else {
                Vec::new()
            },
        }
    }
}

/// Launch all dense OVO group tiers against an already sorted reference.
/// Reference and tie buffers may be immutable full-cache column views. The
/// caller validates device/dtype/layout and retains every owner until stream
/// completion. Column bounds and the validated plan's population size are checked
/// here before dispatch.
pub(crate) fn launch_ovo_tiers<'py>(
    cp: &Bound<'py, PyModule>,
    ra: &Array,
    group: &Bound<'py, PyAny>,
    ga: &Array,
    oa: &Array,
    plan: &OvoTierPlan,
    reference_ties: Option<&Array>,
    r: &Array,
    ties: Option<&Array>,
    cols: usize,
    start: usize,
    huge_sort: Option<&mut crate::rank_sort::Segmented<'py>>,
    s: usize,
) -> PyResult<()> {
    let (nref, block_cols) = matrix(ra, "sorted reference")?;
    let (ngroups_rows, group_cols) = matrix(ga, "group data")?;
    let groups = plan.groups;
    if group_cols != block_cols
        || ngroups_rows != plan.rows as u64
        || start
            .checked_add(block_cols as usize)
            .is_none_or(|end| end > cols)
    {
        return Err(PyValueError::new_err(
            "OVO tier buffers have inconsistent dimensions",
        ));
    }
    if reference_ties.is_some() && ties.is_none() {
        return Err(PyValueError::new_err(
            "OVO tie output is required with a reference tie base",
        ));
    }
    if !plan.huge_groups.is_empty() && huge_sort.is_none() {
        return Err(PyValueError::new_err(
            "huge OVO groups require segmented sort scratch",
        ));
    }
    if let Some(ta) = reference_ties {
        for (enabled, name) in [
            (plan.has_medium, "rank_ovo_medium"),
            (plan.has_large, "rank_ovo_large"),
        ] {
            if enabled {
                crate::rank_sort::checked_launch_threads(
                    ra.device,
                    name,
                    product(&[
                        groups as u64,
                        block_cols,
                        if name == "rank_ovo_large" { 1024 } else { 256 },
                    ])?,
                    if name == "rank_ovo_large" { 1024 } else { 256 },
                    s,
                    &mut [
                        Arg::P(ra.pointer),
                        Arg::P(ga.pointer),
                        Arg::P(oa.pointer),
                        Arg::P(ta.pointer),
                        Arg::P(r.pointer),
                        Arg::P(ties.unwrap().pointer),
                        Arg::N(nref),
                        Arg::N(ngroups_rows),
                        Arg::N(block_cols),
                        Arg::N(groups as u64),
                        Arg::N(cols as u64),
                        Arg::N(start as u64),
                    ],
                )?;
            }
        }
        if let Some(group_sort) = huge_sort {
            let sorted = group_sort.sort(cp, group)?;
            let sorted = floating(&sorted, cp, "sorted groups", Layout::F)?;
            crate::rank_sort::checked_launch(
                ra.device,
                "rank_ovo_huge_segments",
                product(&[groups as u64, block_cols, 256])?,
                s,
                &mut [
                    Arg::P(ra.pointer),
                    Arg::P(sorted.pointer),
                    Arg::P(oa.pointer),
                    Arg::P(ta.pointer),
                    Arg::P(r.pointer),
                    Arg::P(ties.unwrap().pointer),
                    Arg::N(nref),
                    Arg::N(ngroups_rows),
                    Arg::N(block_cols),
                    Arg::N(groups as u64),
                    Arg::N(cols as u64),
                    Arg::N(start as u64),
                ],
            )?;
        }
    } else {
        crate::rank_sort::checked_launch(
            ra.device,
            "rank_ovo_unsorted",
            product(&[groups as u64, block_cols, 256])?,
            s,
            &mut [
                Arg::P(ra.pointer),
                Arg::P(ga.pointer),
                Arg::P(oa.pointer),
                Arg::P(r.pointer),
                Arg::N(nref),
                Arg::N(ngroups_rows),
                Arg::N(block_cols),
                Arg::N(groups as u64),
                Arg::N(cols as u64),
                Arg::N(start as u64),
            ],
        )?;
    }
    Ok(())
}

struct OvoScratch<'py> {
    reference_sort: crate::rank_sort::Workspace<'py>,
    group_sort: Option<crate::rank_sort::Segmented<'py>>,
    _tie_base: Option<Bound<'py, PyAny>>,
    tie_array: Option<Array>,
    reference_f32: DenseCast<'py>,
    group_f32: DenseCast<'py>,
    reference_f64: DenseCast<'py>,
    group_f64: DenseCast<'py>,
}

/// Rank each group against a reference sorted once per column batch.
pub fn ovo<'py>(
    cp: &Bound<'py, PyModule>,
    nref: usize,
    offsets: &[usize],
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    compute: bool,
    requested: isize,
    statistics: Option<Stats<'_, 'py>>,
    mut tiles: impl FnMut(usize, usize) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<()> {
    let s = stream(cp)?;
    let (r, t, groups, cols) = rank_outputs(cp, ranks, tie, compute, true)?;
    if offsets.len() != groups + 1 {
        return Err(PyValueError::new_err(
            "grp_offsets must have n_groups + 1 entries",
        ));
    }
    if let Some(t) = &t
        && t.shape != [groups, cols]
    {
        return Err(PyValueError::new_err(
            "tie_corr must have shape (n_groups, n_cols)",
        ));
    }
    let stats_arrays = if let Some(st) = &statistics {
        stats_outputs(cp, st, (groups + 1) as u64, cols as u64)?
    } else {
        Vec::new()
    };
    let mut outputs = vec![&r];
    outputs.extend(t.iter());
    outputs.extend(stats_arrays.iter());
    crate::harmony::disjoint(&outputs, &[])?;
    let ngroups_rows = offsets.last().copied().unwrap_or(0);
    // Empty individual groups still rank against the reference. An empty
    // entire input population preserves every caller-provided output instead.
    if nref == 0 || ngroups_rows == 0 || cols == 0 || groups == 0 {
        return Ok(());
    }
    for output in &outputs {
        zero(cp, output, s)?;
    }
    let rows = nref
        .checked_add(ngroups_rows)
        .ok_or_else(|| PyValueError::new_err("row count overflow"))?;
    let host = statistics.is_some();
    let (slots, width) = staging_plan(cp, rows, requested, cols)?;
    let width = width.min(65_535);
    let kw = PyDict::new(cp.py());
    kw.set_item("dtype", "uint64")?;
    let device_offsets = cp
        .getattr("asarray")?
        .call((offsets.to_vec(),), Some(&kw))?;
    let oa = vector(&device_offsets, cp, "group offsets", Some(Dtype::U64))?;
    let plan = OvoTierPlan::new(offsets, compute);
    let mut scratch = (0..slots)
        .map(|_| {
            let reference_sort =
                crate::rank_sort::Workspace::new(cp, nref, width, Dtype::F32, false)?;
            let group_sort = if plan.huge_groups.is_empty() {
                None
            } else {
                Some(crate::rank_sort::Segmented::new(
                    cp,
                    offsets,
                    &plan.huge_groups,
                    width,
                )?)
            };
            let tie_base = if compute {
                Some(empty(cp, &[width], "float64", "C", false)?)
            } else {
                None
            };
            let tie_array = if let Some(tie_base) = &tie_base {
                Some(vector(
                    tie_base,
                    cp,
                    "reference tie base",
                    Some(Dtype::F64),
                )?)
            } else {
                None
            };
            Ok(OvoScratch {
                reference_sort,
                group_sort,
                _tie_base: tie_base,
                tie_array,
                reference_f32: DenseCast::new(nref, width, Dtype::F32),
                group_f32: DenseCast::new(ngroups_rows, width, Dtype::F32),
                reference_f64: DenseCast::new(nref, width, Dtype::F64),
                group_f64: DenseCast::new(ngroups_rows, width, Dtype::F64),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut pipeline = if slots > 1 {
        BatchStreams::new(cp, slots)?
    } else {
        BatchStreams::caller(cp)?
    };
    for (batch_index, start) in (0..cols).step_by(width).enumerate() {
        let slot = batch_index % slots;
        let _scope = pipeline.enter(slot)?;
        let s = stream(cp)?;
        let active = &mut scratch[slot];
        let stop = (start + width).min(cols);
        let block_cols = stop - start;
        let (ref_original, grp_original) = tiles(start, stop)?;
        let (ref_original, grp_original) = if host {
            (
                pipeline.upload_dense(slot, 0, &ref_original)?,
                pipeline.upload_dense(slot, 1, &grp_original)?,
            )
        } else {
            (ref_original, grp_original)
        };
        pipeline.retain(slot, ref_original.clone());
        pipeline.retain(slot, grp_original.clone());
        let ref_float = if host {
            active
                .reference_f32
                .convert(cp, &mut pipeline, slot, &ref_original)?
        } else {
            ref_original.clone()
        };
        pipeline.retain(slot, ref_float.clone());
        let reference = active.reference_sort.sort(cp, &ref_float)?;
        let ra = floating(&reference, cp, "ref_data", Layout::F)?;
        let group = if host {
            active
                .group_f32
                .convert(cp, &mut pipeline, slot, &grp_original)?
        } else {
            grp_original.clone()
        };
        pipeline.retain(slot, group.clone());
        let ga = floating(&group, cp, "grp_data", Layout::F)?;
        if ra.shape != [nref, block_cols] || ga.shape != [ngroups_rows, block_cols] {
            return Err(PyValueError::new_err("OVO input tile has incorrect shape"));
        }
        current_device(cp, &[&ra, &ga, &r, &oa])?;
        if let Some(ta) = &active.tie_array {
            crate::rank_sort::checked_launch(
                ra.device,
                "rank_ovo_ref_ties",
                (block_cols * 256) as u64,
                s,
                &mut [
                    Arg::P(ra.pointer),
                    Arg::P(ta.pointer),
                    Arg::N(nref as u64),
                    Arg::N(block_cols as u64),
                ],
            )?;
        }
        launch_ovo_tiers(
            cp,
            &ra,
            &group,
            &ga,
            &oa,
            &plan,
            active.tie_array.as_ref(),
            &r,
            t.as_ref(),
            cols,
            start,
            active.group_sort.as_mut(),
            s,
        )?;
        if let Some(st) = &statistics {
            let group_data = active
                .group_f64
                .convert(cp, &mut pipeline, slot, &grp_original)?;
            pipeline.retain(slot, group_data.clone());
            let reference_data =
                active
                    .reference_f64
                    .convert(cp, &mut pipeline, slot, &ref_original)?;
            pipeline.retain(slot, reference_data.clone());
            let gs = floating(&group_data, cp, "group statistics data", Layout::F)?;
            let rs = floating(&reference_data, cp, "reference statistics data", Layout::F)?;
            let sums = read(st.sums, cp, "group_sums", Some(Dtype::F64), Layout::C)?;
            let nnz = if let Some(nnz) = st.nnz {
                Some(read(nnz, cp, "group_nnz", Some(Dtype::F64), Layout::C)?)
            } else {
                None
            };
            crate::rank_sort::checked_launch(
                ra.device,
                "rank_ovo_stats",
                product(&[(groups + 1) as u64, block_cols as u64, 256])?,
                s,
                &mut [
                    Arg::P(rs.pointer),
                    Arg::P(gs.pointer),
                    Arg::P(oa.pointer),
                    Arg::P(sums.pointer),
                    Arg::P(nnz.as_ref().map_or(0, |a| a.pointer)),
                    Arg::N(nref as u64),
                    Arg::N(ngroups_rows as u64),
                    Arg::N(block_cols as u64),
                    Arg::N(groups as u64),
                    Arg::N(cols as u64),
                    Arg::N(start as u64),
                ],
            )?;
        }
    }
    pipeline.finish()?;
    sync(cp)
}
pub fn host_dense(object: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let np = object.py().import("numpy")?;
    if !object.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("X must be a NumPy array"));
    }
    let dims = shape(object)?;
    let dtype: String = object.getattr("dtype")?.getattr("name")?.extract()?;
    let f = object.getattr("flags")?;
    if dims.len() != 2
        || !(f.getattr("c_contiguous")?.extract::<bool>()?
            || f.getattr("f_contiguous")?.extract::<bool>()?)
    {
        return Err(PyValueError::new_err(
            "X must be a contiguous two-dimensional array",
        ));
    }
    if dtype != "float32" && dtype != "float64" {
        return Err(PyTypeError::new_err("X must have dtype float32 or float64"));
    }
    Ok((dims[0], dims[1]))
}
#[pyfunction]
#[pyo3(signature=(block,group_codes,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64,stream=0))]
pub fn ovr_rank_dense_streaming(
    py: Python<'_>,
    block: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let x = read(block, &cp, "block", Some(Dtype::F32), Layout::F)?;
    let (rows, cols) = matrix(&x, "block")?;
    let r = read(rank_sums, &cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (_, out_cols) = matrix(&r, "rank_sums")?;
    if cols != out_cols {
        return Err(PyValueError::new_err(
            "rank output columns must match input",
        ));
    }
    r.require_disjoint(&x)?;
    current_device(&cp, &[&x, &r])?;
    if compute_tie_corr {
        let t = read(tie_corr, &cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        t.require_disjoint(&x)?;
    }

    let _scope = StreamScope::new(&cp, stream)?;
    ovr(
        &cp,
        rows as usize,
        group_codes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        None,
        |a, b| window(block, a, b),
    )
}
#[pyfunction]
#[pyo3(signature=(X,group_codes,rank_sums,tie_corr,group_sums,group_nnz,total_sums,total_nnz,*,compute_tie_corr,compute_nnz,compute_totals,col_start=0,col_stop=-1,sub_batch_cols=64), text_signature="(X,group_codes,rank_sums,tie_corr,group_sums,group_nnz,total_sums,total_nnz,*,compute_tie_corr,compute_nnz,compute_totals,col_start=0,col_stop=-1,sub_batch_cols=64)")]
pub fn ovr_rank_dense_host_streaming(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    group_sums: &Bound<'_, PyAny>,
    group_nnz: &Bound<'_, PyAny>,
    total_sums: &Bound<'_, PyAny>,
    total_nnz: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (rows, cols) = host_dense(X)?;
    let (start, stop) = range(col_start, col_stop, cols)?;
    if shape(rank_sums)?.get(1) != Some(&(stop - start)) {
        return Err(PyValueError::new_err(
            "rank output must match column window",
        ));
    }
    let st = Stats {
        sums: group_sums,
        nnz: compute_nnz.then_some(group_nnz),
        total: compute_totals.then_some(total_sums),
        total_nnz: (compute_totals && compute_nnz).then_some(total_nnz),
    };
    ovr(
        &cp,
        rows,
        group_codes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        Some(st),
        |a, b| window(X, start + a, start + b),
    )
}
#[pyfunction]
#[pyo3(signature=(ref_data,grp_data,grp_offsets,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64,stream=0))]
pub fn ovo_rank_dense_tiered_unsorted_ref(
    py: Python<'_>,
    ref_data: &Bound<'_, PyAny>,
    grp_data: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let ra = read(ref_data, &cp, "ref_data", Some(Dtype::F32), Layout::F)?;
    let ga = read(grp_data, &cp, "grp_data", Some(Dtype::F32), Layout::F)?;
    let (rows, cols) = matrix(&ra, "ref_data")?;
    let (ng, gc) = matrix(&ga, "grp_data")?;
    let r = read(rank_sums, &cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (_, rc) = matrix(&r, "rank_sums")?;
    if cols != gc || cols != rc {
        return Err(PyValueError::new_err(
            "all dense rank buffers must have matching columns",
        ));
    }
    r.require_disjoint(&ra)?;
    r.require_disjoint(&ga)?;
    current_device(&cp, &[&ra, &ga, &r])?;
    if compute_tie_corr {
        let t = read(tie_corr, &cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        t.require_disjoint(&ra)?;
        t.require_disjoint(&ga)?;
    }

    let _scope = StreamScope::new(&cp, stream)?;
    let off = offsets(&cp, grp_offsets, ng as usize)?;
    ovo(
        &cp,
        rows as usize,
        &off,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        None,
        |a, b| Ok((window(ref_data, a, b)?, window(grp_data, a, b)?)),
    )
}
/// Validate host gather indices before rank/statistics outputs are cleared.
fn host_row_ids(ids: &Bound<'_, PyAny>, rows: usize) -> PyResult<usize> {
    let np = ids.py().import("numpy")?;
    if !ids.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("row ids must be NumPy arrays"));
    }
    let dims = shape(ids)?;
    let dtype: String = ids.getattr("dtype")?.getattr("name")?.extract()?;
    if dims.len() != 1 || (dtype != "int32" && dtype != "int64") {
        return Err(PyTypeError::new_err(
            "row ids must be one-dimensional int32 or int64 arrays",
        ));
    }
    if dims[0] != 0 {
        let low = ids.call_method0("min")?.extract::<i64>()?;
        let high = ids.call_method0("max")?.extract::<i64>()?;
        if low < 0 || high as u64 >= rows as u64 {
            return Err(PyValueError::new_err(
                "row ids must be within the input row range",
            ));
        }
    }
    Ok(dims[0])
}
#[pyfunction]
#[pyo3(signature=(X,ref_row_ids,grp_row_ids,grp_offsets,rank_sums,tie_corr,group_sums,group_nnz,*,compute_tie_corr,compute_nnz,col_start=0,col_stop=-1,sub_batch_cols=64), text_signature="(X,ref_row_ids,grp_row_ids,grp_offsets,rank_sums,tie_corr,group_sums,group_nnz,*,compute_tie_corr,compute_nnz,col_start=0,col_stop=-1,sub_batch_cols=64)")]
pub fn ovo_rank_dense_host_streaming(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    ref_row_ids: &Bound<'_, PyAny>,
    grp_row_ids: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    group_sums: &Bound<'_, PyAny>,
    group_nnz: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (rows, cols) = host_dense(X)?;
    let (start, stop) = range(col_start, col_stop, cols)?;
    let ref_rows = host_row_ids(ref_row_ids, rows)?;
    let group_rows = host_row_ids(grp_row_ids, rows)?;
    let off = offsets(&cp, grp_offsets, group_rows)?;
    if shape(rank_sums)?.get(1) != Some(&(stop - start)) {
        return Err(PyValueError::new_err(
            "rank output must match column window",
        ));
    }
    let st = Stats {
        sums: group_sums,
        nnz: compute_nnz.then_some(group_nnz),
        total: None,
        total_nnz: None,
    };
    ovo(
        &cp,
        ref_rows,
        &off,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        Some(st),
        |a, b| {
            let columns = slice(py, start + a, start + b);
            Ok((
                X.get_item((ref_row_ids, columns.clone()))?,
                X.get_item((grp_row_ids, columns))?,
            ))
        },
    )
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._wilcoxon_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(
        crate::rank_stream::_set_host_worker_limit,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(ovr_rank_dense_streaming, &m)?)?;
    m.add_function(wrap_pyfunction!(ovr_rank_dense_host_streaming, &m)?)?;
    m.add_function(wrap_pyfunction!(ovo_rank_dense_tiered_unsorted_ref, &m)?)?;
    m.add_function(wrap_pyfunction!(ovo_rank_dense_host_streaming, &m)?)?;
    parent.add_submodule(&m)
}
