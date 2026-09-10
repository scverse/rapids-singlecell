//! Owned host memberships and sparse segment counts for reference comparisons.
use super::*;

pub struct PopulationPlan {
    pub offsets: Vec<u64>,
    pub has_nan: bool,
}

pub struct Populations {
    labels: Vec<i32>,
    groups: usize,
    reference_count: usize,
    row_order: Vec<usize>,
}

impl Populations {
    pub fn new(
        py: Python<'_>,
        references: &Bound<'_, PyAny>,
        groups: &Bound<'_, PyAny>,
        group_offsets: &[usize],
        rows: usize,
    ) -> PyResult<Self> {
        let references = offsets(py, references)?;
        let members = offsets(py, groups)?;
        if group_offsets.is_empty()
            || group_offsets.first() != Some(&0)
            || group_offsets.last() != Some(&members.len())
            || group_offsets.windows(2).any(|v| v[1] < v[0])
            || group_offsets.len() > i32::MAX as usize
        {
            return Err(PyValueError::new_err("invalid host population offsets"));
        }
        let reference_count = references.len();
        host_parallel::run(py, rows, |_| {
            let mut labels = allocate::<i32>(rows)?;
            labels.fill(-1);
            let mut assign = |row: usize, group: i32| -> PyResult<()> {
                let target = labels.get_mut(row).ok_or_else(|| {
                    PyValueError::new_err("selected row IDs must be within the source matrix")
                })?;
                if *target != -1 {
                    return Err(PyValueError::new_err(
                        "reference and group row IDs must be unique and disjoint",
                    ));
                }
                *target = group;
                Ok(())
            };
            for &row in &references {
                assign(row, 0)?;
            }
            for (group, span) in group_offsets.windows(2).enumerate() {
                for &row in &members[span[0]..span[1]] {
                    assign(row, group as i32 + 1)?;
                }
            }
            let mut row_order = references;
            row_order.extend(members);
            Ok(Self {
                labels,
                groups: group_offsets.len() - 1,
                reference_count,
                row_order,
            })
        })?
    }

    /// The GPU packer and native segment planner must use the same snapshot.
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    pub fn reference_count(&self) -> usize {
        self.reference_count
    }

    pub fn row_order(&self) -> &[usize] {
        &self.row_order
    }

    #[cfg(test)]
    pub fn offsets(&self, py: Python<'_>, window: &CscWindow) -> PyResult<Vec<u64>> {
        self.plan(py, window).map(|plan| plan.offsets)
    }

    pub fn plan(&self, py: Python<'_>, window: &CscWindow) -> PyResult<PopulationPlan> {
        fn typed<I: Index>(
            populations: &Populations,
            indices: &[I],
            pointers: &[u64],
            is_nan: impl Fn(usize) -> bool + Sync,
            plan: host_parallel::Parallelism,
        ) -> PyResult<PopulationPlan> {
            use std::sync::atomic::{AtomicBool, Ordering};
            let cols = pointers.len() - 1;
            let groups = populations.groups + 1;
            let segments = cols
                .checked_mul(groups)
                .ok_or_else(|| PyValueError::new_err("host population count overflow"))?;
            let mut counts = allocate::<u64>(segments)?;
            let parts = plan.workers().min(indices.len().div_ceil(32_768).max(1));
            let columns_per_part = cols.div_ceil(parts).max(1);
            let mut columns: Vec<_> = counts.chunks_mut(groups * columns_per_part).collect();
            let invalid = AtomicBool::new(false);
            let has_nan = AtomicBool::new(false);
            // Each column owns all of its population counters. This gives
            // disjoint writes without native atomics or Python buffer borrows.
            plan.for_each_chunk(&mut columns, |first, columns| {
                for (part, block) in columns.iter_mut().enumerate() {
                    for (local, counts) in block.chunks_mut(groups).enumerate() {
                        let col = (first + part) * columns_per_part + local;
                        let begin = pointers[col] as usize;
                        for (local, &row) in indices[begin..pointers[col + 1] as usize]
                            .iter()
                            .enumerate()
                        {
                            let row = row.into();
                            if row < 0 || row as usize >= populations.labels.len() {
                                invalid.store(true, Ordering::Relaxed);
                                continue;
                            }
                            let group = populations.labels[row as usize];
                            if group >= 0 {
                                counts[group as usize] += 1;
                                if is_nan(begin + local) {
                                    has_nan.store(true, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            });
            if invalid.load(Ordering::Relaxed) {
                return Err(PyValueError::new_err("CSC row index out of bounds"));
            }
            let mut offsets = allocate::<u64>(
                segments
                    .checked_add(1)
                    .ok_or_else(|| PyValueError::new_err("host population count overflow"))?,
            )?;
            for group in 0..groups {
                for col in 0..cols {
                    let position = group * cols + col;
                    offsets[position + 1] = offsets[position] + counts[col * groups + group];
                }
            }
            Ok(PopulationPlan {
                offsets,
                has_nan: has_nan.load(Ordering::Relaxed),
            })
        }
        let items = window.data().2;
        host_parallel::run(py, if items < 32_768 { 0 } else { items }, |plan| {
            match (&window.indices, &window.data) {
                (Values::I32(indices), Values::F32(data)) => {
                    typed(self, indices, &window.offsets, |i| data[i].is_nan(), plan)
                }
                (Values::I64(indices), Values::F32(data)) => {
                    typed(self, indices, &window.offsets, |i| data[i].is_nan(), plan)
                }
                (Values::I32(indices), Values::F64(data)) => {
                    typed(self, indices, &window.offsets, |i| data[i].is_nan(), plan)
                }
                (Values::I64(indices), Values::F64(data)) => {
                    typed(self, indices, &window.offsets, |i| data[i].is_nan(), plan)
                }
                _ => unreachable!("CSC staging requires floating values and integer row IDs"),
            }
        })?
    }
}

/// Recover selected row IDs in compact-position order from a host row map.
pub fn mapped_rows(py: Python<'_>, source: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<i64>> {
    fn convert<I: Index>(values: Vec<I>, count: usize) -> PyResult<Vec<i64>> {
        let mut rows = allocate::<i64>(count)?;
        rows.fill(-1);
        for (row, position) in values.into_iter().enumerate() {
            let position = position.into();
            if position < 0 {
                continue;
            }
            let target = rows.get_mut(position as usize).ok_or_else(|| {
                PyValueError::new_err("row map position exceeds selected population")
            })?;
            if *target != -1 {
                return Err(PyValueError::new_err("row map positions must be unique"));
            }
            *target = i64::try_from(row).map_err(|_| PyValueError::new_err("row ID overflow"))?;
        }
        if rows.iter().any(|&row| row < 0) {
            return Err(PyValueError::new_err(
                "row map must contain every selected position",
            ));
        }
        Ok(rows)
    }
    let values = Values::snapshot(py, source)?;
    let items = values.view().2;
    host_parallel::run(py, items, |_| match values {
        Values::I32(v) => convert(v, count),
        Values::I64(v) => convert(v, count),
        _ => Err(PyTypeError::new_err(
            "host row maps require integer storage",
        )),
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn population_plan_detects_only_selected_nan_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let options = PyDict::new(py);
            let refs = np.call_method1("array", (vec![0], "int32"))?;
            let members = np.call_method1("array", (vec![2], "int32"))?;
            let populations = Populations::new(py, &refs, &members, &[0, 1], 3)?;
            for index_dtype in ["int32", "int64"] {
                let indices = np.call_method1("array", (vec![0, 1, 2], index_dtype))?;
                let pointers = np.call_method1("array", (vec![0, 3], "int64"))?;
                for dtype in ["float32", "float64"] {
                    options.set_item("dtype", dtype)?;
                    for (values, expected) in [
                        ([0.0, f64::NAN, f64::INFINITY], false),
                        ([f64::NAN, 0.0, 1.0], true),
                        ([0.0, 0.0, f64::NAN], true),
                    ] {
                        let data = np.call_method("array", (values.to_vec(),), Some(&options))?;
                        let window = CscWindow::from_arrays(py, &data, &indices, &pointers)?;
                        let plan = populations.plan(py, &window)?;
                        assert_eq!(plan.offsets, [0, 1, 2]);
                        assert_eq!(plan.has_nan, expected);
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn counts_preserve_stored_zeros_exclusions_and_empty_columns() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let options = PyDict::new(py);
            for index_dtype in ["int32", "int64"] {
                options.set_item("dtype", index_dtype)?;
                let references = np.call_method("array", (vec![2],), Some(&options))?;
                let members = np.call_method("array", (vec![0, 4, 1],), Some(&options))?;
                let populations = Populations::new(py, &references, &members, &[0, 2, 3], 5)?;
                assert_eq!(populations.labels, [1, 2, 0, -1, 1]);
                assert_eq!(populations.reference_count(), 1);
                let indices = np.call_method("array", (vec![0, 2, 3, 4, 1, 4],), Some(&options))?;
                let row_map = np.call_method("array", (vec![2, -3, 0, -1, 1],), Some(&options))?;
                assert_eq!(mapped_rows(py, &row_map, 3)?, [2, 4, 0]);
                for pointer_dtype in ["int32", "int64"] {
                    options.set_item("dtype", pointer_dtype)?;
                    let indptr = np.call_method("array", (vec![0, 3, 3, 6],), Some(&options))?;
                    for dtype in ["float32", "float64"] {
                        options.set_item("dtype", dtype)?;
                        let data = np.call_method(
                            "array",
                            (vec![0., 7., 0., -2., 0., 9.],),
                            Some(&options),
                        )?;
                        let window = CscWindow::from_arrays(py, &data, &indices, &indptr)?;
                        assert_eq!(
                            populations.offsets(py, &window)?,
                            [0, 1, 1, 1, 2, 2, 4, 4, 4, 5]
                        );
                    }
                }
                options.set_item("dtype", index_dtype)?;
                let duplicate = np.call_method("array", (vec![2],), Some(&options))?;
                assert!(Populations::new(py, &references, &duplicate, &[0, 1], 5).is_err());
                let invalid = np.call_method("array", (vec![5],), Some(&options))?;
                assert!(Populations::new(py, &invalid, &members, &[0, 2, 3], 5).is_err());
                let duplicate_map = np.call_method("array", (vec![0, 0],), Some(&options))?;
                assert!(mapped_rows(py, &duplicate_map, 2).is_err());
                assert!(mapped_rows(py, &row_map, 4).is_err());
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parallel_population_offsets_match_numpy_bincounts() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let globals = PyDict::new(py);
            py.run(pyo3::ffi::c_str!(
                "import numpy as np\nrows, cols, stored = 4097, 67, 1163\nindices = (np.arange(cols * stored, dtype=np.int64) * 31 % rows)\nindptr = np.arange(cols + 1, dtype=np.int32) * stored\ndata = np.zeros(len(indices), dtype=np.float64)\nrefs = np.arange(0, rows, 4, dtype=np.int64)\nmembers = np.r_[np.arange(1, rows, 4), np.arange(2, rows, 4)].astype(np.int64)\ncut = len(np.arange(1, rows, 4))\nlabels = np.full(rows, -1, dtype=np.int32)\nlabels[refs] = 0\nlabels[members[:cut]] = 1\nlabels[members[cut:]] = 2\ncounts = np.array([np.bincount(labels[v][labels[v] >= 0], minlength=3) for v in indices.reshape(cols, stored)], dtype=np.uint64).T.ravel()\nexpected = np.r_[np.uint64(0), np.cumsum(counts)]"
            ), Some(&globals), None)?;
            let get = |name| globals.get_item(name)?.ok_or_else(|| PyValueError::new_err("missing test data"));
            let references = get("refs")?;
            let members = get("members")?;
            let populations = Populations::new(py, &references, &members,
                &[0, get("cut")?.extract()?, members.len()?], 4097)?;
            // The GPU membership upload and native counters must both use the
            // validated snapshot even when caller storage changes afterward.
            references.call_method1("fill", (-1,))?;
            members.call_method1("fill", (-1,))?;
            let window = CscWindow::from_arrays(py, &get("data")?, &get("indices")?, &get("indptr")?)?;
            let expected = PyBuffer::<u64>::get(&get("expected")?)?.to_vec(py)?;
            for workers in [1, 3, 0] {
                let previous = host_parallel::set_worker_limit(workers);
                let result = populations.offsets(py, &window);
                host_parallel::set_worker_limit(previous);
                assert_eq!(result?, expected);
            }
            Ok(())
        }).unwrap();
    }
}
