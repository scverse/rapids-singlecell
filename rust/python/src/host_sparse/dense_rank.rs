//! Owned, row-grouped CSR staging for the bounded dense-reference rank tier.
use super::*;

enum Columns {
    Narrow(Vec<u16>),
    Wide(Vec<u32>),
}

pub struct CsrRankPack {
    data: Values,
    columns: Columns,
    offsets: Vec<u64>,
    pub rows: usize,
    pub cols: usize,
    pub nonnegative_finite: bool,
}

fn primitive_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: only initialized primitive integer slices call this private
    // function. The returned bytes borrow their owning native vector.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl CsrRankPack {
    pub fn data(&self) -> (&[u8], Dtype, usize) {
        self.data.view()
    }
    pub fn columns(&self) -> &[u8] {
        match &self.columns {
            Columns::Narrow(values) => primitive_bytes(values),
            Columns::Wide(values) => primitive_bytes(values),
        }
    }
    pub fn narrow(&self) -> bool {
        matches!(self.columns, Columns::Narrow(_))
    }
    pub fn pointers(&self) -> &[u8] {
        primitive_bytes(&self.offsets)
    }
    pub fn bytes(&self) -> usize {
        self.data().0.len() + self.columns().len() + self.pointers().len()
    }
}

impl CsrSpans {
    /// Preserve only selected row spans in reference-then-group order. The
    /// existing sparse implementation remains the fallback for duplicate
    /// columns, oversized native snapshots, or unsupported dense dimensions.
    pub fn dense_rank_pack(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        populations: &Populations,
        first: usize,
        stop: usize,
    ) -> PyResult<Option<CsrRankPack>> {
        if !self.sorted || stop < first || stop > self.columns || stop - first > u32::MAX as usize {
            return Ok(None);
        }
        let spans: Vec<_> = populations
            .row_order()
            .iter()
            .map(|&row| self.spans[row])
            .collect();
        let Some(snapshot) = owned::OwnedCsr::snapshot(py, data, indices, &spans, self.columns)?
        else {
            return Ok(None);
        };
        let columns = self.columns;
        match snapshot {
            owned::OwnedCsr::F32I32(rows) => pack(py, rows, first, stop, columns),
            owned::OwnedCsr::F32I64(rows) => pack(py, rows, first, stop, columns),
            owned::OwnedCsr::F64I32(rows) => pack(py, rows, first, stop, columns),
            owned::OwnedCsr::F64I64(rows) => pack(py, rows, first, stop, columns),
        }
    }
}

fn pack<T: Scalar + Into<f64>, I: Index>(
    py: Python<'_>,
    source: owned::Rows<T, I>,
    first: usize,
    stop: usize,
    full_cols: usize,
) -> PyResult<Option<CsrRankPack>> {
    // Small snapshots fit cache and their two linear scans are faster on the
    // detached caller than a pool handoff plus tiny scatter tasks. Larger
    // payloads retain the shared pool and its captured per-caller worker cap.
    let items = pack_work_items::<T, I>(source.values.len());
    pack_workload(py, source, first, stop, full_cols, items)
}

fn pack_work_items<T: Scalar, I: Index>(count: usize) -> usize {
    if count <= (1024 * 1024) / (std::mem::size_of::<T>() + std::mem::size_of::<I>()) {
        0
    } else {
        count
    }
}

fn pack_workload<T: Scalar + Into<f64>, I: Index>(
    py: Python<'_>,
    source: owned::Rows<T, I>,
    first: usize,
    stop: usize,
    full_cols: usize,
    items: usize,
) -> PyResult<Option<CsrRankPack>> {
    host_parallel::run(py, items, |plan| {
        let rows = source.offsets.len() - 1;
        let mut offsets = allocate::<u64>(rows + 1)?;
        let mut nonnegative_finite = true;
        for row in 0..rows {
            let mut previous = None;
            let mut count = 0;
            for position in source.offsets[row]..source.offsets[row + 1] {
                let column = source.columns[position];
                let column = usize::try_from(column.into())
                    .ok()
                    .filter(|&c| c < full_cols)
                    .ok_or_else(|| {
                        PyValueError::new_err("CSR column index changed during staging")
                    })?;
                if previous.is_some_and(|previous| previous >= column) {
                    return Ok(None);
                }
                previous = Some(column);
                if first <= column && column < stop {
                    count += 1;
                    let value = source.values[position].into();
                    nonnegative_finite &= value.is_finite() && value >= 0.0;
                }
            }
            offsets[row + 1] = offsets[row] + count as u64;
        }
        let count = offsets[rows] as usize;
        let mut values = allocate::<T>(count)?;
        let cols = stop - first;
        let columns = if cols <= u16::MAX as usize + 1 {
            let mut output = allocate::<u16>(count)?;
            scatter(
                &source,
                &offsets,
                &mut values,
                &mut output,
                first,
                stop,
                plan,
            );
            Columns::Narrow(output)
        } else {
            let mut output = allocate::<u32>(count)?;
            scatter(
                &source,
                &offsets,
                &mut values,
                &mut output,
                first,
                stop,
                plan,
            );
            Columns::Wide(output)
        };
        Ok(Some(CsrRankPack {
            data: T::owned(values),
            columns,
            offsets,
            rows,
            cols,
            nonnegative_finite,
        }))
    })?
}

fn scatter<T: Scalar, I: Index, C: Copy + TryFrom<usize> + Send>(
    source: &owned::Rows<T, I>,
    offsets: &[u64],
    values: &mut [T],
    columns: &mut [C],
    first: usize,
    stop: usize,
    plan: host_parallel::Parallelism,
) {
    let rows = offsets.len() - 1;
    let grain = rows
        .div_ceil(plan.workers().min(values.len().div_ceil(8192).max(1)))
        .max(1);
    let mut pieces = Vec::new();
    let mut remaining_values = values;
    let mut remaining_columns = columns;
    for row in (0..rows).step_by(grain) {
        let last = (row + grain).min(rows);
        let count = (offsets[last] - offsets[row]) as usize;
        let (values, rest_values) = remaining_values.split_at_mut(count);
        let (columns, rest_columns) = remaining_columns.split_at_mut(count);
        pieces.push((row, last, values, columns));
        remaining_values = rest_values;
        remaining_columns = rest_columns;
    }
    plan.for_each_chunk(&mut pieces, |_, pieces| {
        for (first_row, last_row, values, columns) in pieces {
            let mut output = 0;
            for row in *first_row..*last_row {
                for position in source.offsets[row]..source.offsets[row + 1] {
                    let column = source.columns[position].into() as usize;
                    if first <= column && column < stop {
                        values[output] = source.values[position];
                        columns[output] = C::try_from(column - first)
                            .ok()
                            .expect("column width checked");
                        output += 1;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyBytes, PyDict};

    #[test]
    fn pack_threshold_preserves_rebased_values_for_types_and_worker_limits() {
        fn check<T: Scalar + Into<f64> + From<f32>, I: Index>(py: Python<'_>) -> PyResult<()> {
            let threshold = (1024 * 1024) / (std::mem::size_of::<T>() + std::mem::size_of::<I>());
            for count in [threshold - 1, threshold, threshold + 1] {
                assert_eq!(pack_work_items::<T, I>(count) == 0, count <= threshold);
                let make = || owned::Rows {
                    values: (0..count).map(|i| T::from((i % 13) as f32)).collect(),
                    columns: (0..count)
                        .map(|i| I::try_from(i % 64).ok().unwrap())
                        .collect(),
                    offsets: (0..count)
                        .step_by(64)
                        .chain(std::iter::once(count))
                        .collect(),
                };
                let expected_values: Vec<T> = (0..count)
                    .filter(|i| (3..61).contains(&(i % 64)))
                    .map(|i| T::from((i % 13) as f32))
                    .collect();
                let expected_columns: Vec<u16> = (0..count)
                    .filter(|i| (3..61).contains(&(i % 64)))
                    .map(|i| (i % 64 - 3) as u16)
                    .collect();
                let mut expected_pointers = vec![0u64];
                for first in (0..count).step_by(64) {
                    let stored = (count - first).min(64).min(61).saturating_sub(3);
                    expected_pointers.push(expected_pointers.last().unwrap() + stored as u64);
                }
                for workers in [1, 3, 0] {
                    let previous = host_parallel::set_worker_limit(workers);
                    let result = pack(py, make(), 3, 61, 64);
                    assert_eq!(host_parallel::set_worker_limit(previous), workers);
                    let result = result?.expect("unique sorted columns admit dense staging");
                    assert_eq!(result.data().0, bytes(&expected_values));
                    assert_eq!(result.columns(), primitive_bytes(&expected_columns));
                    assert_eq!(result.pointers(), primitive_bytes(&expected_pointers));
                    assert!(result.nonnegative_finite);
                }
            }
            Ok(())
        }
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            check::<f32, i32>(py)?;
            check::<f32, i64>(py)?;
            check::<f64, i32>(py)?;
            check::<f64, i64>(py)
        })
        .unwrap();
    }

    #[test]
    #[ignore = "manual CPU stage profile; run with otherwise idle CPU/GPU"]
    fn profile_dense_csr_pack_stages() {
        use std::time::Instant;
        fn median(values: &mut [f64]) -> f64 {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        }
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            for (rows, stored) in [(10_000usize, 10usize), (10_000, 100), (30_000, 40)] {
                let count = rows * stored;
                let columns: Vec<i32> = (0..count)
                    .map(|i| ((i % stored) * 1000 / stored) as i32)
                    .collect();
                let data = np.call_method1("ones", (count, "float32"))?;
                let indices = np.call_method1("array", (columns.clone(), "int32"))?;
                let refs = np.call_method1("arange", (rows / 4,))?;
                let members = np.call_method1("arange", (rows / 4, rows))?;
                let group_offsets = [0, rows / 4, rows / 2, rows - rows / 4];
                let spans: Vec<_> = (0..rows)
                    .map(|row| (row * stored, (row + 1) * stored))
                    .collect();
                let mut memberships = Vec::new();
                let mut snapshots = Vec::new();
                for iteration in 0..35 {
                    let start = Instant::now();
                    let populations = Populations::new(py, &refs, &members, &group_offsets, rows)?;
                    if iteration >= 5 {
                        memberships.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                    drop(populations);
                    let start = Instant::now();
                    let snapshot = owned::OwnedCsr::snapshot(py, &data, &indices, &spans, 1000)?;
                    if iteration >= 5 {
                        snapshots.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                    drop(snapshot);
                }
                eprintln!(
                    "rows={rows} stored={stored} memberships_us={:.3} snapshot_us={:.3}",
                    median(&mut memberships),
                    median(&mut snapshots)
                );
                for limit in [-1, 1, 4, 0] {
                    let previous = host_parallel::set_worker_limit(limit.max(0));
                    let mut durations = Vec::new();
                    for iteration in 0..35 {
                        let source = owned::Rows {
                            values: vec![1.0f32; count],
                            columns: columns.clone(),
                            offsets: (0..=rows).map(|row| row * stored).collect(),
                        };
                        let start = Instant::now();
                        let packed = pack_workload(
                            py,
                            source,
                            0,
                            1000,
                            1000,
                            if limit == -1 { 0 } else { count },
                        )?;
                        if iteration >= 5 {
                            durations.push(start.elapsed().as_secs_f64() * 1e6);
                        }
                        assert_eq!(packed.as_ref().map(|p| p.data().2), Some(count));
                        drop(packed);
                    }
                    host_parallel::set_worker_limit(previous);
                    eprintln!(
                        "rows={rows} stored={stored} workers={limit} pack_us={:.3}",
                        median(&mut durations)
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn packed_rows_preserve_spans_precision_and_column_widths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let locals = PyDict::new(py);
            locals.set_item("np", &np)?;
            for dtype in ["float32", "float64"] {
                for index in ["int32", "int64"] {
                    locals.set_item("dtype", dtype)?;
                    locals.set_item("index", index)?;
                    py.run(c"data=np.arange(30,dtype=dtype)+np.asarray(1e-12,dtype=dtype)\nindices=np.tile(np.arange(6,dtype=index),5)\nstarts=np.arange(5,dtype=np.int64)*6+1\nstops=starts+4\nrefs=np.asarray([3,1],np.int32)\nmembers=np.asarray([4,0],np.int64)\nexpected=data.reshape(5,6)[[3,1,4,0],2:5].copy()", None, Some(&locals))?;
                    let get = |name: &str| locals.get_item(name)?.ok_or_else(|| PyValueError::new_err("missing test array"));
                    let indices = get("indices")?;
                    let spans = CsrSpans::new(py, &indices, &get("starts")?, &get("stops")?, 6)?;
                    let groups = Populations::new(py, &get("refs")?, &get("members")?, &[0,1,2], 5)?;
                    let pack = spans.dense_rank_pack(py, &get("data")?, &indices, &groups, 2, 5)?.unwrap();
                    assert_eq!((pack.rows, pack.cols), (4,3));
                    assert_eq!(pack.offsets, [0,3,6,9,12]);
                    assert!(pack.narrow());
                    let expected = get("expected")?.call_method0("tobytes")?;
                    assert_eq!(pack.data().0, expected.cast::<PyBytes>()?.as_bytes());
                    match pack.columns { Columns::Narrow(columns) => assert_eq!(columns, [0,1,2].repeat(4)), _ => unreachable!() }
                }
            }
            for cols in [65_536usize, 65_537] {
                let data = np.call_method1("asarray", (vec![2.0], "float64"))?;
                let indices = np.call_method1("asarray", (vec![cols-1], "int64"))?;
                let starts = np.call_method1("asarray", (vec![0], "int32"))?;
                let stops = np.call_method1("asarray", (vec![1], "int64"))?;
                let refs = np.call_method1("asarray", (vec![0], "int32"))?;
                let members = np.call_method1("asarray", (Vec::<i32>::new(), "int32"))?;
                let groups = Populations::new(py, &refs, &members, &[0,0], 1)?;
                let spans = CsrSpans::new(py, &indices, &starts, &stops, cols)?;
                let pack = spans.dense_rank_pack(py, &data, &indices, &groups, 0, cols)?.unwrap();
                assert_eq!(pack.narrow(), cols == 65_536);
            }
            Ok(())
        }).unwrap();
    }
}
