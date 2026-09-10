//! Bounded CSR snapshots and parallel, stable conversion to CSC upload windows.
mod dense_rank;
mod owned;
mod populations;
use crate::{array::Dtype, host_parallel};
pub use dense_rank::CsrRankPack;
pub use populations::{PopulationPlan, Populations, mapped_rows};
use pyo3::{
    buffer::{Element, PyBuffer, ReadOnlyCell},
    exceptions::{PyMemoryError, PyTypeError, PyValueError},
    prelude::*,
};

enum Values {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

// Only these primitive, padding-free types can be exposed as upload bytes.
trait Scalar: Copy + Default + Send + Sync + Element {
    fn owned(values: Vec<Self>) -> Values;
}
macro_rules! scalar {
    ($ty:ty, $variant:ident) => {
        impl Scalar for $ty {
            fn owned(values: Vec<Self>) -> Values {
                Values::$variant(values)
            }
        }
    };
}
scalar!(f32, F32);
scalar!(f64, F64);
scalar!(i32, I32);
scalar!(i64, I64);

trait Index: Scalar + Into<i64> + TryFrom<usize> {}
impl Index for i32 {}
impl Index for i64 {}

fn bytes<T: Scalar>(values: &[T]) -> &[u8] {
    // SAFETY: Scalar is private and implemented only for initialized primitive
    // floats/integers without padding. The borrow retains the owning vector.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
impl Values {
    fn snapshot(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Self> {
        fn typed<T: Scalar>(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Values> {
            let buffer = PyBuffer::<T>::get(source)?;
            if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "CSC staging requires contiguous vectors",
                ));
            }
            Ok(T::owned(buffer.to_vec(py)?))
        }
        let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
        match dtype.as_str() {
            "float32" => typed::<f32>(py, source),
            "float64" => typed::<f64>(py, source),
            "int32" => typed::<i32>(py, source),
            "int64" => typed::<i64>(py, source),
            _ => Err(PyTypeError::new_err("unsupported CSC staging dtype")),
        }
    }
    fn slice(&self, first: usize, stop: usize) -> PyResult<Self> {
        fn copy<T: Scalar>(values: &[T], first: usize, stop: usize) -> PyResult<Values> {
            let mut output = allocate::<T>(stop - first)?;
            output.copy_from_slice(&values[first..stop]);
            Ok(T::owned(output))
        }
        match self {
            Self::F32(v) => copy(v, first, stop),
            Self::F64(v) => copy(v, first, stop),
            Self::I32(v) => copy(v, first, stop),
            Self::I64(v) => copy(v, first, stop),
        }
    }
    fn view(&self) -> (&[u8], Dtype, usize) {
        match self {
            Self::F32(v) => (bytes(v), Dtype::F32, v.len()),
            Self::F64(v) => (bytes(v), Dtype::F64, v.len()),
            Self::I32(v) => (bytes(v), Dtype::I32, v.len()),
            Self::I64(v) => (bytes(v), Dtype::I64, v.len()),
        }
    }
}

/// Owns every byte needed to upload one bounded column window.
pub struct CscWindow {
    data: Values,
    indices: Values,
    indptr: Values,
    offsets: Vec<u64>,
}
impl CscWindow {
    pub fn from_arrays(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        indptr: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let data = Values::snapshot(py, data)?;
        let indices = Values::snapshot(py, indices)?;
        let indptr = Values::snapshot(py, indptr)?;
        if !matches!(data, Values::F32(_) | Values::F64(_))
            || !matches!(indices, Values::I32(_) | Values::I64(_))
            || data.view().2 != indices.view().2
        {
            return Err(PyTypeError::new_err(
                "invalid CSC value or row-index buffers",
            ));
        }
        fn pointer_values<I: Index>(values: &[I]) -> PyResult<Vec<u64>> {
            let mut result = allocate::<u64>(values.len())?;
            for (output, &value) in result.iter_mut().zip(values) {
                *output = u64::try_from(value.into())
                    .map_err(|_| PyValueError::new_err("negative CSC offset"))?;
            }
            Ok(result)
        }
        let offsets = match &indptr {
            Values::I32(v) => pointer_values(v)?,
            Values::I64(v) => pointer_values(v)?,
            _ => return Err(PyTypeError::new_err("CSC offsets require integer storage")),
        };
        if offsets.first() != Some(&0)
            || offsets.last() != Some(&(data.view().2 as u64))
            || offsets.windows(2).any(|v| v[1] < v[0])
        {
            return Err(PyValueError::new_err("invalid CSC staging offsets"));
        }
        Ok(Self {
            data,
            indices,
            indptr,
            offsets,
        })
    }
    fn window(&self, py: Python<'_>, first: usize, stop: usize) -> PyResult<Self> {
        // Bulk copies over owned native buffers need no worker scheduling.
        py.detach(|| {
            let begin = self.offsets[first] as usize;
            let end = self.offsets[stop] as usize;
            let mut offsets = allocate::<u64>(stop - first + 1)?;
            for (target, &value) in offsets.iter_mut().zip(&self.offsets[first..=stop]) {
                *target = value - begin as u64;
            }
            let mut indptr = self.indptr.slice(first, stop + 1)?;
            match &mut indptr {
                Values::I32(v) => v.iter_mut().for_each(|v| *v -= begin as i32),
                Values::I64(v) => v.iter_mut().for_each(|v| *v -= begin as i64),
                _ => unreachable!("CSC offsets are integer vectors"),
            }
            Ok(Self {
                data: self.data.slice(begin, end)?,
                indices: self.indices.slice(begin, end)?,
                indptr,
                offsets,
            })
        })
    }
    pub fn data(&self) -> (&[u8], Dtype, usize) {
        self.data.view()
    }
    pub fn indices(&self) -> (&[u8], Dtype, usize) {
        self.indices.view()
    }
    pub fn indptr(&self) -> (&[u8], Dtype, usize) {
        self.indptr.view()
    }
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
}

fn allocate<T: Default + Clone>(len: usize) -> PyResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("unable to allocate CSR staging workspace"))?;
    values.resize(len, T::default());
    Ok(values)
}

fn lower<I: Index>(
    values: &[ReadOnlyCell<I>],
    mut first: usize,
    mut stop: usize,
    target: usize,
) -> usize {
    while first < stop {
        let middle = first + (stop - first) / 2;
        if values[middle].get().into() < target as i64 {
            first = middle + 1;
        } else {
            stop = middle;
        }
    }
    first
}

struct Snapshot<T, I> {
    values: Vec<T>,
    columns: Vec<usize>,
    rows: Vec<I>,
}

fn convert<T: Scalar, I: Index>(
    py: Python<'_>,
    snapshot: Snapshot<T, I>,
    columns: usize,
) -> PyResult<CscWindow> {
    host_parallel::run(py, snapshot.values.len(), |plan| {
        convert_native(snapshot, columns, plan)
    })?
}

fn convert_native<T: Scalar, I: Index>(
    snapshot: Snapshot<T, I>,
    columns: usize,
    plan: host_parallel::Parallelism,
) -> PyResult<CscWindow> {
    let count = snapshot.values.len();
    // Give each partition enough stored entries to amortize histogram and
    // scatter scheduling; ordinary narrow sparse windows need few tasks.
    let parts = plan.workers().min(count.div_ceil(8192).max(1));
    let chunk = count.div_ceil(parts).max(1);
    let mut histograms = (0..parts)
        .map(|_| allocate::<usize>(columns))
        .collect::<PyResult<Vec<_>>>()?;
    plan.for_each_chunk(&mut histograms, |offset, histograms| {
        for (local, histogram) in histograms.iter_mut().enumerate() {
            let start = (offset + local) * chunk;
            let stop = (start + chunk).min(count);
            for &column in &snapshot.columns[start.min(count)..stop] {
                histogram[column] += 1;
            }
        }
    });
    let mut offsets = allocate::<u64>(columns + 1)?;
    for column in 0..columns {
        offsets[column + 1] =
            offsets[column] + histograms.iter().map(|h| h[column] as u64).sum::<u64>();
    }
    let indptr = offsets
        .iter()
        .map(|&v| {
            I::try_from(v as usize)
                .map_err(|_| PyValueError::new_err("CSC offsets exceed index dtype"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut output = allocate::<T>(count)?;
    let mut indices = allocate::<I>(count)?;
    struct Target<'a, T, I> {
        start: usize,
        data: Vec<&'a mut [T]>,
        rows: Vec<&'a mut [I]>,
    }
    let mut targets = (0..parts)
        .map(|part| Target {
            start: part * chunk,
            data: Vec::with_capacity(columns),
            rows: Vec::with_capacity(columns),
        })
        .collect::<Vec<_>>();
    let mut value_tail = output.as_mut_slice();
    let mut index_tail = indices.as_mut_slice();
    // Each row partition owns a separate slice of every CSC column. Rust's
    // split borrows establish disjoint writes without atomic scatter or raw
    // Send pointers, and partition order preserves stable row ordering.
    // Traverse partition-major counts in column order to split CSC output.
    #[allow(clippy::needless_range_loop)]
    for column in 0..columns {
        for (part, target) in targets.iter_mut().enumerate() {
            let width = histograms[part][column];
            let (values, tail) = value_tail.split_at_mut(width);
            value_tail = tail;
            target.data.push(values);
            let (rows, tail) = index_tail.split_at_mut(width);
            index_tail = tail;
            target.rows.push(rows);
        }
    }
    plan.for_each_chunk(&mut targets, |_, targets| {
        for target in targets {
            for position in target.start.min(count)..(target.start + chunk).min(count) {
                let column = snapshot.columns[position];
                let values = std::mem::take(&mut target.data[column]);
                let (first, rest) = values.split_first_mut().expect("counted CSC values");
                *first = snapshot.values[position];
                target.data[column] = rest;
                let rows = std::mem::take(&mut target.rows[column]);
                let (first, rest) = rows.split_first_mut().expect("counted CSC row indices");
                *first = snapshot.rows[position];
                target.rows[column] = rest;
            }
        }
    });
    Ok(CscWindow {
        data: T::owned(output),
        indices: I::owned(indices),
        indptr: I::owned(indptr),
        offsets,
    })
}

fn typed_spans<T: Scalar, I: Index>(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    source_spans: &[(usize, usize)],
    first_column: usize,
    stop_column: usize,
    sorted: bool,
) -> PyResult<CscWindow> {
    let data = PyBuffer::<T>::get(data)?;
    let indices = PyBuffer::<I>::get(indices)?;
    if data.dimensions() != 1
        || indices.dimensions() != 1
        || data.item_count() != indices.item_count()
    {
        return Err(PyValueError::new_err(
            "invalid CSR staging array dimensions",
        ));
    }
    let values = data
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("CSR data must be contiguous"))?;
    let index_values = indices
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("CSR indices must be contiguous"))?;
    let rows = source_spans.len();
    if rows > 0 && I::try_from(rows - 1).is_err() {
        return Err(PyValueError::new_err("CSC row indices exceed index dtype"));
    }
    let mut spans = allocate::<(usize, usize)>(rows)?;
    let mut count = 0usize;
    for (row, &(start, stop)) in source_spans.iter().enumerate() {
        if stop < start || stop > values.len() {
            return Err(PyValueError::new_err("invalid CSR row span"));
        }
        let (first, last, selected) = if sorted {
            let first = lower(index_values, start, stop, first_column);
            let last = lower(index_values, first, stop, stop_column);
            (first, last, last - first)
        } else {
            let selected = index_values[start..stop]
                .iter()
                .filter(|v| {
                    let column = v.get().into();
                    column >= first_column as i64 && column < stop_column as i64
                })
                .count();
            (start, stop, selected)
        };
        spans[row] = (first, last);
        count = count
            .checked_add(selected)
            .ok_or_else(|| PyValueError::new_err("CSR window size overflow"))?;
    }
    let mut snapshot = Snapshot {
        values: allocate::<T>(count)?,
        columns: allocate::<usize>(count)?,
        rows: allocate::<I>(count)?,
    };
    let mut position = 0;
    for (row, (first, stop)) in spans.into_iter().enumerate() {
        let row_id = I::try_from(row).ok().expect("row dtype validated above");
        for source in first..stop {
            let column = index_values[source].get().into();
            if column < first_column as i64 || column >= stop_column as i64 {
                continue;
            }
            if position >= count {
                return Err(PyValueError::new_err("CSR indices changed during staging"));
            }
            snapshot.values[position] = values[source].get();
            snapshot.columns[position] = column as usize - first_column;
            snapshot.rows[position] = row_id;
            position += 1;
        }
    }
    if position != count {
        return Err(PyValueError::new_err("CSR indices changed during staging"));
    }
    // No Python-owned data is read by detached conversion workers.
    drop((data, indices));
    convert(py, snapshot, stop_column - first_column)
}

fn offsets(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    fn typed<I: Index>(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let buffer = PyBuffer::<I>::get(source)?;
        if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "CSR row spans require contiguous vectors",
            ));
        }
        buffer
            .to_vec(py)?
            .into_iter()
            .map(|v| {
                usize::try_from(v.into())
                    .map_err(|_| PyValueError::new_err("negative CSR row span"))
            })
            .collect()
    }
    match source
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?
        .as_str()
    {
        "int32" => typed::<i32>(py, source),
        "int64" => typed::<i64>(py, source),
        _ => Err(PyTypeError::new_err("CSR row spans require int32 or int64")),
    }
}

/// A reusable snapshot of each caller-supplied row interval. Offsets may have a
/// different integer width from column indices, and gaps remain excluded.
/// The Python data and index owners remain with the caller until all batches
/// finish. Inputs fitting a 64 MiB conversion payload budget are converted
/// once into owned CSC storage; larger inputs copy each bounded window separately.
pub struct CsrSpans {
    spans: Vec<(usize, usize)>,
    columns: usize,
    sorted: bool,
    owned: std::cell::OnceCell<Option<CscWindow>>,
}
impl CsrSpans {
    pub fn new(
        py: Python<'_>,
        indices: &Bound<'_, PyAny>,
        starts: &Bound<'_, PyAny>,
        stops: &Bound<'_, PyAny>,
        columns: usize,
    ) -> PyResult<Self> {
        if columns > i64::MAX as usize {
            return Err(PyValueError::new_err("CSR column count overflow"));
        }
        let starts = offsets(py, starts)?;
        let stops = offsets(py, stops)?;
        if starts.len() != stops.len() {
            return Err(PyValueError::new_err("CSR row span lengths must match"));
        }
        let spans: Vec<_> = starts.into_iter().zip(stops).collect();
        fn validate<I: Index>(
            py: Python<'_>,
            source: &Bound<'_, PyAny>,
            spans: &[(usize, usize)],
            columns: usize,
        ) -> PyResult<bool> {
            let buffer = PyBuffer::<I>::get(source)?;
            if buffer.dimensions() != 1 {
                return Err(PyValueError::new_err("CSR indices require a vector"));
            }
            let values = buffer
                .as_slice(py)
                .ok_or_else(|| PyValueError::new_err("CSR indices must be contiguous"))?;
            let mut sorted = true;
            for &(start, stop) in spans {
                if stop < start || stop > values.len() {
                    return Err(PyValueError::new_err("invalid CSR row span"));
                }
                let mut previous = -1i64;
                for value in &values[start..stop] {
                    let value = value.get().into();
                    if value < 0 || value as usize >= columns {
                        return Err(PyValueError::new_err("CSR column index out of bounds"));
                    }
                    sorted &= value >= previous;
                    previous = value;
                }
            }
            Ok(sorted)
        }
        let sorted = match indices
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?
            .as_str()
        {
            "int32" => validate::<i32>(py, indices, &spans, columns)?,
            "int64" => validate::<i64>(py, indices, &spans, columns)?,
            _ => return Err(PyTypeError::new_err("CSR indices require int32 or int64")),
        };
        Ok(Self {
            spans,
            columns,
            sorted,
            owned: std::cell::OnceCell::new(),
        })
    }

    pub fn window(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        first: usize,
        stop: usize,
    ) -> PyResult<CscWindow> {
        if stop < first || stop > self.columns {
            return Err(PyValueError::new_err("invalid CSR staging column window"));
        }
        if self.owned.get().is_none() {
            let snapshot = owned::OwnedCsr::snapshot(py, data, indices, &self.spans, self.columns)?
                .map(|snapshot| snapshot.into_csc(py, self.columns))
                .transpose()?;
            // This plan belongs to one staging call. No Python-owned buffer
            // or plan cache is accessed by its detached conversion workers.
            let _ = self.owned.set(snapshot);
        }
        if let Some(Some(snapshot)) = self.owned.get() {
            return snapshot.window(py, first, stop);
        }
        let value_dtype: String = data.getattr("dtype")?.getattr("name")?.extract()?;
        let index_dtype: String = indices.getattr("dtype")?.getattr("name")?.extract()?;
        match (value_dtype.as_str(), index_dtype.as_str()) {
            ("float32", "int32") => {
                typed_spans::<f32, i32>(py, data, indices, &self.spans, first, stop, self.sorted)
            }
            ("float32", "int64") => {
                typed_spans::<f32, i64>(py, data, indices, &self.spans, first, stop, self.sorted)
            }
            ("float64", "int32") => {
                typed_spans::<f64, i32>(py, data, indices, &self.spans, first, stop, self.sorted)
            }
            ("float64", "int64") => {
                typed_spans::<f64, i64>(py, data, indices, &self.spans, first, stop, self.sorted)
            }
            _ => Err(PyTypeError::new_err(
                "CSR staging requires float32/float64 and int32/int64",
            )),
        }
    }
}

/// Snapshot only selected entries of a validated, sorted SciPy CSR source.
#[cfg(test)]
pub fn csr_window(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    first: usize,
    stop: usize,
) -> PyResult<CscWindow> {
    if source.getattr("format")?.extract::<String>()? != "csr" {
        return Err(PyTypeError::new_err("CSR staging requires CSR storage"));
    }
    let (rows, columns): (usize, usize) = source.getattr("shape")?.extract()?;
    if columns > i64::MAX as usize || rows == usize::MAX {
        return Err(PyValueError::new_err("CSR dimensions overflow"));
    }
    let pointers = offsets(py, &source.getattr("indptr")?)?;
    if pointers.len() != rows + 1 {
        return Err(PyValueError::new_err("invalid CSR indptr length"));
    }
    let spans = pointers.windows(2).map(|p| (p[0], p[1])).collect();
    CsrSpans {
        spans,
        columns,
        sorted: true,
        owned: std::cell::OnceCell::new(),
    }
    .window(
        py,
        &source.getattr("data")?,
        &source.getattr("indices")?,
        first,
        stop,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PySlice};

    /// Reproduce bounded host staging costs independently of CUDA or Python
    /// sparse conversion. Run explicitly with --ignored --nocapture.
    #[test]
    #[ignore]
    fn profile_csr_span_windows() {
        Python::initialize();
        Python::attach(|py| {
            for rows in [10_000usize, 50_000] {
                let globals = PyDict::new(py);
                globals.set_item("rows", rows).unwrap();
                py.run(
                    pyo3::ffi::c_str!(
                        "import numpy as np\nindices = (np.arange(10)[None, :] * 100 + np.arange(rows)[:, None] % 100).astype(np.int32).ravel()\nstarts = np.arange(rows, dtype=np.int32) * 10\nstops = starts + 10\ndata = (np.arange(rows * 10) % 7).astype(np.float32)"
                    ),
                    Some(&globals),
                    None,
                ).unwrap();
                let indices = globals.get_item("indices").unwrap().unwrap();
                let data = globals.get_item("data").unwrap().unwrap();
                let starts = globals.get_item("starts").unwrap().unwrap();
                let stops = globals.get_item("stops").unwrap().unwrap();
                let plans = [
                    CsrSpans::new(py, &indices, &starts, &stops, 1000).unwrap(),
                    CsrSpans::new(py, &indices, &starts, &stops, 1000).unwrap(),
                ];
                assert!(plans[0].owned.set(None).is_ok());
                for workers in [1, 4, 0] {
                    let previous = host_parallel::set_worker_limit(workers);
                    let mut samples = [Vec::new(), Vec::new()];
                    for repetition in 0..12 {
                        for index in [repetition % 2, 1 - repetition % 2] {
                            let start = std::time::Instant::now();
                            let mut count = 0;
                            for first in (0..1000).step_by(64) {
                                let window = plans[index]
                                    .window(py, &data, &indices, first, (first + 64).min(1000))
                                    .unwrap();
                                count += std::hint::black_box(window.data().2);
                            }
                            assert_eq!(count, rows * 10);
                            samples[index].push(start.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    host_parallel::set_worker_limit(previous);
                    for (index, samples) in samples.iter_mut().enumerate() {
                        samples.sort_by(f64::total_cmp);
                        println!(
                            "csr_span_windows rows={rows} cols=1000 workers={workers} cached={} milliseconds={}",
                            index == 1,
                            samples[6]
                        );
                    }
                }
            }
        });
    }

    fn compare(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        first: usize,
        stop: usize,
        index_dtype: &str,
    ) -> PyResult<()> {
        let got = csr_window(py, source, first, stop)?;
        let expected = source
            .get_item((
                PySlice::new(py, 0, isize::MAX, 1),
                PySlice::new(py, first as isize, stop as isize, 1),
            ))?
            .call_method0("tocsc")?;
        for (name, (bytes, _, len)) in [
            ("data", got.data()),
            ("indices", got.indices()),
            ("indptr", got.indptr()),
        ] {
            let values = expected.getattr(name)?;
            let values = if name == "data" {
                values
            } else {
                values.call_method1("astype", (index_dtype,))?
            };
            let reference: Vec<u8> = values.call_method0("tobytes")?.extract()?;
            assert_eq!(bytes, reference, "{name}, columns {first}..{stop}");
            assert_eq!(len, values.len()?);
        }
        let reference: Vec<u64> = expected
            .getattr("indptr")?
            .call_method0("tolist")?
            .extract()?;
        assert_eq!(got.offsets(), reference);
        Ok(())
    }

    #[test]
    fn bounded_csr_windows_match_scipy_for_types_empty_and_parallel_inputs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let sparse = py.import("scipy.sparse")?;
            for rows in [0, 1, 31, 32, 33, 57, 1025, 4097] {
                for dtype in ["float32", "float64"] {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dtype", dtype)?;
                    let raw = np
                        .call_method("arange", (rows * 37,), Some(&kwargs))?
                        .call_method1("reshape", ((rows, 37),))?;
                    let mask = raw
                        .call_method1("__mod__", (7,))?
                        .call_method1("__lt__", (5,))?;
                    let values = raw.call_method1("__sub__", (123.25,))?;
                    let values = np.call_method1("where", (&mask, 0, values))?;
                    let source = sparse.call_method1("csr_matrix", (values,))?;
                    for index_dtype in ["int32", "int64"] {
                        for name in ["indices", "indptr"] {
                            source.setattr(
                                name,
                                source
                                    .getattr(name)?
                                    .call_method1("astype", (index_dtype,))?,
                            )?;
                        }
                        for (first, stop) in [(0, 0), (0, 17), (3, 31), (37, 37)] {
                            compare(py, &source, first, stop, index_dtype)?;
                        }
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn duplicates_and_explicit_zeros_are_stable_and_bad_spans_rejected() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let sparse = py.import("scipy.sparse")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("shape", (3, 5))?;
            let source = sparse.call_method(
                "csr_matrix",
                ((
                    vec![0.0f64, 7.0, -3.0, 9.0, 8.0],
                    vec![1i32, 1, 4, 0, 3],
                    vec![0i32, 3, 3, 5],
                ),),
                Some(&kwargs),
            )?;
            compare(py, &source, 0, 5, "int32")?;
            compare(py, &source, 1, 4, "int32")?;
            assert!(csr_window(py, &source, 4, 3).is_err());
            assert!(csr_window(py, &source, 0, 6).is_err());
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "int32")?;
            source.setattr(
                "indptr",
                np.call_method("array", (vec![0, 3, 2, 5],), Some(&kwargs))?,
            )?;
            assert!(csr_window(py, &source, 0, 5).is_err());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn wide_sparse_inputs_skip_the_full_conversion_cache() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let options = PyDict::new(py);
            options.set_item("dtype", "int32")?;
            let indices = np.call_method("array", (vec![1, 3],), Some(&options))?;
            let starts = np.call_method("array", (vec![0],), Some(&options))?;
            let stops = np.call_method("array", (vec![2],), Some(&options))?;
            options.set_item("dtype", "float32")?;
            let data = np.call_method("array", (vec![2.0, 7.0],), Some(&options))?;
            // A tiny selected window from a very wide source must not allocate
            // the full source's column histograms or pointer arrays.
            let plan = CsrSpans::new(py, &indices, &starts, &stops, 2_000_000)?;
            let window = plan.window(py, &data, &indices, 0, 4)?;
            assert!(matches!(plan.owned.get(), Some(None)));
            assert_eq!(window.offsets(), [0, 0, 1, 1, 2]);
            assert_eq!(window.data().0, bytes(&[2.0f32, 7.0]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn independent_row_spans_preserve_gaps_unsorted_rows_and_mixed_widths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let np = py.import("numpy")?;
            let sp = py.import("scipy.sparse")?;
            let columns = [4, 3, 1, 1, 0, 4, 2, 0, 3, 4, 1, 2, 0, 4, 1, 0, 3];
            for dtype in ["float32", "float64"] {
                let options = PyDict::new(py);
                options.set_item("dtype", dtype)?;
                let data = np.call_method("arange", (columns.len(),), Some(&options))?;
                for index_dtype in ["int32", "int64"] {
                    options.set_item("dtype", index_dtype)?;
                    let indices = np.call_method("array", (columns.to_vec(),), Some(&options))?;
                    for span_dtype in ["int32", "int64"] {
                        options.set_item("dtype", span_dtype)?;
                        let starts = np.call_method("array", (vec![1, 6, 13],), Some(&options))?;
                        let stops = np.call_method("array", (vec![5, 9, 16],), Some(&options))?;
                        let plan = CsrSpans::new(py, &indices, &starts, &stops, 5)?;
                        let fallback = CsrSpans::new(py, &indices, &starts, &stops, 5)?;
                        assert!(fallback.owned.set(None).is_ok());
                        assert!(!plan.sorted);
                        starts.call_method1("fill", (0,))?;
                        stops.call_method1("fill", (0,))?;
                        let positions: Vec<usize> = (1..5).chain(6..9).chain(13..16).collect();
                        let selected_data = data.get_item(positions.clone())?;
                        let selected_indices = indices.get_item(positions)?;
                        let shape = PyDict::new(py);
                        shape.set_item("shape", (3, 5))?;
                        let source = sp.call_method(
                            "csr_matrix",
                            ((selected_data, selected_indices, vec![0, 4, 7, 10]),),
                            Some(&shape),
                        )?;
                        for (first, stop) in [(0, 5), (1, 4), (3, 3)] {
                            let got = plan.window(py, &data, &indices, first, stop)?;
                            let bounded = fallback.window(py, &data, &indices, first, stop)?;
                            assert_eq!(got.data().0, bounded.data().0);
                            assert_eq!(got.indices().0, bounded.indices().0);
                            assert_eq!(got.indptr().0, bounded.indptr().0);
                            let expected = source
                                .get_item((
                                    PySlice::new(py, 0, 3, 1),
                                    PySlice::new(py, first as isize, stop as isize, 1),
                                ))?
                                .call_method0("tocsc")?;
                            for (name, view) in [
                                ("data", got.data()),
                                ("indices", got.indices()),
                                ("indptr", got.indptr()),
                            ] {
                                let value = expected.getattr(name)?.call_method1(
                                    "astype",
                                    (if name == "data" { dtype } else { index_dtype },),
                                )?;
                                let reference: Vec<u8> =
                                    value.call_method0("tobytes")?.extract()?;
                                assert_eq!(view.0, reference, "{name}, columns {first}..{stop}");
                            }
                        }
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }
}
