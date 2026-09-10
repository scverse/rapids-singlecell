//! Owned, bounded host snapshots for detached staging transformations.
use crate::{array::Dtype, host_parallel};
use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

/// Snapshot a strided window in its contiguous axis's order. NumPy exports
/// direct buffers, so a rectangular slice can copy whole rows/columns through
/// the buffer C API instead of its generic element-at-a-time iterator.
fn snapshot<T: Element + Copy>(py: Python<'_>, buffer: &PyBuffer<T>) -> PyResult<(Vec<T>, bool)> {
    if buffer.is_c_contiguous() {
        return Ok((buffer.to_vec(py)?, false));
    }
    if buffer.is_fortran_contiguous() {
        return Ok((buffer.to_fortran_vec(py)?, true));
    }
    let item_size = std::mem::size_of::<T>() as isize;
    let axis = if buffer.suboffsets().is_none() && buffer.strides()[1] == item_size {
        1
    } else if buffer.suboffsets().is_none() && buffer.strides()[0] == item_size {
        0
    } else {
        return Ok((buffer.to_vec(py)?, false));
    };
    let shape = buffer.shape();
    let mut output = Vec::<T>::new();
    output.try_reserve_exact(buffer.item_count()).map_err(|_| {
        pyo3::exceptions::PyMemoryError::new_err("unable to allocate host snapshot")
    })?;
    if buffer.item_count() == 0 {
        return Ok((output, axis == 0));
    }
    let mut inner = shape[axis] as isize;
    let mut stride = item_size;
    let bytes = inner
        .checked_mul(item_size)
        .ok_or_else(|| PyValueError::new_err("host snapshot row size overflow"))?;
    for outer in 0..shape[1 - axis] {
        let indices = if axis == 1 { [outer, 0] } else { [0, outer] };
        // SAFETY: the exported NumPy buffer remains owned and attached for
        // every copy. The selected axis has positive unit element stride and
        // the complete row/column lies within its validated shape. This
        // temporary descriptor borrows that contiguous region; it acquires no
        // ownership and is never released. The C buffer API copies into a
        // disjoint native allocation, without creating Rust references to
        // mutable Python storage. All elements are initialized before set_len.
        unsafe {
            let view = pyo3::ffi::Py_buffer {
                buf: buffer.get_ptr(&indices),
                obj: std::ptr::null_mut(),
                len: bytes,
                itemsize: item_size,
                readonly: 1,
                ndim: 1,
                format: std::ptr::null_mut(),
                shape: &mut inner,
                strides: &mut stride,
                suboffsets: std::ptr::null_mut(),
                internal: std::ptr::null_mut(),
            };
            let target = output.as_mut_ptr().add(outer * shape[axis]).cast();
            if pyo3::ffi::PyBuffer_ToContiguous(target, &view, bytes, b'C' as _) != 0 {
                return Err(PyErr::fetch(py));
            }
        }
    }
    // SAFETY: every contiguous region was copied successfully above, covering
    // exactly item_count elements in the reserved, disjoint destination.
    unsafe {
        output.set_len(buffer.item_count());
    }
    Ok((output, axis == 0))
}

pub enum FloatValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub struct FloatBuffer {
    values: FloatValues,
    shape: [usize; 2],
}

/// Copy and validate sparse offsets without materializing Python integers.
pub fn sparse_offsets(py: Python<'_>, source: &Bound<'_, PyAny>, nnz: usize) -> PyResult<Vec<i64>> {
    enum Offsets {
        I32(Vec<i32>),
        I64(Vec<i64>),
    }
    let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
    let (values, count) = match dtype.as_str() {
        "int32" => {
            let buffer = PyBuffer::<i32>::get(source)?;
            (Offsets::I32(buffer.to_vec(py)?), buffer.item_count())
        }
        "int64" => {
            let buffer = PyBuffer::<i64>::get(source)?;
            (Offsets::I64(buffer.to_vec(py)?), buffer.item_count())
        }
        _ => {
            return Err(PyTypeError::new_err(
                "sparse offsets require int32 or int64",
            ));
        }
    };
    host_parallel::run(py, count, |plan| {
        let output = match values {
            Offsets::I32(values) => {
                let mut output = vec![0; values.len()];
                plan.for_each_chunk(&mut output, |offset, part| {
                    for (target, &source) in part.iter_mut().zip(&values[offset..]) {
                        *target = i64::from(source);
                    }
                });
                output
            }
            Offsets::I64(values) => values,
        };
        if output.iter().any(|&p| p < 0 || p as usize > nnz)
            || output.windows(2).any(|p| p[0] > p[1])
        {
            return Err(PyValueError::new_err("invalid compressed sparse indptr"));
        }
        Ok(output)
    })?
}

impl FloatBuffer {
    pub fn dtype(&self) -> Dtype {
        match &self.values {
            FloatValues::F32(_) => Dtype::F32,
            FloatValues::F64(_) => Dtype::F64,
        }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: floats contain no padding, all values are initialized, and
        // the returned slice cannot outlive the immutable owned vector borrow.
        match &self.values {
            FloatValues::F32(values) => unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast(),
                    std::mem::size_of_val(values.as_slice()),
                )
            },
            FloatValues::F64(values) => unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast(),
                    std::mem::size_of_val(values.as_slice()),
                )
            },
        }
    }
}

// Architecture-specific tiles only access owned Rust snapshots. Unaligned
// SIMD loads preserve every bit, including signed zeros and NaN payloads.
trait Transpose: Copy + Default + Send + Sync {
    fn tile(
        source: &[Self],
        input_stride: usize,
        output: &mut [Self],
        output_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        for r in 0..rows {
            for c in 0..cols {
                output[c * output_stride + r] = source[r * input_stride + c];
            }
        }
    }
}

impl Transpose for f32 {
    #[cfg(target_arch = "aarch64")]
    fn tile(
        source: &[Self],
        input_stride: usize,
        output: &mut [Self],
        output_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(rows > 0 && cols > 0);
        assert!(source.len() >= (rows - 1) * input_stride + cols);
        assert!(output.len() >= (cols - 1) * output_stride + rows);
        let full_rows = rows / 4 * 4;
        let full_cols = cols / 4 * 4;
        for r in (0..full_rows).step_by(4) {
            for c in (0..full_cols).step_by(4) {
                // SAFETY: the slice checks cover all four input rows and
                // output columns; each vector remains inside its row/column.
                // Input/output are disjoint owned buffers. AArch64 guarantees
                // Advanced SIMD, and vld1/vst1 permit unaligned addresses.
                unsafe {
                    let src = source.as_ptr().add(r * input_stride + c);
                    let dst = output.as_mut_ptr().add(c * output_stride + r);
                    let a = vld1q_f32(src);
                    let b = vld1q_f32(src.add(input_stride));
                    let c = vld1q_f32(src.add(2 * input_stride));
                    let d = vld1q_f32(src.add(3 * input_stride));
                    let ab0 = vtrn1q_f32(a, b);
                    let ab1 = vtrn2q_f32(a, b);
                    let cd0 = vtrn1q_f32(c, d);
                    let cd1 = vtrn2q_f32(c, d);
                    vst1q_f32(dst, vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)));
                    vst1q_f32(
                        dst.add(output_stride),
                        vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)),
                    );
                    vst1q_f32(
                        dst.add(2 * output_stride),
                        vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)),
                    );
                    vst1q_f32(
                        dst.add(3 * output_stride),
                        vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)),
                    );
                }
            }
            for row in r..r + 4 {
                for col in full_cols..cols {
                    output[col * output_stride + row] = source[row * input_stride + col];
                }
            }
        }
        for row in full_rows..rows {
            for col in 0..cols {
                output[col * output_stride + row] = source[row * input_stride + col];
            }
        }
    }
}

impl Transpose for f64 {
    #[cfg(target_arch = "aarch64")]
    fn tile(
        source: &[Self],
        input_stride: usize,
        output: &mut [Self],
        output_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        use std::arch::aarch64::*;
        assert!(rows > 0 && cols > 0);
        assert!(source.len() >= (rows - 1) * input_stride + cols);
        assert!(output.len() >= (cols - 1) * output_stride + rows);
        let full_rows = rows / 2 * 2;
        let full_cols = cols / 2 * 2;
        for r in (0..full_rows).step_by(2) {
            for c in (0..full_cols).step_by(2) {
                // SAFETY: the checked slices contain both full input rows and
                // output columns, and cannot alias. AArch64 guarantees SIMD.
                unsafe {
                    let src = source.as_ptr().add(r * input_stride + c);
                    let dst = output.as_mut_ptr().add(c * output_stride + r);
                    let a = vld1q_f64(src);
                    let b = vld1q_f64(src.add(input_stride));
                    vst1q_f64(dst, vzip1q_f64(a, b));
                    vst1q_f64(dst.add(output_stride), vzip2q_f64(a, b));
                }
            }
            for row in r..r + 2 {
                for col in full_cols..cols {
                    output[col * output_stride + row] = source[row * input_stride + col];
                }
            }
        }
        for row in full_rows..rows {
            for col in 0..cols {
                output[col * output_stride + row] = source[row * input_stride + col];
            }
        }
    }
}

fn transpose_owned<T: Transpose>(
    source: Vec<T>,
    shape: [usize; 2],
    from_fortran: bool,
    plan: Option<host_parallel::Parallelism>,
) -> PyResult<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len()).map_err(|_| {
        pyo3::exceptions::PyMemoryError::new_err("unable to allocate host transpose workspace")
    })?;
    output.resize(source.len(), T::default());
    if source.is_empty() {
        return Ok(output);
    }
    let (major, minor) = if from_fortran {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };
    // Avoid sending tiny column slices to every worker. Four-column
    // alignment retains SIMD tiles even when the worker limit is high.
    let task_count = source
        .len()
        .div_ceil(1_048_576 / std::mem::size_of::<T>())
        .min(plan.map_or(1, |p| p.workers().min(4)))
        .max(1);
    let majors_per_task = major.div_ceil(task_count).div_ceil(4) * 4;
    let mut partitions: Vec<_> = output
        .chunks_mut(majors_per_task * minor)
        .enumerate()
        .collect();
    let operation = |_: usize, parts: &mut [(usize, &mut [T])]| {
        for (part_index, part) in parts {
            let first_major = *part_index * majors_per_task;
            let local_major = part.len() / minor;
            // Cache-blocked transpose. Both source rows and output columns
            // stay in cache for each 32-by-32 tile instead of performing a
            // full strided pass for every output column.
            for minor_start in (0..minor).step_by(32) {
                for major_start in (0..local_major).step_by(32) {
                    T::tile(
                        &source[minor_start * major + first_major + major_start..],
                        major,
                        &mut part[major_start * minor + minor_start..],
                        minor,
                        (minor - minor_start).min(32),
                        (local_major - major_start).min(32),
                    );
                }
            }
        }
    };
    if let Some(plan) = plan {
        plan.for_each_chunk(&mut partitions, operation);
    } else {
        operation(0, &mut partitions);
    }
    Ok(output)
}
fn reorder<T: Transpose>(
    py: Python<'_>,
    source: Vec<T>,
    shape: [usize; 2],
    from_fortran: bool,
) -> PyResult<Vec<T>> {
    // Keep bounded windows on the caller CPU: dispatching the owned snapshot
    // to the pool costs more than its SIMD transpose and evicts useful cache.
    // The interpreter is detached in both paths; larger windows retain bounded
    // shared-pool parallelism and its caller-specific worker ceiling.
    if std::mem::size_of_val(source.as_slice()) <= 8 * 1024 * 1024 {
        py.detach(|| transpose_owned(source, shape, from_fortran, None))
    } else {
        host_parallel::run(py, source.len(), |plan| {
            transpose_owned(source, shape, from_fortran, Some(plan))
        })?
    }
}

/// Snapshot a bounded two-dimensional NumPy window and pack it in C/F order.
///
/// The Python buffer is read while attached; layout conversion then reads only
/// a native snapshot while detached. Callers should retain this snapshot (or
/// its private pinned copy) until the corresponding async transfer completes.
pub fn pack_dense(py: Python<'_>, source: &Bound<'_, PyAny>, order: &str) -> PyResult<FloatBuffer> {
    let fortran = match order {
        "F" => true,
        "C" => false,
        _ => return Err(PyValueError::new_err("host staging order must be C or F")),
    };
    let np = py.import("numpy")?;
    if !source.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("host staging requires a NumPy array"));
    }
    let shape: Vec<usize> = source.getattr("shape")?.extract()?;
    let shape: [usize; 2] = shape
        .try_into()
        .map_err(|_| PyValueError::new_err("host staging requires two dimensions"))?;
    let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
    let values = match dtype.as_str() {
        "float32" => {
            let buffer = PyBuffer::<f32>::get(source)?;
            if buffer.shape() != shape {
                return Err(PyValueError::new_err(
                    "host staging shape does not match its buffer",
                ));
            }
            let (snapshot, native_fortran) = snapshot(py, &buffer)?;
            FloatValues::F32(if native_fortran == fortran {
                snapshot
            } else {
                reorder(py, snapshot, shape, native_fortran)?
            })
        }
        "float64" => {
            let buffer = PyBuffer::<f64>::get(source)?;
            if buffer.shape() != shape {
                return Err(PyValueError::new_err(
                    "host staging shape does not match its buffer",
                ));
            }
            let (snapshot, native_fortran) = snapshot(py, &buffer)?;
            FloatValues::F64(if native_fortran == fortran {
                snapshot
            } else {
                reorder(py, snapshot, shape, native_fortran)?
            })
        }
        _ => {
            return Err(PyTypeError::new_err(
                "host staging dtype must be float32 or float64",
            ));
        }
    };
    Ok(FloatBuffer { values, shape })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn sparse_offset_snapshots_validate_without_python_integer_lists() {
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            for dtype in ["int32", "int64"] {
                let kw = PyDict::new(py);
                kw.set_item("dtype", dtype).unwrap();
                let source = np.call_method("arange", (8193,), Some(&kw)).unwrap();
                assert_eq!(
                    sparse_offsets(py, &source, 8192).unwrap(),
                    (0..8193).collect::<Vec<_>>()
                );
                assert!(sparse_offsets(py, &source, 8191).is_err());
                for values in [vec![-1, 0, 1], vec![0, 2, 1]] {
                    let source = np.call_method("array", (values,), Some(&kw)).unwrap();
                    assert!(sparse_offsets(py, &source, 3).is_err());
                }
            }
            let kw = PyDict::new(py);
            kw.set_item("dtype", "int64").unwrap();
            let values = vec![0i64, 1 << 32, (1 << 32) + 7];
            let source = np.call_method("array", (&values,), Some(&kw)).unwrap();
            assert_eq!(
                sparse_offsets(py, &source, (1usize << 32) + 7).unwrap(),
                values
            );
        });
    }

    #[test]
    fn transpose_preserves_ieee_bits_across_vector_and_partition_tails() {
        Python::initialize();
        Python::attach(|py| {
            for workers in [1, 3, 0] {
                let previous = host_parallel::set_worker_limit(workers);
                for (rows, cols) in [
                    (1, 1),
                    (2, 3),
                    (3, 4),
                    (31, 33),
                    (65, 67),
                    (10003, 17),
                    (16385, 129),
                ] {
                    let f32_bits = [0, 0x80000000, 0x7fc00001, 0xffffffff, 1, 0x7f800000];
                    let f64_bits = [
                        0,
                        0x8000000000000000,
                        0x7ff8000000000001,
                        0xffffffffffffffff,
                        1,
                        0x7ff0000000000000,
                    ];
                    let source32: Vec<_> = (0..rows * cols)
                        .map(|i| f32::from_bits(f32_bits[i % f32_bits.len()]))
                        .collect();
                    let source64: Vec<_> = (0..rows * cols)
                        .map(|i| f64::from_bits(f64_bits[i % f64_bits.len()]))
                        .collect();
                    for fortran in [false, true] {
                        let output32 =
                            reorder(py, source32.clone(), [rows, cols], fortran).unwrap();
                        let output64 =
                            reorder(py, source64.clone(), [rows, cols], fortran).unwrap();
                        for row in 0..rows {
                            for col in 0..cols {
                                let (input, output) = if fortran {
                                    (col * rows + row, row * cols + col)
                                } else {
                                    (row * cols + col, col * rows + row)
                                };
                                assert_eq!(output32[output].to_bits(), source32[input].to_bits());
                                assert_eq!(output64[output].to_bits(), source64[input].to_bits());
                            }
                        }
                    }
                }
                host_parallel::set_worker_limit(previous);
            }
        });
    }

    #[test]
    fn dense_snapshots_preserve_values_and_both_orders() {
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            for rows in [0, 1, 33, 4097] {
                for cols in [0, 1, 17] {
                    for dtype in ["float32", "float64"] {
                        let kw = PyDict::new(py);
                        kw.set_item("dtype", dtype).unwrap();
                        let c = np
                            .call_method("arange", (rows * cols,), Some(&kw))
                            .unwrap()
                            .call_method1("reshape", ((rows, cols),))
                            .unwrap();
                        let f = np.call_method1("asfortranarray", (&c,)).unwrap();
                        for source in [&c, &f] {
                            for order in ["C", "F"] {
                                let packed = pack_dense(py, source, order).unwrap();
                                assert_eq!(packed.shape(), [rows, cols]);
                                let expected = source.call_method1("ravel", (order,)).unwrap();
                                match packed.values {
                                    FloatValues::F32(values) => assert_eq!(
                                        values,
                                        PyBuffer::<f32>::get(&expected)
                                            .unwrap()
                                            .to_vec(py)
                                            .unwrap()
                                    ),
                                    FloatValues::F64(values) => assert_eq!(
                                        values,
                                        PyBuffer::<f64>::get(&expected)
                                            .unwrap()
                                            .to_vec(py)
                                            .unwrap()
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn dense_snapshot_reads_only_the_bounded_strided_window() {
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            let kw = PyDict::new(py);
            kw.set_item("dtype", "float64").unwrap();
            let source = np
                .call_method("arange", (100_000,), Some(&kw))
                .unwrap()
                .call_method1("reshape", ((100, 1000),))
                .unwrap();
            let window = source
                .get_item((
                    pyo3::types::PySlice::new(py, 0, 100, 2),
                    pyo3::types::PySlice::new(py, 7, 39, 3),
                ))
                .unwrap();
            let packed = pack_dense(py, &window, "F").unwrap();
            let expected = window.call_method1("ravel", ("F",)).unwrap();
            assert_eq!(packed.shape(), [50, 11]);
            assert_eq!(packed.as_bytes().len(), 550 * 8);
            let FloatValues::F64(values) = packed.values else {
                panic!("float64 precision changed")
            };
            assert_eq!(
                values,
                PyBuffer::<f64>::get(&expected).unwrap().to_vec(py).unwrap()
            );
        });
    }

    #[test]
    fn dense_snapshot_handles_contiguous_axes_with_reversed_outer_strides() {
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            let slice = pyo3::types::PySlice::new;
            for dtype in ["float32", "float64"] {
                let options = PyDict::new(py);
                options.set_item("dtype", dtype).unwrap();
                let c = np
                    .call_method("arange", (100_000,), Some(&options))
                    .unwrap()
                    .call_method1("reshape", ((100, 1000),))
                    .unwrap();
                let f = np.call_method1("asfortranarray", (&c,)).unwrap();
                let windows = [
                    c.get_item((slice(py, 0, 100, 2), slice(py, 7, 39, 1)))
                        .unwrap(),
                    c.get_item((slice(py, 99, 1, -1), slice(py, 7, 39, 1)))
                        .unwrap(),
                    f.get_item((slice(py, 7, 39, 1), slice(py, 0, 1000, 3)))
                        .unwrap(),
                    f.get_item((slice(py, 7, 39, 1), slice(py, 999, 1, -1)))
                        .unwrap(),
                ];
                for window in windows {
                    for order in ["C", "F"] {
                        let packed = pack_dense(py, &window, order).unwrap();
                        let expected = window.call_method1("ravel", (order,)).unwrap();
                        match packed.values {
                            FloatValues::F32(values) => assert_eq!(
                                values,
                                PyBuffer::<f32>::get(&expected).unwrap().to_vec(py).unwrap()
                            ),
                            FloatValues::F64(values) => assert_eq!(
                                values,
                                PyBuffer::<f64>::get(&expected).unwrap().to_vec(py).unwrap()
                            ),
                        }
                    }
                }
            }
        });
    }

    /// Run explicitly on an otherwise idle CPU with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_dense_window_pack() {
        fn stages<T: Transpose + Element>(
            py: Python<'_>,
            window: &Bound<'_, PyAny>,
            dtype: &str,
            order: &str,
        ) {
            let buffer = PyBuffer::<T>::get(window).unwrap();
            let mut samples = Vec::new();
            for _ in 0..11 {
                let start = std::time::Instant::now();
                std::hint::black_box(snapshot(py, &buffer).unwrap());
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "native_axis_snapshot dtype={dtype} source={order} milliseconds={}",
                samples[5]
            );
            let (values, from_fortran) = snapshot(py, &buffer).unwrap();
            if !from_fortran {
                for workers in [1, 4, 0] {
                    let previous = host_parallel::set_worker_limit(workers);
                    let mut samples = Vec::new();
                    for _ in 0..11 {
                        // Prepare an owned input outside the transpose timer.
                        let values = values.clone();
                        let start = std::time::Instant::now();
                        std::hint::black_box(reorder(py, values, [10_000, 64], false).unwrap());
                        samples.push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    host_parallel::set_worker_limit(previous);
                    samples.sort_by(f64::total_cmp);
                    println!(
                        "owned_transpose dtype={dtype} source={order} workers={workers} milliseconds={}",
                        samples[5]
                    );
                }
            }
        }
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            for dtype in ["float32", "float64"] {
                for order in ["C", "F"] {
                    let options = PyDict::new(py);
                    options.set_item("dtype", dtype).unwrap();
                    options.set_item("order", order).unwrap();
                    let source = np
                        .call_method("ones", ((10_000, 1000),), Some(&options))
                        .unwrap();
                    let window = source
                        .get_item((
                            pyo3::types::PySlice::new(py, 0, 10_000, 1),
                            pyo3::types::PySlice::new(py, 7, 71, 1),
                        ))
                        .unwrap();
                    for workers in [1, 4, 0] {
                        let previous = host_parallel::set_worker_limit(workers);
                        for _ in 0..3 {
                            std::hint::black_box(pack_dense(py, &window, "F").unwrap());
                        }
                        let mut samples = Vec::new();
                        for _ in 0..11 {
                            let start = std::time::Instant::now();
                            std::hint::black_box(pack_dense(py, &window, "F").unwrap());
                            samples.push(start.elapsed().as_secs_f64() * 1000.0);
                        }
                        samples.sort_by(f64::total_cmp);
                        host_parallel::set_worker_limit(previous);
                        println!(
                            "pack_dense dtype={dtype} source={order} workers={workers} milliseconds={}",
                            samples[5]
                        );
                    }
                    match dtype {
                        "float32" => stages::<f32>(py, &window, dtype, order),
                        "float64" => stages::<f64>(py, &window, dtype, order),
                        _ => unreachable!(),
                    }
                    let options = PyDict::new(py);
                    options.set_item("copy", true).unwrap();
                    options.set_item("order", "F").unwrap();
                    let mut samples = Vec::new();
                    for _ in 0..11 {
                        let start = std::time::Instant::now();
                        std::hint::black_box(
                            np.call_method("array", (&window,), Some(&options)).unwrap(),
                        );
                        samples.push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    samples.sort_by(f64::total_cmp);
                    println!(
                        "numpy_F_snapshot dtype={dtype} source={order} milliseconds={}",
                        samples[5]
                    );
                }
            }
        });
    }
}
