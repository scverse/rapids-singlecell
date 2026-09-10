//! Bounded pinned staging with independent, reusable upload/compute slots.
//!
//! Each slot owns its stream, pinned storage and device allocations. Reusing a
//! slot waits only for that slot; CPU packing can overlap other slots' kernels.
use crate::{array::Dtype, host_buffer, rank_support, runtime::StreamScope};
use cuda_core::{IntoResult, sys};
use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::cell::Cell;

fn copy_contiguous_buffer<T: Element>(
    py: Python<'_>,
    buffer: &PyBuffer<T>,
    destination: usize,
    bytes: usize,
) -> PyResult<()> {
    if !(buffer.is_fortran_contiguous() || buffer.is_c_contiguous()) || buffer.len_bytes() != bytes
    {
        return Err(PyValueError::new_err(
            "pinned staging requires a contiguous complete buffer",
        ));
    }
    let mut length = bytes as isize;
    let mut stride = 1isize;
    // SAFETY: the typed export remains alive and attached. Its contiguous
    // storage contains exactly bytes initialized bytes; destination is a
    // separate private pinned allocation with at least that capacity. The
    // temporary descriptor borrows storage and is never released as an owner.
    let status = unsafe {
        let view = pyo3::ffi::Py_buffer {
            buf: buffer.buf_ptr(),
            obj: std::ptr::null_mut(),
            len: length,
            itemsize: 1,
            readonly: 1,
            ndim: 1,
            format: std::ptr::null_mut(),
            shape: &mut length,
            strides: &mut stride,
            suboffsets: std::ptr::null_mut(),
            internal: std::ptr::null_mut(),
        };
        pyo3::ffi::PyBuffer_ToContiguous(destination as *mut _, &view, length, b'C' as _)
    };
    if status != 0 {
        Err(PyErr::fetch(py))
    } else {
        Ok(())
    }
}

/// Copy contiguous rows or columns directly into private pinned storage.
/// The outer stride may be negative or include gaps in the source window.
#[inline]
unsafe fn copy_spans<const SPAN: usize>(
    source: *const u8,
    destination: *mut u8,
    rows: usize,
    stride: isize,
    span: usize,
) {
    let span = if SPAN == 0 { span } else { SPAN };
    for row in 0..rows {
        // SAFETY: the attached caller validated the entire signed row range
        // and separate private destination. No Rust references to the foreign
        // source are created; copying bytes preserves all floating bit patterns.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.offset(row as isize * stride),
                destination.add(row * span),
                span,
            );
        }
    }
}

fn copy_axis_buffer<T: Element>(
    py: Python<'_>,
    buffer: &PyBuffer<T>,
    axis: usize,
    destination: usize,
    bytes: usize,
) -> PyResult<()> {
    if (axis == 0 && buffer.is_fortran_contiguous()) || (axis == 1 && buffer.is_c_contiguous()) {
        return copy_contiguous_buffer(py, buffer, destination, bytes);
    }
    let item_size = std::mem::size_of::<T>() as isize;
    if buffer.dimensions() != 2
        || buffer.suboffsets().is_some()
        || buffer.strides()[axis] != item_size
        || buffer.len_bytes() != bytes
    {
        return Err(PyValueError::new_err("invalid contiguous staging axis"));
    }
    let shape = buffer.shape();
    let outer = shape[1 - axis];
    let stride = buffer.strides()[1 - axis];
    let span_bytes = shape[axis]
        .checked_mul(item_size as usize)
        .ok_or_else(|| PyValueError::new_err("staging span size overflow"))?;
    if outer.checked_mul(span_bytes) != Some(bytes)
        || isize::try_from(outer.saturating_sub(1))
            .ok()
            .and_then(|last| last.checked_mul(stride))
            .is_none()
    {
        return Err(PyValueError::new_err("staging span address overflow"));
    }
    // Keep the export attached throughout every source read. A tight span
    // loop avoids CPython's per-item multidimensional pointer traversal.
    // Common spans use constant-size copies so the compiler can emit SIMD
    // loads/stores; arbitrary widths retain the same bounded byte copy.
    // SAFETY: the live typed export covers each checked span and destination
    // is a separate pinned allocation covering bytes, validated by upload_with.
    unsafe {
        let source = buffer.buf_ptr().cast::<u8>();
        let destination = destination as *mut u8;
        match span_bytes {
            128 => copy_spans::<128>(source, destination, outer, stride, span_bytes),
            256 => copy_spans::<256>(source, destination, outer, stride, span_bytes),
            _ => copy_spans::<0>(source, destination, outer, stride, span_bytes),
        }
    }
    Ok(())
}

struct Buffer<'py> {
    pinned: Bound<'py, PyAny>,
    device: Bound<'py, PyAny>,
    fortran: Option<Bound<'py, PyAny>>,
    dtype: Dtype,
    capacity: usize,
    uploaded: bool,
}

struct Slot<'py> {
    stream: Bound<'py, PyAny>,
    pointer: usize,
    buffers: Vec<Option<Buffer<'py>>>,
    allocations: Vec<Bound<'py, PyAny>>,
    retained: Vec<Bound<'py, PyAny>>,
    pending: Cell<bool>,
}

impl<'py> Slot<'py> {
    fn new(stream: Bound<'py, PyAny>) -> PyResult<Self> {
        let pointer = stream.getattr("ptr")?.extract()?;
        Ok(Self {
            stream,
            pointer,
            buffers: Vec::new(),
            allocations: Vec::new(),
            retained: Vec::new(),
            pending: Cell::new(false),
        })
    }

    fn synchronize(&self) -> PyResult<()> {
        if !self.pending.get() {
            return Ok(());
        }
        let pointer = self.pointer;
        // SAFETY: this slot owns the stream for the duration of the detached
        // wait. No Python data or APIs are touched without the interpreter.
        let result = self
            .stream
            .py()
            .detach(move || unsafe { sys::cuStreamSynchronize(pointer as sys::CUstream).result() })
            .map_err(|error| PyRuntimeError::new_err(format!("CUDA staging wait: {error}")));
        if result.is_ok() {
            self.pending.set(false);
        }
        result
    }
}

pub struct BatchStreams<'py> {
    cp: Bound<'py, PyModule>,
    slots: Vec<Slot<'py>>,
    // Reuse the caller's allocation pool across calls. This separate guard
    // also owns allocations if recording their readiness or ordering a worker
    // fails after an asynchronous allocator has queued work.
    allocator: Slot<'py>,
    // Keep the readiness event alive until all streams have passed it.
    ready: Option<Bound<'py, PyAny>>,
}

impl<'py> BatchStreams<'py> {
    /// Retain scratch on the caller's stream without creating another stream.
    /// The same completion/exception guarantees apply to a single-slot path.
    pub fn caller(cp: &Bound<'py, PyModule>) -> PyResult<Self> {
        let stream = cp.getattr("cuda")?.call_method0("get_current_stream")?;
        Ok(Self {
            cp: cp.clone(),
            slots: vec![Slot::new(stream.clone())?],
            allocator: Slot::new(stream)?,
            ready: None,
        })
    }

    /// Create a bounded ring after shared inputs/output initialization has been
    /// enqueued on the caller's stream. Callers budget scratch for every slot.
    pub fn new(cp: &Bound<'py, PyModule>, slots: usize) -> PyResult<Self> {
        if !(1..=4).contains(&slots) {
            return Err(PyValueError::new_err(
                "staging requires between one and four slots",
            ));
        }
        let cuda = cp.getattr("cuda")?;
        let allocator = Slot::new(cuda.call_method0("get_current_stream")?)?;
        let options = PyDict::new(cp.py());
        options.set_item("disable_timing", true)?;
        let ready = cuda.getattr("Event")?.call((), Some(&options))?;
        ready.call_method0("record")?;
        let options = PyDict::new(cp.py());
        options.set_item("non_blocking", true)?;
        let mut ring = Self {
            cp: cp.clone(),
            slots: Vec::with_capacity(slots),
            allocator,
            ready: Some(ready),
        };
        for _ in 0..slots {
            let stream = cuda.getattr("Stream")?.call((), Some(&options))?;
            stream.call_method1(
                "wait_event",
                (ring.ready.as_ref().expect("readiness event initialized"),),
            )?;
            ring.slots.push(Slot::new(stream)?);
        }
        Ok(ring)
    }

    pub fn enter(&mut self, index: usize) -> PyResult<StreamScope<'py>> {
        let slot = &mut self.slots[index];
        slot.synchronize()?;
        slot.retained.clear();
        slot.allocations.clear();
        for buffer in slot.buffers.iter_mut().flatten() {
            buffer.uploaded = false;
        }
        slot.pending.set(true);
        StreamScope::enter(slot.stream.clone())
    }

    pub fn retain(&mut self, index: usize, object: Bound<'py, PyAny>) {
        self.slots[index].pending.set(true);
        self.slots[index].retained.push(object);
    }

    /// Allocate reusable device storage through the caller's stream pool.
    /// Workers wait for allocator work before the first use, including with
    /// stream-ordered allocators. The slot owns the result before returning.
    pub fn allocate(
        &mut self,
        index: usize,
        dims: &[usize],
        dtype: Dtype,
        order: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _scope = StreamScope::enter(self.allocator.stream.clone())?;
        self.allocator.pending.set(true);
        let output = rank_support::empty(&self.cp, dims, dtype.name(), order, false)?;
        self.allocator.retained.push(output.clone());
        let slot = &mut self.slots[index];
        slot.pending.set(true);
        if slot.pointer != self.allocator.pointer {
            let options = PyDict::new(self.cp.py());
            options.set_item("disable_timing", true)?;
            let event = self
                .cp
                .getattr("cuda")?
                .getattr("Event")?
                .call((), Some(&options))?;
            self.allocator.retained.push(event.clone());
            event.call_method0("record")?;
            slot.stream.call_method1("wait_event", (&event,))?;
            slot.retained.push(event);
        }
        slot.allocations.push(output.clone());
        self.allocator.retained.clear();
        // The selected slot now covers every queued allocator operation.
        self.allocator.pending.set(false);
        Ok(output)
    }

    /// Hand a successful allocation to a cache with its own completion guard.
    /// The cache must keep that owner alive through slot completion and wait
    /// before replacing it. Failed constructors leave ownership in the slot.
    pub fn transfer_allocation(&mut self, index: usize, object: &Bound<'py, PyAny>) {
        let allocations = &mut self.slots[index].allocations;
        if let Some(position) = allocations.iter().position(|owner| owner.is(object)) {
            allocations.swap_remove(position);
        }
    }

    /// Snapshot a bounded NumPy matrix, preserve its original floating dtype,
    /// and enqueue an F-order upload through private pinned memory.
    pub fn upload_dense(
        &mut self,
        index: usize,
        field: usize,
        source: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.upload_dense_order(index, field, source, true)
    }

    /// Preserve contiguous host layout unless the consumer requires F-order.
    /// Both layouts use the same private pinned storage and exception cleanup.
    pub fn upload_dense_order(
        &mut self,
        index: usize,
        field: usize,
        source: &Bound<'py, PyAny>,
        force_fortran: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.cp.py();
        let np = py.import("numpy")?;
        if !source.is_instance(&np.getattr("ndarray")?)? {
            return Err(PyTypeError::new_err("host staging requires a NumPy array"));
        }
        let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
        // Copy the contiguous axis directly into private pinned storage. For
        // C-order windows the GPU performs the layout conversion while other
        // slots continue computing. Both device allocations belong to the slot
        // before their first asynchronous use, including exceptional returns.
        macro_rules! direct {
            ($ty:ty, $dtype:expr) => {{
                let buffer = PyBuffer::<$ty>::get(source)?;
                if buffer.dimensions() != 2 {
                    return Err(PyValueError::new_err(
                        "host staging requires two dimensions",
                    ));
                }
                let axis = if buffer.is_fortran_contiguous()
                    || (buffer.suboffsets().is_none()
                        && buffer.strides()[0] == std::mem::size_of::<$ty>() as isize)
                {
                    Some(0)
                } else if buffer.is_c_contiguous()
                    || (buffer.suboffsets().is_none()
                        && buffer.strides()[1] == std::mem::size_of::<$ty>() as isize)
                {
                    Some(1)
                } else {
                    None
                };
                if let Some(axis) = axis {
                    let dims = buffer.shape().to_vec();
                    let uploaded = self.upload_with(
                        index,
                        field,
                        $dtype,
                        &dims,
                        if axis == 0 { "F" } else { "C" },
                        |destination, bytes| {
                            copy_axis_buffer(py, &buffer, axis, destination, bytes)
                        },
                    )?;
                    if axis == 0 || !force_fortran {
                        return Ok(uploaded);
                    }
                    let slot_buffer = self.slots[index].buffers[field]
                        .as_ref()
                        .expect("uploaded buffer initialized");
                    if slot_buffer.fortran.is_none() {
                        let converted =
                            self.allocate(index, &[slot_buffer.capacity], $dtype, "C")?;
                        self.slots[index].buffers[field]
                            .as_mut()
                            .expect("uploaded buffer initialized")
                            .fortran = Some(converted);
                    }
                    let elements = buffer.item_count();
                    let view = self.slots[index].buffers[field]
                        .as_ref()
                        .expect("uploaded buffer initialized")
                        .fortran
                        .as_ref()
                        .expect("conversion buffer initialized")
                        .get_item(rank_support::slice(py, 0, elements))?;
                    let options = PyDict::new(py);
                    options.set_item("order", "F")?;
                    let output = view.call_method("reshape", (dims,), Some(&options))?;
                    self.cp.call_method1("copyto", (&output, &uploaded))?;
                    return Ok(output);
                }
            }};
        }
        match dtype.as_str() {
            "float32" => direct!(f32, Dtype::F32),
            "float64" => direct!(f64, Dtype::F64),
            _ => {
                return Err(PyTypeError::new_err(
                    "host staging dtype must be float32 or float64",
                ));
            }
        }
        let order = if force_fortran { "F" } else { "C" };
        let data = host_buffer::pack_dense(py, source, order)?;
        self.upload(
            index,
            field,
            data.as_bytes(),
            data.dtype(),
            &data.shape(),
            order,
        )
    }

    /// Upload compact CSC data/offsets without narrowing their native dtype.
    pub fn upload_vector(
        &mut self,
        index: usize,
        field: usize,
        source: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.cp.py();
        let np = py.import("numpy")?;
        if !source.is_instance(&np.getattr("ndarray")?)?
            || source.getattr("ndim")?.extract::<usize>()? != 1
            || !source
                .getattr("flags")?
                .getattr("c_contiguous")?
                .extract::<bool>()?
        {
            return Err(PyTypeError::new_err(
                "sparse staging requires a contiguous NumPy vector",
            ));
        }
        let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
        let dtype = match dtype.as_str() {
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            "int32" => Dtype::I32,
            "int64" => Dtype::I64,
            _ => return Err(PyTypeError::new_err("unsupported sparse staging dtype")),
        };
        if !source
            .getattr("dtype")?
            .getattr("isnative")?
            .extract::<bool>()?
        {
            return Err(PyTypeError::new_err(
                "sparse staging requires native byte order",
            ));
        }
        let view = source.call_method1("view", ("uint8",))?;
        let bytes = PyBuffer::<u8>::get(&view)?.to_vec(py)?;
        self.upload(index, field, &bytes, dtype, &[source.len()?], "C")
    }

    pub(crate) fn upload(
        &mut self,
        index: usize,
        field: usize,
        bytes: &[u8],
        dtype: Dtype,
        dims: &[usize],
        order: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| PyValueError::new_err("staging shape overflow"))?;
        if elements.checked_mul(dtype.size() as usize) != Some(bytes.len()) {
            return Err(PyValueError::new_err("staging size does not match shape"));
        }
        let py = self.cp.py();
        self.upload_with(index, field, dtype, dims, order, |destination, _| {
            // SAFETY: destination belongs to the synchronized private slot and
            // covers bytes.len(); source is an immutable, owned snapshot.
            py.detach(|| unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination as *mut u8, bytes.len());
            });
            Ok(())
        })
    }

    fn upload_with(
        &mut self,
        index: usize,
        field: usize,
        dtype: Dtype,
        dims: &[usize],
        order: &str,
        fill: impl FnOnce(usize, usize) -> PyResult<()>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| PyValueError::new_err("staging shape overflow"))?;
        let bytes = elements
            .checked_mul(dtype.size() as usize)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| PyValueError::new_err("staging byte size overflow"))?;
        let slot = &mut self.slots[index];
        slot.pending.set(true);
        if slot.buffers.len() <= field {
            slot.buffers.resize_with(field + 1, || None);
        }
        if slot.buffers[field].as_ref().is_some_and(|b| b.uploaded) {
            return Err(PyValueError::new_err(
                "staging field already uploaded in this batch",
            ));
        }
        if slot.buffers[field]
            .as_ref()
            .is_none_or(|b| b.capacity < elements || b.dtype != dtype)
        {
            // Entering this slot waited for its previous batch and cleared
            // retained views; the uploaded check excludes current-batch data.
            // Release completed storage before allocating its replacement so
            // capacity growth does not temporarily double the staging budget.
            drop(slot.buffers[field].take());
            let pinned = self
                .cp
                .getattr("cuda")?
                .call_method1("alloc_pinned_memory", (bytes,))?;
            let device = self.allocate(index, &[elements], dtype, "C")?;
            self.slots[index].buffers[field] = Some(Buffer {
                pinned,
                device,
                fortran: None,
                dtype,
                capacity: elements,
                uploaded: false,
            });
        }
        let slot = &mut self.slots[index];
        let buffer = slot.buffers[field]
            .as_mut()
            .expect("buffer initialized above");
        buffer.uploaded = true;
        if bytes != 0 {
            let host: usize = buffer.pinned.getattr("ptr")?.extract()?;
            let device: u64 = buffer.device.getattr("data")?.getattr("ptr")?.extract()?;
            fill(host, bytes)?;
            // SAFETY: both allocations belong to this slot and remain alive
            // through its next synchronization, including exceptional returns.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    device,
                    host as *const _,
                    bytes,
                    slot.pointer as sys::CUstream,
                )
            }
            .result()
            .map_err(|error| PyRuntimeError::new_err(format!("CUDA staging upload: {error}")))?;
        }
        let view = buffer
            .device
            .get_item(rank_support::slice(self.cp.py(), 0, elements))?;
        let kwargs = PyDict::new(self.cp.py());
        kwargs.set_item("order", order)?;
        view.call_method("reshape", (dims.to_vec(),), Some(&kwargs))
    }

    pub fn finish(&self) -> PyResult<()> {
        // Wait every stream even after an error so no buffer is freed early.
        let mut result = Ok(());
        for slot in &self.slots {
            if let Err(error) = slot.synchronize()
                && result.is_ok()
            {
                result = Err(error);
            }
        }
        if let Err(error) = self.allocator.synchronize()
            && result.is_ok()
        {
            result = Err(error);
        }
        result
    }
}

impl Drop for BatchStreams<'_> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytes;

    #[test]
    fn staging_spans_preserve_bits_and_signed_outer_strides() {
        fn check<T: Element>(
            py: Python<'_>,
            source: &Bound<'_, PyAny>,
            axis: usize,
        ) -> PyResult<()> {
            let buffer = PyBuffer::<T>::get(source)?;
            let expected = source.call_method1("tobytes", (if axis == 1 { "C" } else { "F" },))?;
            let mut output = vec![0u8; buffer.len_bytes()];
            copy_axis_buffer(
                py,
                &buffer,
                axis,
                output.as_mut_ptr() as usize,
                output.len(),
            )?;
            assert_eq!(output, expected.cast::<PyBytes>()?.as_bytes());
            Ok(())
        }
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let data = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(r#"import numpy as np
bits = {
    'float32': np.array([0, 1, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000], dtype=np.uint32),
    'float64': np.array([0, 1, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8123456789abc, 0x3ff0000000000000], dtype=np.uint64),
}
cases = {}
for dtype, pattern in bits.items():
    x = np.resize(pattern, (19, 17)).view(dtype)
    f = np.asfortranarray(x)
    cases[dtype] = [(x, 1), (f, 0), (x[::-1, 2:13], 1), (x[::3, 1:16], 1), (f[2:17, ::-2], 0), (f[1:18, ::2], 0)]
    wide = np.resize(pattern, (97, 97)).view(dtype)
    wide_f = np.asfortranarray(wide)
    for span_bytes in (128, 256):
        size = span_bytes // wide.itemsize
        cases[dtype].extend([(wide[::2, 1:1+size], 1), (wide[::-3, 1:1+size], 1), (wide_f[1:1+size, ::3], 0), (wide_f[1:1+size, ::-3], 0)])
"#),
                Some(&data),
                None,
            )?;
            let cases = data.get_item("cases")?.expect("test cases initialized");
            for dtype in ["float32", "float64"] {
                for case in cases.get_item(dtype)?.try_iter()? {
                    let case = case?;
                    let source = case.get_item(0)?;
                    let axis = case.get_item(1)?.extract()?;
                    if dtype == "float32" {
                        check::<f32>(py, &source, axis)?;
                    } else {
                        check::<f64>(py, &source, axis)?;
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }
}
