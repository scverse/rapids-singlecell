//! Validated, non-owning descriptions of CuPy device allocations.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

pub enum Layout {
    C,
    F,
    Contiguous,
}

pub struct Array {
    pub pointer: u64,
    pub device: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub len: u64,
    pub c_contiguous: bool,
    end: u64,
}

impl Array {
    pub fn read(
        object: &Bound<'_, PyAny>,
        cupy: &Bound<'_, PyModule>,
        name: &str,
        dtype: Option<Dtype>,
        shape: Option<&[usize]>,
        layout: Layout,
    ) -> PyResult<Self> {
        let py = object.py();
        if !object.is_instance(&cupy.getattr(pyo3::intern!(py, "ndarray"))?)? {
            return Err(PyTypeError::new_err(format!(
                "{name} must be a CuPy device array"
            )));
        }
        let metadata = metadata(object, cupy, name)?;
        let actual_dtype = metadata.dtype;
        if let Some(expected) = dtype
            && actual_dtype != expected
        {
            return Err(PyTypeError::new_err(format!(
                "{name} must have dtype {}",
                expected.name()
            )));
        }
        let actual_shape = metadata.shape;
        if let Some(expected) = shape
            && actual_shape != expected
        {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape {expected:?}"
            )));
        }
        let c_contiguous = metadata.c_contiguous;
        let f_contiguous = metadata.f_contiguous;
        let contiguous = match layout {
            Layout::C => c_contiguous,
            Layout::F => f_contiguous,
            Layout::Contiguous => c_contiguous || f_contiguous,
        };
        if !contiguous {
            return Err(PyValueError::new_err(format!(
                "{name} has an unsupported memory layout"
            )));
        }
        let len = actual_shape
            .iter()
            .try_fold(1_u64, |size, &n| size.checked_mul(n as u64))
            .ok_or_else(|| {
                PyValueError::new_err(format!("{name} shape exceeds addressable memory"))
            })?;
        let bytes = len
            .checked_mul(actual_dtype.size())
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or_else(|| {
                PyValueError::new_err(format!("{name} size exceeds addressable memory"))
            })?;
        let data = object.getattr(pyo3::intern!(py, "data"))?;
        let pointer = metadata.pointer;
        let allocation = data.getattr(pyo3::intern!(py, "mem"))?;
        let start = allocation
            .getattr(pyo3::intern!(py, "ptr"))?
            .extract::<u64>()?;
        let size = allocation
            .getattr(pyo3::intern!(py, "size"))?
            .extract::<u64>()?;
        let end = pointer
            .checked_add(bytes)
            .ok_or_else(|| PyValueError::new_err(format!("invalid {name} address range")))?;
        if bytes > 0 {
            if pointer == 0 || pointer % actual_dtype.size() != 0 {
                return Err(PyValueError::new_err(format!(
                    "{name} has an invalid or unaligned device pointer"
                )));
            }
            // Stride tricks can describe a contiguous view beyond its storage.
            if !pointer
                .checked_sub(start)
                .and_then(|offset| offset.checked_add(bytes))
                .is_some_and(|extent| extent <= size)
            {
                return Err(PyValueError::new_err(format!(
                    "{name} extends beyond its CUDA allocation"
                )));
            }
        }
        Ok(Self {
            pointer,
            device: metadata.device,
            dtype: actual_dtype,
            shape: actual_shape,
            len,
            c_contiguous,
            end,
        })
    }

    pub fn require_vector(&self, name: &str) -> PyResult<()> {
        if self.shape.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be one-dimensional"
            )));
        }
        Ok(())
    }

    pub fn require_disjoint(&self, other: &Self) -> PyResult<()> {
        if self.len > 0 && other.len > 0 && self.pointer < other.end && other.pointer < self.end {
            return Err(PyValueError::new_err(
                "output must not overlap any input array",
            ));
        }
        Ok(())
    }
}

pub fn current_device(cupy: &Bound<'_, PyModule>, arrays: &[&Array]) -> PyResult<usize> {
    let py = cupy.py();
    let device = cupy
        .getattr(pyo3::intern!(py, "cuda"))?
        .getattr(pyo3::intern!(py, "runtime"))?
        .call_method0(pyo3::intern!(py, "getDevice"))?
        .extract()?;
    if arrays.iter().any(|array| array.device != device) {
        return Err(PyValueError::new_err(
            "all arrays must be on the current CUDA device",
        ));
    }
    Ok(device)
}

// The stable, pre-1.0 DLPack ABI. CuPy returns this layout when max_version is
// omitted. Only metadata is borrowed; we never consume the capsule or take
// ownership of a device allocation. See https://dmlc.github.io/dlpack/latest/.
#[repr(C)]
struct DlDevice {
    kind: i32,
    index: i32,
}
#[repr(C)]
struct DlDtype {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[repr(C)]
struct DlTensor {
    data: *mut std::ffi::c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDtype,
    shape: *const i64,
    strides: *const i64,
    byte_offset: u64,
}
struct Metadata {
    pointer: u64,
    device: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    c_contiguous: bool,
    f_contiguous: bool,
}

fn metadata(
    object: &Bound<'_, PyAny>,
    cupy: &Bound<'_, PyModule>,
    name: &str,
) -> PyResult<Metadata> {
    use pyo3::types::{PyCapsule, PyCapsuleMethods, PyDict};
    let py = object.py();
    let options = PyDict::new(py);
    options.set_item(pyo3::intern!(py, "stream"), -1)?;
    // Invoke CuPy's descriptor directly: subclasses cannot substitute a capsule
    // with a different native layout. -1 requests metadata without stream waits.
    let capsule = cupy
        .getattr(pyo3::intern!(py, "ndarray"))?
        .getattr(pyo3::intern!(py, "__dlpack__"))?
        .call((object,), Some(&options))?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let pointer = capsule.pointer_checked(Some(c"dltensor"))?;
    // SAFETY: the native CuPy descriptor created this named legacy capsule. Its
    // first member is DLTensor, with metadata alive until the capsule is dropped.
    // No Python callback is invoked while its shape/stride pointers are borrowed.
    let tensor = unsafe { &*pointer.as_ptr().cast::<DlTensor>() };
    if tensor.ndim < 0 || tensor.device.index < 0 || !matches!(tensor.device.kind, 2 | 13) {
        return Err(PyValueError::new_err(format!(
            "invalid {name} CUDA tensor metadata"
        )));
    }
    let dtype = match (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes) {
        (0, 8, 1) => Dtype::I8,
        (0, 32, 1) => Dtype::I32,
        (0, 64, 1) => Dtype::I64,
        (1, 8, 1) => Dtype::U8,
        (1, 32, 1) => Dtype::U32,
        (1, 64, 1) => Dtype::U64,
        (2, 32, 1) => Dtype::F32,
        (2, 64, 1) => Dtype::F64,
        (6, 8, 1) => Dtype::Bool,
        _ => return Err(PyTypeError::new_err(format!("unsupported {name} dtype"))),
    };
    let dimensions = tensor.ndim as usize;
    if dimensions > 0 && tensor.shape.is_null() {
        return Err(PyValueError::new_err(format!("missing {name} shape")));
    }
    let shape = if dimensions == 0 {
        &[][..]
    } else {
        // SAFETY: CuPy owns an ndim-element shape vector for this capsule.
        unsafe { std::slice::from_raw_parts(tensor.shape, dimensions) }
    };
    let shape = shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| PyValueError::new_err(format!("invalid {name} shape")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let (c_contiguous, f_contiguous) = if shape.contains(&0) {
        (true, true)
    } else if tensor.strides.is_null() {
        (true, shape.iter().filter(|&&dim| dim > 1).count() <= 1)
    } else {
        // SAFETY: non-null CuPy strides have exactly ndim signed elements.
        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, dimensions) };
        let contiguous = |reverse: bool| {
            let mut expected = 1usize;
            for position in 0..dimensions {
                let axis = if reverse {
                    dimensions - 1 - position
                } else {
                    position
                };
                if shape[axis] > 1 {
                    if usize::try_from(strides[axis]).ok() != Some(expected) {
                        return false;
                    }
                    let Some(next) = expected.checked_mul(shape[axis]) else {
                        return false;
                    };
                    expected = next;
                }
            }
            true
        };
        (contiguous(true), contiguous(false))
    };
    let pointer = (tensor.data as u64)
        .checked_add(tensor.byte_offset)
        .ok_or_else(|| PyValueError::new_err(format!("invalid {name} address range")))?;
    // DLPack permits managed memory to report device 0 independent of its
    // allocating CUDA device; retain CuPy's actual device for those allocations.
    let device = if tensor.device.kind == 13 {
        object
            .getattr(pyo3::intern!(py, "device"))?
            .getattr(pyo3::intern!(py, "id"))?
            .extract()?
    } else {
        tensor.device.index as usize
    };
    Ok(Metadata {
        pointer,
        device,
        dtype,
        shape,
        c_contiguous,
        f_contiguous,
    })
}
