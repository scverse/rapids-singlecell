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
        if !object.is_instance(&cupy.getattr("ndarray")?)? {
            return Err(PyTypeError::new_err(format!(
                "{name} must be a CuPy device array"
            )));
        }
        let dtype_name = object
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?;
        let actual_dtype = match dtype_name.as_str() {
            "bool" => Dtype::Bool,
            "int8" => Dtype::I8,
            "uint8" => Dtype::U8,
            "int32" => Dtype::I32,
            "uint32" => Dtype::U32,
            "int64" => Dtype::I64,
            "uint64" => Dtype::U64,
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported {name} dtype: {dtype_name}"
                )));
            }
        };
        if let Some(expected) = dtype
            && actual_dtype != expected
        {
            return Err(PyTypeError::new_err(format!(
                "{name} must have dtype {}",
                expected.name()
            )));
        }
        let actual_shape = object.getattr("shape")?.extract::<Vec<usize>>()?;
        if let Some(expected) = shape
            && actual_shape != expected
        {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape {expected:?}"
            )));
        }
        let flags = object.getattr("flags")?;
        let c_contiguous = flags.getattr("c_contiguous")?.extract::<bool>()?;
        let f_contiguous = flags.getattr("f_contiguous")?.extract::<bool>()?;
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
        let data = object.getattr("data")?;
        let pointer = data.getattr("ptr")?.extract::<u64>()?;
        let allocation = data.getattr("mem")?;
        let start = allocation.getattr("ptr")?.extract::<u64>()?;
        let size = allocation.getattr("size")?.extract::<u64>()?;
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
            device: object.getattr("device")?.getattr("id")?.extract()?,
            dtype: actual_dtype,
            shape: actual_shape,
            len,
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
    let device = cupy
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("getDevice")?
        .extract()?;
    if arrays.iter().any(|array| array.device != device) {
        return Err(PyValueError::new_err(
            "all arrays must be on the current CUDA device",
        ));
    }
    Ok(device)
}
