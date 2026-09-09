//! Bounded stable radix sorting with native CUDA kernels and allocator-owned scratch.
use crate::{
    array::{Dtype, Layout},
    rank_support::*,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

/// Sort float32 or uint32 F-order columns, returning values or their original row indices.
/// All work uses the current CUDA stream; the caller owns synchronization.
pub fn sort<'py>(
    cp: &Bound<'py, PyModule>,
    input: &Bound<'py, PyAny>,
    indices: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let x = read(input, cp, "sort input", None, Layout::F)?;
    let (dtype, raw_keys) = match x.dtype {
        Dtype::F32 => ("float32", 0),
        Dtype::U32 => ("uint32", 256),
        _ => {
            return Err(PyTypeError::new_err(
                "sort input must have dtype float32 or uint32",
            ));
        }
    };
    let (rows, cols) = matrix(&x, "sort input")?;
    if rows > u32::MAX as u64 {
        return Err(PyValueError::new_err(
            "radix sort supports at most 2^32-1 rows per column",
        ));
    }
    let dims = [rows as usize, cols as usize];
    if rows <= 1 || cols == 0 {
        return if indices {
            empty(cp, &dims, "int64", "F", true)
        } else {
            Ok(input.clone())
        };
    }
    let s = stream(cp)?;
    let tiles = rows.div_ceil(256);
    let values = [
        empty(cp, &dims, dtype, "F", false)?,
        empty(cp, &dims, dtype, "F", false)?,
    ];
    let va = [
        read(&values[0], cp, "sort values", Some(x.dtype), Layout::F)?,
        read(&values[1], cp, "sort values", Some(x.dtype), Layout::F)?,
    ];
    let order = if indices {
        Some([
            empty(cp, &dims, "int64", "F", false)?,
            empty(cp, &dims, "int64", "F", false)?,
        ])
    } else {
        None
    };
    let oa = if let Some(order) = &order {
        Some([
            read(&order[0], cp, "sort indices", Some(Dtype::I64), Layout::F)?,
            read(&order[1], cp, "sort indices", Some(Dtype::I64), Layout::F)?,
        ])
    } else {
        None
    };
    let histogram = empty(
        cp,
        &[cols as usize, tiles as usize, 256],
        "uint32",
        "C",
        false,
    )?;
    let ha = read(
        &histogram,
        cp,
        "sort histogram",
        Some(Dtype::U32),
        Layout::C,
    )?;
    let local = empty(cp, &dims, "uint8", "F", false)?;
    let la = read(&local, cp, "sort local ranks", Some(Dtype::U8), Layout::F)?;
    for pass in 0..4 {
        let dst = pass % 2;
        let src = if pass == 0 { &x } else { &va[1 - dst] };
        let output = &va[dst];
        let shift = (pass as u32 * 8) | raw_keys;
        launch(
            cp,
            "rank_radix_hist",
            cols * tiles * 256,
            s,
            &[src, &ha, &la],
            &mut [
                Arg::P(src.pointer),
                Arg::P(ha.pointer),
                Arg::P(la.pointer),
                Arg::N(rows),
                Arg::N(cols),
                Arg::N(tiles),
                Arg::U(shift),
            ],
        )?;
        launch(
            cp,
            "rank_radix_prefix",
            cols * 256,
            s,
            &[&ha],
            &mut [Arg::P(ha.pointer), Arg::N(cols), Arg::N(tiles)],
        )?;
        launch(
            cp,
            "rank_radix_scatter",
            x.len,
            s,
            &[src, output, &ha, &la],
            &mut [
                Arg::P(src.pointer),
                Arg::P(
                    oa.as_ref()
                        .filter(|_| pass != 0)
                        .map_or(0, |a| a[1 - dst].pointer),
                ),
                Arg::P(output.pointer),
                Arg::P(oa.as_ref().map_or(0, |a| a[dst].pointer)),
                Arg::P(ha.pointer),
                Arg::P(la.pointer),
                Arg::N(rows),
                Arg::N(cols),
                Arg::N(tiles),
                Arg::U(shift),
            ],
        )?;
    }
    if let Some(order) = order {
        Ok(order[1].clone())
    } else {
        Ok(values[1].clone())
    }
}
