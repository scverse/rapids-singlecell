//! Dense streaming over shared, once-validated device operands.
use super::*;

fn offset_pointer(array: &Array, first: u64, count: u64) -> PyResult<u64> {
    if first.checked_add(count).is_none_or(|last| last > array.len) {
        return Err(PyValueError::new_err(
            "streaming operand window exceeds its allocation",
        ));
    }
    first
        .checked_mul(array.dtype.size())
        .and_then(|bytes| array.pointer.checked_add(bytes))
        .ok_or_else(|| PyValueError::new_err("streaming operand address overflow"))
}

impl ValidatedStream {
    /// The original argument owners outlive the entire streaming ring. Only
    /// the fresh private block needs metadata validation in each batch.
    fn dense_block(
        &self,
        cp: &Bound<'_, PyModule>,
        object: &Bound<'_, PyAny>,
        expected: [usize; 2],
        dtype: Dtype,
    ) -> PyResult<Array> {
        let block = read(
            object,
            cp,
            "streaming block",
            Some(dtype),
            Layout::Contiguous,
        )?;
        if block.shape != expected {
            return Err(PyValueError::new_err("streaming block changed shape"));
        }
        for output in self.outputs.iter().flatten().chain(self.histogram.iter()) {
            output.require_disjoint(&block)?;
        }
        Ok(block)
    }

    fn consume_dense(
        &self,
        cp: &Bound<'_, PyModule>,
        block: &Array,
        row_start: u64,
        col_start: u64,
        full_cols: u64,
        histogram: Option<&Histogram<'_, '_>>,
        stream: usize,
    ) -> PyResult<()> {
        let (rows, cols) = matrix(block, "streaming block")?;
        if col_start
            .checked_add(cols)
            .is_none_or(|last| last > full_cols)
        {
            return Err(PyValueError::new_err(
                "streaming output window exceeds its columns",
            ));
        }
        let codes = offset_pointer(&self.codes, row_start, rows)?;
        let mask = self
            .mask
            .as_ref()
            .map(|mask| offset_pointer(mask, row_start, rows))
            .transpose()?;
        let mut arrays = vec![block, &self.codes];
        arrays.extend(self.mask.iter());
        arrays.extend(self.outputs.iter().flatten());
        arrays.extend(self.histogram.iter());
        if let Some(histogram) = histogram {
            // Column histograms require the private F-order block supplied by
            // staging. Degenerate C/F blocks have identical storage order.
            if block.c_contiguous && rows > 1 && cols > 1 {
                return Err(PyValueError::new_err(
                    "histogram staging requires column-major storage",
                ));
            }
            let plane = product(&[histogram.groups, histogram.bins + 1])?;
            let first = col_start
                .checked_sub(histogram.start as u64)
                .ok_or_else(|| PyValueError::new_err("invalid histogram streaming window"))?;
            let output = offset_pointer(
                self.histogram
                    .as_ref()
                    .expect("histogram validated before streaming"),
                product(&[first, plane])?,
                product(&[cols, plane])?,
            )?;
            launch(
                cp,
                "rank_hist_dense",
                product(&[cols, 256])?,
                stream,
                &arrays,
                &mut [
                    Arg::P(block.pointer),
                    Arg::P(codes),
                    Arg::P(output),
                    Arg::N(rows),
                    Arg::N(cols),
                    Arg::N(histogram.groups),
                    Arg::N(histogram.bins),
                    Arg::F(histogram.low),
                    Arg::F(histogram.inverse),
                    Arg::U(wide(block.dtype)?),
                    Arg::U(0),
                ],
            )?;
        }
        if self.outputs.iter().any(Option::is_some) {
            let pointer = |index: usize| {
                self.outputs[index]
                    .as_ref()
                    .map_or(0, |array| array.pointer)
            };
            launch(
                cp,
                "rank_stats",
                block.len,
                stream,
                &arrays,
                &mut [
                    Arg::P(block.pointer),
                    Arg::P(codes),
                    Arg::P(mask.unwrap_or(0)),
                    Arg::P(pointer(0)),
                    Arg::P(pointer(2)),
                    Arg::P(pointer(1)),
                    Arg::P(0),
                    Arg::P(0),
                    Arg::N(rows),
                    Arg::N(cols),
                    Arg::N(self.groups),
                    Arg::N(full_cols),
                    Arg::N(col_start),
                    Arg::U(wide(block.dtype)?),
                    Arg::U(u32::from(block.c_contiguous) | 2),
                ],
            )?;
        }
        Ok(())
    }
}

pub(super) fn dense_host<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    sums: Option<&Bound<'py, PyAny>>,
    counts: Option<&Bound<'py, PyAny>>,
    squares: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    requested: isize,
    hist: Option<Histogram<'_, 'py>>,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let np = py.import("numpy")?;
    if !X.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("X must be a NumPy host array"));
    }
    // The original ndarray binding uses buffer storage directly. Normalize
    // subclasses so overridden slicing cannot change planned window extents.
    let X = np.call_method1("asarray", (X,))?;
    let (dims, dtype) = host_array(&X, "X", false)?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err("X must be two-dimensional"));
    }
    let rows = dims[0];
    let cols = dims[1];
    let operands = validate_stream(
        &cp,
        codes,
        mask,
        [sums, counts, squares],
        rows,
        cols,
        hist.as_ref(),
    )?;
    let device_dtype = if dtype == "float32" {
        Dtype::F32
    } else {
        Dtype::F64
    };
    // The original C overload wins when both flags are true (one row/column).
    let fortran = !X
        .getattr("flags")?
        .getattr("c_contiguous")?
        .extract::<bool>()?;
    if cols == 0 || (rows == 0 && hist.is_none()) {
        return Ok(());
    }
    let columns = fortran || hist.is_some();
    let first = hist.as_ref().map_or(0, |h| h.start);
    let last = if columns {
        hist.as_ref().map_or(cols, |h| h.stop)
    } else {
        rows
    };
    if first == last {
        return Ok(());
    }
    let per_item = product(&[
        if columns { rows } else { cols } as u64,
        dtype_bytes(&dtype),
        if columns && !fortran { 2 } else { 1 },
    ])?
    .max(1);
    let budget = staging_budget(&cp)?;
    if per_item > budget {
        return Err(PyMemoryError::new_err(
            "one dense streaming row/column exceeds the device-memory budget",
        ));
    }
    // Two reusable slots avoid the additional pinned-allocation overhead of
    // four slots for these short reductions, while overlapping host staging.
    let slots = if budget / per_item >= 2 && last - first > 1 {
        2
    } else {
        1
    };
    let sub = (if requested <= 0 {
        4096
    } else {
        requested as usize
    })
    .min((budget / slots as u64 / per_item) as usize)
    .min(last - first)
    .max(1);
    let mut caller = BatchStreams::caller(&cp)?;
    for output in [sums, counts, squares].into_iter().flatten() {
        caller.retain(0, output.clone());
    }
    if let Some(histogram) = &hist {
        caller.retain(0, histogram.out.clone());
    }
    caller.retain(0, codes.clone());
    if let Some(mask) = mask {
        caller.retain(0, mask.clone());
    }
    if fortran && hist.is_none() {
        for a in [sums, counts, squares].into_iter().flatten() {
            a.call_method1("fill", (0,))?;
        }
    }
    if let Some(h) = &hist {
        for a in [sums, counts].into_iter().flatten() {
            a.get_item((all(py), slice(py, h.start, h.stop)))?
                .call_method1("fill", (0,))?;
        }
    }
    let mut streams = BatchStreams::new(&cp, slots)?;
    let mut start = first;
    let mut batch_i = 0;
    while start < last {
        let stop = (start + sub).min(last);
        let slot = batch_i % slots;
        let _scope = streams.enter(slot)?;
        let stream = stream(&cp)?;
        let hx = if columns {
            window(&X, start, stop)?
        } else {
            X.get_item((slice(py, start, stop), all(py)))?
        };
        let dx = streams.upload_dense_order(slot, 0, &hx, columns)?;
        let expected = if columns {
            [rows, stop - start]
        } else {
            [stop - start, cols]
        };
        let block = operands.dense_block(&cp, &dx, expected, device_dtype)?;
        operands.consume_dense(
            &cp,
            &block,
            if columns { 0 } else { start as u64 },
            if columns { start as u64 } else { 0 },
            cols as u64,
            hist.as_ref(),
            stream,
        )?;
        start = stop;
        batch_i += 1;
    }
    streams.finish()
}
