//! Bounded stable radix sorting with reusable allocator-owned scratch.
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

#[derive(Clone, Copy)]
struct Region {
    offset: usize,
    pointer: u64,
    bytes: usize,
}
impl Region {
    fn reserve(total: &mut usize, elements: u64, item_size: usize) -> PyResult<Self> {
        let overflow = || PyValueError::new_err("radix scratch exceeds addressable memory");
        let offset = total.checked_add(255).ok_or_else(overflow)? & !255;
        let bytes = usize::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(item_size))
            .ok_or_else(overflow)?;
        *total = offset
            .checked_add(bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(overflow)?;
        Ok(Self {
            offset,
            pointer: 0,
            bytes,
        })
    }
    fn bind(mut self, base: u64) -> Self {
        self.pointer = base + self.offset as u64;
        self
    }
}

#[derive(Clone, Copy)]
struct Segments {
    begins: u64,
    ends: u64,
    tile_offsets: u64,
    count: u64,
    tiles: u64,
}

/// One CuPy/RMM allocation owns all scratch for one stream. Checked, 256-byte
/// offsets partition the arena; returned views retain its Python allocation.
/// Results remain valid until this workspace's next sort on the same stream.
pub struct Workspace<'py> {
    storage: Bound<'py, PyAny>,
    storage_array: Array,
    dtype: Dtype,
    index_dtype: Dtype,
    va: [Region; 2],
    oa: Option<[Region; 2]>,
    ha: Region,
    la: Region,
    max_rows: u64,
    max_cols: u64,
    masks: Option<Region>,
    output_index: usize,
    trivial_values: Option<Bound<'py, PyAny>>,
}
impl<'py> Workspace<'py> {
    pub fn new(
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
        dtype: Dtype,
        indices: bool,
    ) -> PyResult<Self> {
        Self::with_tiles(
            cp,
            rows,
            cols,
            dtype,
            indices.then_some(Dtype::I64),
            if rows <= 1024 { 0 } else { rows.div_ceil(1024) },
        )
    }
    /// OVR uses the original 32-bit within-column position payload, independent
    /// of the source matrix's row-index width and float/encoded-key storage.
    pub fn with_compact_indices(
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
        dtype: Dtype,
    ) -> PyResult<Self> {
        Self::with_tiles(
            cp,
            rows,
            cols,
            dtype,
            Some(Dtype::U32),
            if rows <= 1024 { 0 } else { rows.div_ceil(1024) },
        )
    }
    fn with_tiles(
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
        dtype: Dtype,
        index_dtype: Option<Dtype>,
        tiles: usize,
    ) -> PyResult<Self> {
        Self::with_tiles_allocator(cp, rows, cols, dtype, index_dtype, tiles, |bytes| {
            empty(cp, &[bytes], "uint8", "C", false)
        })
    }

    /// Supply one arena allocation, for example from a stream ring's retained
    /// caller pool. All offsets, capacity and alignment checks remain shared.
    pub fn with_allocator(
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
        dtype: Dtype,
        index_dtype: Option<Dtype>,
        allocate: impl FnOnce(usize) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        Self::with_tiles_allocator(
            cp,
            rows,
            cols,
            dtype,
            index_dtype,
            if rows <= 1024 { 0 } else { rows.div_ceil(1024) },
            allocate,
        )
    }

    fn with_tiles_allocator(
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
        dtype: Dtype,
        index_dtype: Option<Dtype>,
        tiles: usize,
        allocate: impl FnOnce(usize) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if !matches!(dtype, Dtype::F32 | Dtype::U32) {
            return Err(PyTypeError::new_err(
                "sort input must have dtype float32 or uint32",
            ));
        }
        if index_dtype.is_some_and(|dtype| !matches!(dtype, Dtype::I64 | Dtype::U32)) {
            return Err(PyTypeError::new_err(
                "sort positions must have dtype int64 or uint32",
            ));
        }
        if rows > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "radix sort supports at most 2^32-1 rows per column",
            ));
        }
        let size = product(&[rows as u64, cols as u64])?;
        let mut bytes = 0;
        let va = [
            Region::reserve(&mut bytes, size, 4)?,
            Region::reserve(&mut bytes, size, 4)?,
        ];
        let oa = if let Some(index_dtype) = index_dtype {
            Some([
                Region::reserve(&mut bytes, size, index_dtype.size() as usize)?,
                Region::reserve(&mut bytes, size, index_dtype.size() as usize)?,
            ])
        } else {
            None
        };
        // Ordinary small-column sorts need no global histogram/permutation.
        // Explicit segmented workspaces pass their real tile count here.
        let ha = Region::reserve(&mut bytes, product(&[cols as u64, tiles as u64, 256])?, 4)?;
        let la = Region::reserve(
            &mut bytes,
            if tiles == 0 {
                0
            } else if index_dtype.is_some() {
                size
            } else {
                product(&[cols as u64, tiles as u64, 256])?
            },
            4,
        )?;
        let masks = if rows >= 32768 && dtype == Dtype::F32 {
            Some(Region::reserve(&mut bytes, cols as u64, 4)?)
        } else {
            None
        };
        let storage = allocate(bytes)?;
        let allocation = read(
            &storage,
            cp,
            "radix scratch arena",
            Some(Dtype::U8),
            Layout::C,
        )?;
        capacity(&allocation, bytes as u64, "radix scratch arena")?;
        if !allocation.pointer.is_multiple_of(8) {
            return Err(PyValueError::new_err(
                "radix scratch allocator must return eight-byte aligned memory",
            ));
        }
        let base = allocation.pointer;
        Ok(Self {
            storage,
            storage_array: allocation,
            dtype,
            index_dtype: index_dtype.unwrap_or(Dtype::I64),
            va: va.map(|r| r.bind(base)),
            oa: oa.map(|a| a.map(|r| r.bind(base))),
            ha: ha.bind(base),
            la: la.bind(base),
            masks: masks.map(|r| r.bind(base)),
            max_rows: rows as u64,
            max_cols: cols as u64,
            output_index: 1,
            trivial_values: None,
        })
    }
    fn view(
        &self,
        cp: &Bound<'py, PyModule>,
        region: Region,
        dtype: Dtype,
        dims: &[usize],
    ) -> PyResult<Bound<'py, PyAny>> {
        let count = product(&dims.iter().map(|&d| d as u64).collect::<Vec<_>>())?;
        let bytes = count
            .checked_mul(dtype.size())
            .ok_or_else(|| PyValueError::new_err("radix result size overflow"))?;
        if bytes > region.bytes as u64 {
            return Err(PyValueError::new_err(
                "radix result exceeds scratch capacity",
            ));
        }
        let kw = PyDict::new(cp.py());
        kw.set_item("order", "F")?;
        self.storage
            .get_item(slice(
                cp.py(),
                region.offset,
                region.offset + bytes as usize,
            ))?
            .call_method1("view", (dtype.name(),))?
            .call_method("reshape", (dims.to_vec(),), Some(&kw))
    }

    /// View the sorted value buffer after `sort`; its lifetime is tied to this
    /// workspace exactly like the returned indices.
    pub fn values(
        &self,
        cp: &Bound<'py, PyModule>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(input) = &self.trivial_values {
            return Ok(input.clone());
        }
        self.view(cp, self.va[self.output_index], self.dtype, &[rows, cols])
    }

    fn passes(
        &self,
        cp: &Bound<'py, PyModule>,
        x: &Array,
        rows: u64,
        cols: u64,
        s: usize,
    ) -> PyResult<Vec<u32>> {
        let mask = if rows >= 32768
            && let Some(masks) = self.masks
        {
            checked_launch(
                x.device,
                "rank_sort_byte_mask",
                cols * 256,
                s,
                &mut [
                    Arg::P(x.pointer),
                    Arg::P(masks.pointer),
                    Arg::N(rows),
                    Arg::N(cols),
                ],
            )?;
            let active = self.view(cp, masks, Dtype::U32, &[cols as usize])?;
            let flags = cp
                .call_method1("asnumpy", (&active,))?
                .call_method0("tolist")?
                .extract::<Vec<u32>>()?;
            flags.into_iter().fold(0, |a, b| a | b)
        } else {
            15
        };
        let passes: Vec<_> = (0u32..4).filter(|byte| mask & (1 << byte) != 0).collect();
        Ok(passes)
    }

    /// Keys-only passes reuse A for locally ordered tiles and B for the
    /// globally ordered result. Per-digit prefixes replace the full per-value
    /// permutation, keeping scatter reads contiguous and reducing scratch.
    fn sort_keys(
        &mut self,
        cp: &Bound<'py, PyModule>,
        x: &Array,
        rows: u64,
        cols: u64,
        stream: usize,
        segments: Option<Segments>,
    ) -> PyResult<()> {
        // The bounded block tier avoids all global histogram/scatter launches
        // for medium reference columns. Narrow dispatch follows measured wins
        // for both continuous and tied values; short final windows stay radix.
        if segments.is_none()
            && self.dtype == Dtype::F32
            && (5000..=8192).contains(&rows)
            && (32..=65_535).contains(&cols)
        {
            checked_launch(
                x.device,
                "rank_sort_packed",
                cols * 256,
                stream,
                &mut [
                    Arg::P(x.pointer),
                    Arg::P(self.va[1].pointer),
                    Arg::N(rows),
                    Arg::N(cols),
                    Arg::U(0),
                ],
            )?;
            self.output_index = 1;
            return Ok(());
        }
        let tiles = segments.map_or(rows.div_ceil(1024), |s| s.tiles);
        let passes = self.passes(cp, x, rows, cols, stream)?;
        for (pass, byte) in passes.into_iter().enumerate() {
            let source = if pass == 0 {
                x.pointer
            } else {
                self.va[1].pointer
            };
            let shift = (byte * 8) | if self.dtype == Dtype::U32 { 256 } else { 0 };
            let mut dimensions = Vec::with_capacity(8);
            if let Some(s) = segments {
                dimensions.extend([Arg::P(s.begins), Arg::P(s.ends), Arg::P(s.tile_offsets)]);
            }
            dimensions.extend([Arg::N(rows), Arg::N(cols)]);
            if let Some(s) = segments {
                dimensions.push(Arg::N(s.count));
            }
            dimensions.extend([Arg::N(tiles), Arg::U(shift)]);
            let mut histogram = vec![
                Arg::P(source),
                Arg::P(self.va[0].pointer),
                Arg::P(self.ha.pointer),
                Arg::P(self.la.pointer),
            ];
            histogram.extend(dimensions.iter().copied());
            checked_launch(
                x.device,
                if segments.is_some() {
                    "rank_radix_segments_hist_values"
                } else {
                    "rank_radix_hist_values"
                },
                cols * tiles * 256,
                stream,
                &mut histogram,
            )?;
            if let Some(s) = segments {
                checked_launch(
                    x.device,
                    "rank_radix_segments_prefix",
                    cols * s.count * 256,
                    stream,
                    &mut [
                        Arg::P(self.ha.pointer),
                        Arg::P(s.tile_offsets),
                        Arg::N(cols),
                        Arg::N(s.count),
                        Arg::N(tiles),
                    ],
                )?;
            } else {
                checked_launch(
                    x.device,
                    "rank_radix_prefix",
                    cols * 256,
                    stream,
                    &mut [Arg::P(self.ha.pointer), Arg::N(cols), Arg::N(tiles)],
                )?;
            }
            let mut scatter = vec![
                Arg::P(self.va[0].pointer),
                Arg::P(self.va[1].pointer),
                Arg::P(self.ha.pointer),
                Arg::P(self.la.pointer),
            ];
            scatter.extend(dimensions);
            checked_launch(
                x.device,
                if segments.is_some() {
                    "rank_radix_segments_scatter_values"
                } else {
                    "rank_radix_scatter_values"
                },
                cols * tiles * 256,
                stream,
                &mut scatter,
            )?;
        }
        self.output_index = 1;
        Ok(())
    }

    pub fn sort(
        &mut self,
        cp: &Bound<'py, PyModule>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = read(input, cp, "sort input", Some(self.dtype), Layout::F)?;
        let (rows, cols) = matrix(&x, "sort input")?;
        if rows > self.max_rows || cols > self.max_cols {
            return Err(PyValueError::new_err(
                "sort input exceeds its workspace capacity",
            ));
        }
        current_device(cp, &[&x, &self.storage_array])?;
        self.trivial_values = None;
        self.output_index = 1;
        if rows <= 1 || cols == 0 {
            self.trivial_values = Some(input.clone());
            return if let Some(order) = self.oa {
                let result = self.view(
                    cp,
                    order[1],
                    self.index_dtype,
                    &[rows as usize, cols as usize],
                )?;
                result.call_method1(pyo3::intern!(cp.py(), "fill"), (0,))?;
                Ok(result)
            } else {
                Ok(input.clone())
            };
        }
        let s = stream(cp)?;
        let tiles = rows.div_ceil(1024);
        let raw_keys = (if x.dtype == Dtype::U32 { 256 } else { 0 })
            | (if self.index_dtype == Dtype::U32 {
                512
            } else {
                0
            });
        if rows <= 1024 {
            checked_launch(
                x.device,
                "rank_sort_small",
                cols * 256,
                s,
                &mut [
                    Arg::P(x.pointer),
                    Arg::P(self.va[1].pointer),
                    Arg::P(self.oa.as_ref().map_or(0, |a| a[1].pointer)),
                    Arg::N(rows),
                    Arg::N(cols),
                    Arg::U(raw_keys),
                ],
            )?;
        } else if self.oa.is_none() {
            self.sort_keys(cp, &x, rows, cols, s, None)?;
        } else {
            let passes = self.passes(cp, &x, rows, cols, s)?;
            self.output_index = (passes.len() - 1) % 2;
            for (pass, byte) in passes.into_iter().enumerate() {
                let dst = pass % 2;
                let src = if pass == 0 {
                    x.pointer
                } else {
                    self.va[1 - dst].pointer
                };
                let output = &self.va[dst];
                let shift = (byte * 8) | raw_keys;
                checked_launch(
                    x.device,
                    "rank_radix_hist",
                    cols * tiles * 256,
                    s,
                    &mut [
                        Arg::P(src),
                        Arg::P(self.ha.pointer),
                        Arg::P(self.la.pointer),
                        Arg::N(rows),
                        Arg::N(cols),
                        Arg::N(tiles),
                        Arg::U(shift),
                    ],
                )?;
                checked_launch(
                    x.device,
                    "rank_radix_prefix",
                    cols * 256,
                    s,
                    &mut [Arg::P(self.ha.pointer), Arg::N(cols), Arg::N(tiles)],
                )?;
                checked_launch(
                    x.device,
                    "rank_radix_scatter",
                    cols * tiles * 256,
                    s,
                    &mut [
                        Arg::P(src),
                        Arg::P(
                            self.oa
                                .as_ref()
                                .filter(|_| pass != 0)
                                .map_or(0, |a| a[1 - dst].pointer),
                        ),
                        Arg::P(output.pointer),
                        Arg::P(self.oa.as_ref().map_or(0, |a| a[dst].pointer)),
                        Arg::P(self.ha.pointer),
                        Arg::P(self.la.pointer),
                        Arg::N(rows),
                        Arg::N(cols),
                        Arg::N(tiles),
                        Arg::U(shift),
                    ],
                )?;
            }
        }
        if let Some(order) = &self.oa {
            self.view(
                cp,
                order[self.output_index],
                self.index_dtype,
                &[rows as usize, cols as usize],
            )
        } else {
            self.values(cp, rows as usize, cols as usize)
        }
    }
}

/// The caller has validated every allocation, launch extent, stream and device.
/// Keeping Python validation outside radix passes avoids twelve repeated device
/// queries and allocation introspections for each column batch.
pub(crate) fn checked_launch(
    device: usize,
    name: &str,
    work: u64,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    checked_launch_threads(device, name, work, 256, stream, args)
}

pub(crate) fn checked_launch_threads(
    device: usize,
    name: &str,
    work: u64,
    threads: u32,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    checked_launch_shared(device, name, work, threads, 0, stream, args)
}

pub(crate) fn checked_launch_shared(
    device: usize,
    name: &str,
    work: u64,
    threads: u32,
    shared_bytes: u32,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    if work == 0 {
        return Ok(());
    }
    let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
    // SAFETY: only internal, validated workspace and ranking callers use this.
    unsafe {
        runtime::launch_shared(
            device,
            name,
            (work.div_ceil(threads as u64).min(65535) as u32, 1, 1),
            (threads, 1, 1),
            shared_bytes,
            stream,
            &mut pointers,
        )
    }
}

/// Reusable radix workspace for selected, non-overlapping row segments. OVO
/// sorts all huge groups together; medium/large groups remain untouched.
pub struct Segmented<'py> {
    scratch: Workspace<'py>,
    _begins: Bound<'py, PyAny>,
    _ends: Bound<'py, PyAny>,
    _tile_offsets: Bound<'py, PyAny>,
    begins: Array,
    ends: Array,
    tile_offsets: Array,
    segments: u64,
    tiles: u64,
}
impl<'py> Segmented<'py> {
    pub fn new(
        cp: &Bound<'py, PyModule>,
        offsets: &[usize],
        active: &[usize],
        cols: usize,
    ) -> PyResult<Self> {
        let rows = offsets.last().copied().unwrap_or(0);
        let mut cumulative = vec![0u64];
        let mut begins = Vec::with_capacity(active.len());
        let mut ends = Vec::with_capacity(active.len());
        for &g in active {
            begins.push(offsets[g] as u64);
            ends.push(offsets[g + 1] as u64);
            cumulative.push(
                cumulative.last().copied().unwrap()
                    + (offsets[g + 1] - offsets[g]).div_ceil(1024) as u64,
            );
        }
        let tiles = *cumulative.last().unwrap();
        let scratch = Workspace::with_tiles(cp, rows, cols, Dtype::F32, None, tiles as usize)?;
        let kw = PyDict::new(cp.py());
        kw.set_item("dtype", "uint64")?;
        let mut uploads = Vec::with_capacity(3);
        let metadata: PyResult<_> = (|| {
            let b = cp.getattr("asarray")?.call((begins,), Some(&kw))?;
            uploads.push(b.clone());
            let e = cp.getattr("asarray")?.call((ends,), Some(&kw))?;
            uploads.push(e.clone());
            let t = cp.getattr("asarray")?.call((cumulative,), Some(&kw))?;
            uploads.push(t.clone());
            let begins = vector(&b, cp, "segment begins", Some(Dtype::U64))?;
            let ends = vector(&e, cp, "segment ends", Some(Dtype::U64))?;
            let tile_offsets = vector(&t, cp, "segment tile offsets", Some(Dtype::U64))?;
            Ok((b, e, t, begins, ends, tile_offsets))
        })();
        if metadata.is_err() {
            // Successful uploads must outlive a later Python allocation or
            // metadata failure. The normal path transfers every owner to Self
            // and keeps the original asynchronous constructor behavior.
            sync(cp)?;
        }
        let (b, e, t, begins, ends, tile_offsets) = metadata?;
        Ok(Self {
            scratch,
            _begins: b,
            _ends: e,
            _tile_offsets: t,
            begins,
            ends,
            tile_offsets,
            segments: active.len() as u64,
            tiles,
        })
    }
    pub fn sort(
        &mut self,
        cp: &Bound<'py, PyModule>,
        input: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = read(
            input,
            cp,
            "segmented sort input",
            Some(Dtype::F32),
            Layout::F,
        )?;
        let (rows, cols) = matrix(&x, "segmented sort input")?;
        if rows != self.scratch.max_rows || cols > self.scratch.max_cols {
            return Err(PyValueError::new_err(
                "segmented sort input exceeds its workspace",
            ));
        }
        current_device(cp, &[&x, &self.scratch.storage_array])?;
        if cols == 0 || self.segments == 0 {
            return Ok(input.clone());
        }
        let s = stream(cp)?;
        self.scratch.sort_keys(
            cp,
            &x,
            rows,
            cols,
            s,
            Some(Segments {
                begins: self.begins.pointer,
                ends: self.ends.pointer,
                tile_offsets: self.tile_offsets.pointer,
                count: self.segments,
                tiles: self.tiles,
            }),
        )?;
        self.scratch.values(cp, rows as usize, cols as usize)
    }
}
