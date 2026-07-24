---
name: rapids-singlecell
description: "Analyze single-cell and spatial data with RAPIDS-singlecell (rsc): GPU setup, scientific workflows, conservative annotation, and executed notebooks. Use for any RSC analysis or setup request."
---

# RAPIDS-singlecell

## Start with the analysis shape

- For every analysis or notebook request, read [`references/notebooks.md`](references/notebooks.md)
  before coding. Follow **question → inspect → preserve → compute → evaluate → interpret → export**.

## Keep GPU data flow explicit

For the ordinary single-GPU route, initialize RMM before importing CuPy or RSC:

```python
import rmm

oversubscribe = False
rmm.reinitialize(managed_memory=oversubscribe, pool_allocator=not oversubscribe)
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

cp.cuda.set_allocator(rmm_cupy_allocator)
import rapids_singlecell as rsc
```

Set `oversubscribe=True` only for managed memory. Before changing allocator mode,
read [`references/memory.md`](references/memory.md); for Dask, out-of-core, or
multi-GPU work, also read [`references/dask.md`](references/dask.md).

- Keep GPU preprocessing inputs on-device through the sequence. If a live call
  raises `_check_gpu_X` or says the input is not a CuPy matrix, inspect
  `rsc.get.anndata_to_GPU` and move the required data; `convert_all=True` includes
  layers. Treat this as transitional recovery guidance, not a fixed namespace rule.
- Move data back only at an intentional interop or export boundary with the live
  `rsc.get.X_to_CPU` or `rsc.get.anndata_to_CPU` signature. Leave representations
  and graphs that are already host-backed where they are.
- For spatial work, read [`references/spatial.md`](references/spatial.md). Squidpy
  may supply a physical graph when live RSC cannot; attribute the boundary.

## Make scientific decisions explicit

- Begin with the **biological question and unit of replication**. Base inference on
  biological samples; when replication is absent, keep conclusions descriptive.
- **Inspect the input and uncorrected data first.** Verify count location, labels,
  batches, and inferred design instead of treating metadata as fact.
- Respect and accommodate the user's requested analysis choices when compatible
  with the observed data and design. Push back with evidence when a request is
  inapplicable or weakens inference, explain why, and propose a valid alternative.
- Keep an uncorrected baseline. Apply batch correction only to an identifiable
  technical variable crossed with replicated biology; Harmony on single-batch data
  has no batch contrast to correct.
- Separate observations from interpretations. Report contradictions, plausible
  alternatives, uncertainty, and design limits.
- Annotate from coherent positive and exclusion programs across samples. Use broad,
  `unknown`, or `mixed` labels with confidence and contradictory evidence when the
  data do not support a narrower label.
- **Critically evaluate every output before trusting it.** Compare group counts and
  sizes, figure pixels, render cost, and file size with data scale. Hundreds of
  megapixels or near-cell-count groups are stop signals: adjust and rerun.

## Discover and compose live APIs

- After preflight, use `python -m rapids_singlecell_skills.api search "<intent>"`
  and `describe <symbol>` against the live package. If the helper is unavailable,
  use the fallback in [`references/setup.md`](references/setup.md). A miss is not
  proof of absence.
- RSC's public API includes neighboring GPU ports: `rsc.gr` for supported
  Squidpy-compatible spatial methods, `rsc.ptg` for Pertpy-compatible perturbation
  methods, and `rsc.dcg` for Decoupler-compatible cell-level scoring. Search both
  ecosystem and method names, then describe the returned public symbol.
- Use public RSC APIs for supported computation. When a required capability is
  absent, state the gap and either stop for a decision or compose an explicitly
  requested or scientifically essential ecosystem method under its own skill with
  a clearly attributed boundary; do not silently substitute CPU computation.
- For pathway/activity work, obtain resources and pathway-native plots with
  canonical `decoupler`. Default supported per-cell scoring to a live `rsc.dcg`
  method and summarize relevant groups descriptively. Use canonical `decoupler`
  pseudobulk when requested or for replicated cross-condition inference, preserving
  biological sample as the replication unit. Attribute each boundary.
- Use `sc.pl` or `sq.pl` only for compatible embedding, spatial, or other displays
  that canonical `decoupler` plotting does not express. Make the expression source
  explicit and disable options that would compute dendrograms, layouts, graphs,
  smoothing, or transformations.

## Verify and finish

- Before Jupyter, run `rapids-singlecell-check-kernel`. For a new or broken runtime,
  installation/version checks, helper fallback, or preflight failure, read
  [`references/setup.md`](references/setup.md) and stop until the GPU check passes.
- Preserve immutable input provenance and source counts. Execute from a fresh kernel,
  resolve unexpected warnings, suppress only understood routine noise, and inspect
  every rendered output.
- Return the executed `.ipynb`, a continuation-ready AnnData artifact, compact audit
  tables, and a concise `.md` report of findings, evidence, limitations, and open
  questions.
