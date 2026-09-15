---
name: rapids-singlecell
description: "GPU single-cell and spatial analysis with RAPIDS-singlecell (rsc) — QC, filtering, normalization, HVG selection, PCA, neighbors, clustering, UMAP, batch integration, differential expression, cell-type annotation, pathway activity scoring, perturbation and CRISPR screens, spatial graphs and niches, and out-of-core Dask runs. Use for any of these, or to set up the runtime: RSC's GPU residency rules, memory routes and API coverage are not guessable from function names."
---

# RAPIDS-singlecell

## Start with the analysis shape

- For every analysis or notebook request, read [`references/notebooks.md`](references/notebooks.md) before coding. Follow **question → inspect → preserve → compute → evaluate → interpret → export**.

## Keep GPU data flow explicit

Importing RSC already installs an RMM-backed CuPy allocator, so explicit configuration is optional. To select a route deliberately, configure RMM before importing CuPy or RSC:

```python
# Cell 1 — configure RMM exactly once per kernel; never reinitialize again below.
import rmm

oversubscribe = False
rmm.reinitialize(managed_memory=oversubscribe, pool_allocator=not oversubscribe)
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

cp.cuda.set_allocator(rmm_cupy_allocator)
import rapids_singlecell as rsc
import scanpy as sc

SEED = 0
```

Adapt this shape, one stage per cell, resolving every call against the live package:

```python
adata = sc.read_h5ad(path)                 # inspect shape/sparsity/counts before choosing methods
adata.layers["counts"] = adata.X.copy()    # preserve immutable source counts
rsc.get.anndata_to_GPU(adata, convert_all=True)      # layers too, or pp raises _check_gpu_X
rsc.pp.calculate_qc_metrics(adata, ...)    # plot distributions with sc.pl, justify thresholds
rsc.pp.filter_cells(adata, ...); rsc.pp.filter_genes(adata, ...)
rsc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3", n_top_genes=2000)
rsc.pp.normalize_total(adata); rsc.pp.log1p(adata)   # normalize the full object
rsc.pp.pca(adata, mask_var="highly_variable")        # float32
rsc.pp.neighbors(adata)
rsc.tl.leiden(adata, dtype="float64", random_state=SEED)   # float64 keeps the partition stable
rsc.tl.umap(adata, random_state=SEED)
rsc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")   # annotate from evidence
rsc.get.anndata_to_CPU(adata)              # one intentional interop boundary
adata.write(out_path)
```

Set `oversubscribe=True` only for managed memory. Before changing allocator mode, read [`references/memory.md`](references/memory.md); for Dask, out-of-core, or multi-GPU work, also read [`references/dask.md`](references/dask.md).

- Keep GPU preprocessing inputs on-device through the sequence. If a live call raises `_check_gpu_X` or says the input is not a CuPy matrix, inspect `rsc.get.anndata_to_GPU` and move the required data; `convert_all=True` includes layers. Treat this as transitional recovery guidance, not a fixed namespace rule.
- Move data back only at an intentional interop or export boundary with the live `rsc.get.X_to_CPU` or `rsc.get.anndata_to_CPU` signature. Leave already host-backed representations and graphs alone.
- For spatial work, read [`references/spatial.md`](references/spatial.md). Squidpy may supply a physical graph when live RSC cannot; attribute the boundary.

## Default every computation to RSC

RSC is the analysis namespace, not an occasional accelerator: `rsc.pp`, `rsc.tl`, `rsc.get`, `rsc.gr`, `rsc.ptg`, `rsc.dcg`.

- **Print the whole surface before writing code**, so coverage is never a guess: `python -m rapids_singlecell_skills.api map` lists every public symbol with a one-line summary in a single pass; add `--options` for the accepted values of every enumerated argument, or `--namespace get` to narrow. That costs far less than discovering the API a call at a time — do it first, not after a `scanpy` call has already failed.
- To check one specific name, search it: `python -m rapids_singlecell_skills.api search "sc.pp.scale"` returns `rsc.pp.scale`. Prefer the RSC symbol whenever it computes the same result. If the helper is unavailable, use the fallback in [`references/setup.md`](references/setup.md); a miss is not proof of absence.
- Treat a lexical hit as a candidate, not an equivalence; confirm it with `describe`. `sq.gr.spatial_neighbors` returns `rsc.gr.calculate_niche` and `rsc.pp.neighbors`, yet RSC builds no physical spatial graph — never pass an expression kNN graph off as one.
- Only plotting, IO, and named gaps belong outside RSC. When a capability is genuinely absent, state the gap and either stop for a decision or compose the ecosystem method under its own skill at an attributed boundary; never silently substitute CPU computation.
- Before returning the notebook, list every remaining `sc.`/`sq.` computation and justify it. An unjustified one is a defect.
- RSC's public API includes neighboring GPU ports: `rsc.gr` for supported Squidpy-compatible spatial methods, `rsc.ptg` for Pertpy-compatible perturbation methods, and `rsc.dcg` for Decoupler-compatible cell-level scoring. Search both ecosystem and method names, then describe the returned public symbol.

## Hold these scientific invariants

These constrain the work rather than being its subject; keep them true without explaining them.

- Anchor claims to the **biological question and unit of replication**: infer from biological samples, stay descriptive without replication, and keep observation separate from interpretation with contradictions and design limits stated.
- **Inspect the input and uncorrected data first** — count location, labels, batches, design. Keep an uncorrected baseline and batch-correct only an identifiable technical variable crossed with replicated biology.
- Annotate from positive and exclusion programs; use broad, `unknown`, or `mixed` labels with confidence and contradictory evidence when the data do not support a narrower one.
- Respect and accommodate the user's requested analysis choices when compatible with the observed data and design. Push back with evidence when a request is inapplicable or weakens inference, explain why, and propose a valid alternative.
- **Critically evaluate every output before trusting it.** Compare group counts and sizes, figure pixels, render cost, and file size with data scale. Hundreds of megapixels or near-cell-count groups are stop signals: adjust and rerun. Keep the check cheap: read the summary numbers, not every value.

## Cross package boundaries deliberately

Every ecosystem boundary is also a device boundary. Cross it explicitly; when unsure where an array lives, inspect `type(adata.X)` rather than defensively re-converting.

| Crossing | What it needs |
|---|---|
| `sq.gr.spatial_neighbors` → RSC | builds from host coordinates; move the graph on-device |
| `decoupler.op` resources → `rsc.dcg` | host objects; scoring runs on-device |
| RSC → `sc.pl` / `sq.pl` / `pt.pl` | `rsc.get.anndata_to_CPU` first; never hand-roll a converter |

- For perturbation or CRISPR-screen work, read [`references/perturbation.md`](references/perturbation.md). `rsc.ptg` ports four pertpy APIs; the rest of pertpy stays an attributed boundary.
- For pathway/activity work, obtain resources and pathway-native plots with canonical `decoupler`. Default supported per-cell scoring to a live `rsc.dcg` method and summarize relevant groups descriptively. Aggregate with `rsc.get.aggregate`, then use canonical `decoupler` pseudobulk when requested or for replicated cross-condition inference, preserving biological sample as the replication unit. Attribute each boundary.
- RSC ships no plotting module: plot with the owning package's scverse API — `sc.pl`, `sq.pl`, or `pt.pl` — not hand-rolled `matplotlib`. Use `sc.pl` or `sq.pl` only for embedding, spatial, or other displays that canonical `decoupler` plotting does not express. Make the expression source explicit and disable options that would compute dendrograms, layouts, graphs, smoothing, or transformations.

## Verify and finish

- Before Jupyter, run `rapids-singlecell-check-kernel`. For a new or broken runtime, installation/version checks, helper fallback, or preflight failure, read [`references/setup.md`](references/setup.md) and stop until the GPU check passes.
- Preserve immutable input provenance and source counts; execute from a fresh kernel and inspect every rendered output.
- Return the executed `.ipynb`, a continuation-ready AnnData artifact, compact audit tables, and a concise `.md` report of findings, evidence, limitations, and open questions.
