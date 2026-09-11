# Analysis notebooks

Read this file for every analysis or notebook request.

## Follow this scaffold

This is a notebook shape, not an API catalog. Resolve calls against the live package and give each numbered item its own Markdown or code cell.

1. **Markdown — Question and design:** state the biological question, unit of replication, relevant metadata, intended inference, and known limits.
2. **Code — Runtime and provenance:** configure RMM as shown in the core skill, exactly once in this first cell — reinitializing later in the same kernel, while allocations are live, is undefined behavior. Record the seed, input provenance, source revision, and `session_info2`.
3. **Code — Load and inspect:** check shape, sparsity, count location, labels, batches, coordinates, and missing values before choosing methods.
4. **Code — Preserve and place data:** keep immutable input/counts, create the working object, choose the capacity route, and establish required GPU residency.
5. **Code — QC and filter:** plot the relevant distributions with `sc.pl`, justify thresholds, and summarize what was retained; keep the interpretation in the next Markdown cell.
6. **Code — Preprocess:** with `rsc.pp`, select HVGs from counts and normalize/log-transform the full object for a standard workflow. Subset to HVGs only when a later step needs it; otherwise pass `mask_var="highly_variable"` where the call supports it.
7. **Code — Structure:** compute `rsc.pp` PCA and neighbors in float32. Run `rsc.tl` Leiden with `dtype="float64"` by default; for reproducible reruns also hold the graph, observation order, `random_state`, parameters, and package versions fixed.
8. **Code + Markdown — Annotate:** show positive, exclusion, and contradictory marker evidence, tentative labels, confidence, and checkable sources.
9. **Conditional spatial branch:** read [`spatial.md`](spatial.md), build an attributed physical graph, run justified spatial/niche analyses, and inspect granularity.
10. **Code + Markdown — Render and evaluate:** make bounded plots of computed results, sanity-check the consequential outputs against data scale, then interpret them.
11. **Code — Export:** return intended data to host and save AnnData, audit tables, the executed notebook, and the findings report.

## Make the notebook iterable

- Organize sections as question → computation → plot or table → interpretation.
- Author the analysis in the notebook itself. If helper scripts exist for batch execution, still write the narrative cells by hand; never assemble a notebook by pasting self-contained scripts into cells, which duplicates each script's imports and runtime setup inside one shared kernel.
- Put one function definition or one analysis task in each code cell, and keep a cell under roughly 25 lines. A stage that needs 90 lines is several tasks, not one. Keep setup, provenance, scientific computation, summarization, plotting, and export separate.
- Put the plot in the next visible cell after preparing the result it renders. Keep its interpretation immediately after it.
- Keep infrastructure compact or hidden, but leave consequential scientific calls visible. Never hide an entire workflow in one omnibus cell.

## Keep the analysis human

- Prefer the owning package's native plot for already-computed results: canonical `decoupler` for pathways, `pt.pl` for Pertpy results, and `sc.pl` or `sq.pl` for compatible AnnData or spatial results. Use custom plotting only when no standard ecosystem plot expresses the result.
- Resolve unexpected warnings and suppress only understood routine progress or information logging. Deliver concise outputs without errors or raw log streams.
- Keep preflight, assertion walls, recursive conversion helpers, and package-test scaffolding outside the narrative. Turn useful diagnostics into a compact table or a clearly named validation cell.
- Use `session_info2` for the environment record; it already is the package-version dump, so do not hand-build a second one. Record the seed, input provenance, and source revision alongside it.
- Rebuild tentative labels from current marker evidence after partition changes; never reuse cluster-ID maps. Require complete coverage, assign `unknown` for weak evidence, and show positive, exclusion, contradictory evidence, confidence, and checkable source links.
- Present the work as a scientific analysis, not an RSC-versus-Scanpy benchmark, unless benchmarking was requested.
- Preserve explicit analysis preferences across revisions; verify API-bearing choices in live RSC and keep them visible. Persist one as a standing default only on explicit request, with scope and conditions.
- If a requested choice conflicts with the data or design, show evidence, explain the concern, propose a valid alternative, and never silently ignore or substitute it.

## Preserve scverse data flow

- For a standard log-normalized workflow, preserve source counts in a layer, select HVGs from counts, normalize and log-transform the full object, then subset one AnnData copy to HVGs if a later step requires it.
- Before a GPU call consumes a layer, apply the core residency check: moving `X` does not imply that every layer moved.
- Use `rsc.get.X_to_CPU` or `rsc.get.anndata_to_CPU` at an intentional interop boundary; do not invent host-conversion helpers or transfer representations or graphs that are already host-backed.
- Use a non-RSC computation only when requested or essential to the stated question, under its relevant skill and with explicit attribution. State the gap rather than hiding an external computation as an RSC fallback.
- For pathways, obtain resources and pathway-native plots with canonical `decoupler`; default supported per-cell scoring to live `rsc.dcg` and summarize reporting groups. Keep a single sample's group means and cell spread descriptive. Use canonical `decoupler` pseudobulk when requested or for replicated cross-condition inference, aggregating source counts by biological sample and relevant group with `rsc.get.aggregate` before handing off. Load the decoupler skill and attribute each boundary.

## Finish cleanly

- Execute every cell from a fresh kernel and inspect the rendered notebook.
- Require every code cell to complete without errors; resolve unexpected warnings and remove only understood routine logs.
- Save the executed `.ipynb`, a continuation-ready AnnData artifact, and compact audit tables. After reviewing them, write a concise `.md` report of findings, evidence, limitations, and unresolved questions.
