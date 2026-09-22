# decoupler-GPU: `dcg`

{mod}`decoupler.mt` contains different statistical methods to extract biological activities {cite}`Badia2022`.
{mod}`rapids_singlecell.dcg` accelerates some of these methods.

```{eval-rst}
.. module:: rapids_singlecell.dcg
.. currentmodule:: rapids_singlecell

.. autosummary::
    :toctree: generated

    dcg.mlm
    dcg.ulm
    dcg.aucell
    dcg.ora
    dcg.query_set
    dcg.waggr
    dcg.zscore
```

For a list of selected genes, use `query_set` directly:

```python
result = rsc.dcg.query_set(["TP53", "CDKN1A", "BAX"], net, tmin=3)
```

It returns a DataFrame of log odds ratios, Fisher p-values, and adjusted p-values.
Unlike the matrix-based `ora`, its default test is one-sided (`alternative="greater"`).

`ora` targets **decoupler 2.2.0**. Existing calls retain their arguments,
positional order, defaults, return values, and AnnData output keys when changing
`dc.mt.ora` to `rsc.dcg.ora`. This includes `bsize=250_000`, `n_up=None`, `n_bm=0`,
`n_bg=20_000`, and `ha_corr=0.5`. `query_set` likewise accepts decoupler's original
positional arguments; its optional named background is keyword-only.

ORA converts matrix data to float32 before ranking or applying a value cutoff,
as the other GPU methods do. Fisher calculations and outputs use float64.
Values that become equal in float32 are treated as ties.

```python
# Use the same arguments as the existing decoupler call.
scores, enrichment_padj = rsc.dcg.ora(data, net, n_up=500, n_bm=0)
```

For compatibility, ranked ORA reproduces the released 2.2.0 selection behavior:
genes are selected when their ascending, one-based rank is **greater than `n_up`
or less than `n_bm`**. Fractional thresholds are compared directly, and overlapping
selections are combined as a set. With `n_up=None`, the threshold is
`max(ceil(0.05 * n_genes), 2)`, so the default selects approximately **95%** of genes.
This is the known upstream ranking bug; the correction listed for 2.2.1 in the
[decoupler changelog](https://decoupler.readthedocs.io/en/latest/changelog.html)
is intentionally not applied here. The original CPU Fisher comparison of tied
probabilities is also preserved, including cases where it differs from SciPy.
Shared log-gamma factors use the same CPU math implementation; overlap counting
and probability lookups run on the GPU.

For differential-expression comparisons, the additional `value_cutoff` option
accepts a wide DataFrame with comparisons as rows and tested genes as columns.
It selects genes independently for each comparison:

```python
# padj has adjusted gene-level p-values for every gene and comparison.
scores, enrichment_padj = rsc.dcg.ora(
    padj,
    net,
    value_cutoff=0.05,
    value_direction="less",
    n_bg=None,
    alternative="greater",
)
```

For multiple DEG criteria, first construct binary labels on DataFrames with matching
rows and columns:

```python
labels = (padj < 0.05) & (log2fc.abs() >= 1.0)
scores, enrichment_padj = rsc.dcg.ora(
    labels, net, value_cutoff=0.5, n_bg=None, alternative="greater"
)
```

The default value direction is `"greater"`; comparisons are strict (`>` or `<`).
`value_direction` controls gene selection, while `alternative` controls the Fisher
test (`"two-sided"` by default, `"greater"` for enrichment, `"less"` for depletion).
Without `value_cutoff`, `ora` retains its ranked selection behavior.

The default background remains 20,000 in both modes. The examples use `n_bg=None`
to count **all input genes**, including non-DEGs and genes without term annotations.
Terms are restricted to input genes before
applying `tmin`. When using `value_cutoff`, all-zero rows and columns are retained
even with `empty=True`;
comparisons with no selected genes return p-values of one. Cutoffs are applied
to the float32 matrix. Include all tested genes, and exclude untested
genes before making labels. Missing or infinite input values are rejected.

This follows the comparison-by-gene layout in the
[decoupler bulk RNA-seq tutorial](https://decoupler.readthedocs.io/en/latest/notebooks/bulk/rna.html).
For its single `results_df`, a p-value input can be constructed with
`results_df[["padj"]].T.rename(index={"padj": "treatment.vs.control"})` after removing
genes without usable test results. Multiple comparisons sharing the same tested
universe can be concatenated as rows.

For each term, the Fisher table is:

| | In term | Outside term |
|---|---:|---:|
| DEG | DEGs in term | DEGs outside term |
| Non-DEG | Non-DEGs in term | Background genes in neither the DEG list nor the term |

The two returned DataFrames contain corrected log odds ratios and enrichment
p-values adjusted across retained terms within each comparison. This adjustment is
separate from the adjusted gene-level p-values used to select DEGs.

For an existing DEG list, supply all tested genes as an explicit background:

```python
result = rsc.dcg.query_set(deg_names, net, background=tested_gene_names)
```

The explicit background includes tested genes absent from the network and replaces
the numeric `n_bg` setting. All selected genes must belong to the background.
