# Comparing conditions

- The replication unit is the biological sample (donor, animal, culture), never the cell.
- Find the columns holding the biological sample and the condition, whatever they are called, and cross-tabulate them with `pd.crosstab` before any test.
  If no column identifies biological samples, ask the user rather than treating batches or cells as replicates.
- The code below calls these columns `sample` and `condition`, and the annotation column `cell_type`.
- With fewer than two samples per condition, describe differences and do not test them.
- Include pairing or batch covariates in the design (`~donor + condition`) when samples are paired or batches are crossed with condition.

## Pseudobulk DE

```python
pb = rsc.get.aggregate(
    adata, by=["sample", "cell_type", "condition"], func="sum", layer="counts"
)
pb.X = rsc.get.X_to_CPU(pb.layers.pop("sum"))
n_cells = adata.obs.value_counts(["sample", "cell_type"]).rename("n_cells")
pb.obs = pb.obs.join(n_cells, on=["sample", "cell_type"])
```

```python
import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

results = {}
for ct in pb.obs["cell_type"].unique():
    sub = pb[(pb.obs["cell_type"] == ct) & (pb.obs["n_cells"] >= 10)].copy()
    dc.pp.filter_by_expr(sub, group="condition")
    dds = DeseqDataSet(adata=sub, design="~condition", quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "treated", "control"], quiet=True)
    stats.summary()
    results[ct] = stats.results_df
```

- Sum raw counts per sample and cell type.
  Never test conditions with `rank_genes_groups` on cells, which treats cells as replicates.
- Skip cell types with too few samples per condition after the cell-count filter, and say so.
- Run a pseudobulk PCA to spot outlier samples or unmodeled covariates before trusting results.

## Differential abundance

- Proportions are compositional.
- Never use Fisher or chi-square tests on pooled cell counts.

## Pathways and TF activity

- Load resources with decoupler: `dc.op.progeny(organism=..., top=500)` for pathways, `dc.op.collectri(organism=...)` for TFs, `dc.op.hallmark(organism=...)` for gene sets.
- For condition contrasts, score the DE statistic rather than cells: `dc.mt.ulm(results[ct][["stat"]].T, net)` returns scores and p-values per source.
- For a gene list such as significant DE genes, `rsc.dcg.query_set(genes, net, background=tested_genes)` runs over-representation per set.
  Pass the genes that were tested as `background`, not the whole genome.
- Per-cell scores are for display: `rsc.dcg.ulm(adata, net)` writes `obsm["score_ulm"]` and `rsc.dcg.aucell` suits unweighted gene sets.
  View them with `dc.pp.get_obsm(adata, "score_ulm")` and `sc.pl`.
- Rank-based per-cell scores (AUCell) lose or flip signal in case-control comparisons, so do not test conditions on them.
- Score TF activity from targets (CollecTRI), not TF expression.
