"""Regenerate ``tests/_data/wnn_seurat_5.5.1.npz`` with R, Seurat 5.5.1 and RANN.

Run ``python tests/_scripts/generate_wnn_seurat.py [/path/to/Rscript]``. Seurat gets
the float32 inputs exactly (17 digits) and only its neighbor search is changed, to
exact RANN; the Python WNN and its NumPy test reference are not involved.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

R_CODE = """
suppressPackageStartupMessages(library(Seurat))
stopifnot(as.character(packageVersion("Seurat")) == "5.5.1")
# only the neighbor search changes, to exact RANN instead of Annoy
nn_helper <- Seurat:::NNHelper
assignInNamespace("NNHelper", ns = "Seurat", function(data, query = data, k, method, cache.index = FALSE, ...) {
  nn_helper(data = data, query = query, k = k, method = "rann", cache.index = cache.index)
})
pca <- as.matrix(read.csv("pca.csv", header = FALSE))
apca <- as.matrix(read.csv("apca.csv", header = FALSE))
cells <- paste0("cell", 1:200)
dimnames(pca) <- list(cells, paste0("PC_", 1:12))
dimnames(apca) <- list(cells, paste0("apca_", 1:8))
counts <- Matrix::sparseMatrix(i = rep(1, 200), j = 1:200, x = 1, dims = c(2, 200), dimnames = list(1:2, cells))
object <- CreateSeuratObject(counts = counts, assay = "RNA")
object[["ADT"]] <- CreateAssayObject(counts = counts)
object[["pca"]] <- CreateDimReducObject(embeddings = pca, key = "PC_", assay = "RNA")
object[["apca"]] <- CreateDimReducObject(embeddings = apca, key = "apca_", assay = "ADT")
object <- FindMultiModalNeighbors(
  object, reduction.list = list("pca", "apca"), dims.list = list(1:12, 1:8), k.nn = 20, knn.range = 40,
  l2.norm = TRUE, sd.scale = 1, smooth = FALSE, prune.SNN = 1 / 15, verbose = FALSE
)
save <- function(x, name) write.table(x, paste0(name, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)
save(cbind(object$RNA.weight, object$ADT.weight), "weights")
save(Indices(object[["weighted.nn"]]) - 1L, "indices")
save(Distances(object[["weighted.nn"]]), "distances")
save(as.matrix(object[["wsnn"]]), "snn")
cat(paste("Seurat", packageVersion("Seurat"), "; RANN", packageVersion("RANN")), R.version.string, sep = "\\n")
"""

rng = np.random.default_rng(0)
labels = rng.integers(0, 4, 200)
reps = {}
for name, dims, groups in (("pca", 12, 4), ("apca", 8, 2)):
    centers = rng.normal(size=(4, dims)) * 3
    reps[name] = (centers[labels % groups] + rng.normal(size=(200, dims))).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    for name, values in reps.items():
        np.savetxt(Path(tmp, f"{name}.csv"), values, delimiter=",", fmt="%.17g")
    rscript = sys.argv[1] if len(sys.argv) > 1 else "Rscript"
    run = subprocess.run([rscript, "-e", R_CODE], cwd=tmp, check=True, stdout=subprocess.PIPE, text=True)
    out = {name: np.loadtxt(Path(tmp, f"{name}.csv"), delimiter=",") for name in ("weights", "distances", "snn")}
    out["indices"] = np.loadtxt(Path(tmp, "indices.csv"), delimiter=",", dtype=np.int32)
provenance = f"""NumPy {np.__version__}
Synthetic inputs: NumPy default_rng(0), 200 cells, 12 RNA PCs / 8 ADT PCs.
{run.stdout.strip()}
FindMultiModalNeighbors with NNHelper redirected to exact RANN search.
k.nn=20; knn.range=40; l2.norm=TRUE; sd.scale=1; smooth=FALSE; prune.SNN=1/15.
Cross-modality constants use the Seurat default (1e-4).
Indices are zero-based; weights, distances and SNN values retain R precision.
Regenerate with tests/_scripts/generate_wnn_seurat.py.
"""
destination = Path(__file__).resolve().parents[1] / "_data" / "wnn_seurat_5.5.1.npz"
np.savez_compressed(destination, **reps, **out, provenance=provenance)
print(f"Wrote {destination}\n{provenance}")
