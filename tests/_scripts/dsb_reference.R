# Reference values for tests/test_dsb.py (dsb 2.0.1, mclust 6.1.3, limma >= 3.66)
#   Rscript tests/_scripts/dsb_reference.R tests/_data/dsb
suppressPackageStartupMessages(library(dsb))
stopifnot(packageVersion("limma") >= "3.66.0")
out <- commandArgs(trailingOnly = TRUE)[1]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cells <- dsb::cells_citeseq_mtx[, 1:100]
empty <- dsb::empty_drop_citeseq_mtx[, 1:500]
iso <- rownames(cells)[67:70]
w <- function(m, name) {
  con <- gzfile(file.path(out, paste0(name, ".csv.gz")), "w")
  write.csv(m, con, row.names = FALSE)
  close(con)
}
w(t(as.matrix(cells)), "cells")
w(t(as.matrix(empty)), "empty")
quiet <- function(expr) { invisible(capture.output(r <- expr)); r }
r <- quiet(DSBNormalizeProtein(cells, empty, isotype.control.name.vec = iso,
                               return.stats = TRUE))
w(t(r$dsb_normalized_matrix), "default")
w(r$technical_stats, "default_technical_stats")
r <- quiet(DSBNormalizeProtein(cells, empty, isotype.control.name.vec = iso,
                               define.pseudocount = TRUE, pseudocount.use = 5,
                               scale.factor = "mean.subtract", quantile.clipping = TRUE))
w(t(r), "meansub_clip")
r <- quiet(ModelNegativeADTnorm(cells, isotype.control.name.vec = iso))
w(t(r), "negative")
r <- quiet(DSBNormalizeProtein(cells, empty, use.isotype.control = FALSE))
w(t(r), "no_isotypes")
r <- quiet(ModelNegativeADTnorm(cells, use.isotype.control = FALSE))
w(t(r), "negative_no_isotypes")
# Mclust fits of vectors longer than 2000 values from given subsets
suppressPackageStartupMessages(library(mclust))
set.seed(1)
n <- 2500
x <- cbind(log(rpois(n, 1.5) + 1), c(rnorm(1700), rnorm(n - 1700, 4, 1.5)),
           log(rpois(n, rgamma(n, 2, 0.2)) + 10))
idx <- sapply(seq_len(ncol(x)), function(j) sample(n, 2000))
fits <- t(sapply(seq_len(ncol(x)), function(j) {
  g <- Mclust(x[, j], G = 2, verbose = FALSE,
                      initialization = list(subset = idx[, j]))
  p <- g$parameters
  c(p$mean, rep_len(p$variance$sigmasq, 2), p$pro)
}))
w(x, "subset_x")
w(idx, "subset_idx")
w(fits, "subset_params")
