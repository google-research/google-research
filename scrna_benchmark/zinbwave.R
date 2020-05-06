#!/usr/bin/env Rscript
# Example use:
# time Rscript zinbwave.R \
# --input_path='Marrow.rds' \
# --output_loom='Marrow_zinb.loom' \
# --zinb_dim=10 \
# --keep_variance=100 \
# --gene_covariate=1 \
# --epsilon=1000
#
# Follows the vignette in https://bioconductor.org/packages/release/bioc/vignettes/zinbwave/inst/doc/intro.html

suppressPackageStartupMessages({
  library("argparse")
  library("SingleCellExperiment")
  library("scater")
  library("scran")
  library("Seurat")
  library("zinbwave")
  library("matrixStats")
  library("magrittr")
  library("loomR")
})

parser <- ArgumentParser()

parser$add_argument("--input_path",
  type = "character",
  help = "Input cleaned SingleCellExperiment file."
)
parser$add_argument("--output_loom",
  type = "character",
  help = "Output loom"
)
parser$add_argument("--zinb_dim",
  type = "integer",
  help = "Dimensions of the embedding space"
)
parser$add_argument("--epsilon",
  type = "integer",
  help = "epsilon parameter in zinbwave"
)
parser$add_argument("--keep_variance",
  type = "integer",
  help = "Number of highly variable genes to keep"
)
parser$add_argument("--gene_covariate",
  type = "integer",
  help = "Whether to use gene covariates in zinbwave, use 0 or 1"
)

# get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults,
args <- parser$parse_args()

write_sce_to_loom <- function(sce, out_path) {
  facs_seurat <- as.Seurat(sce, counts = "counts", data = "counts")
  # as.loom expects a layer for variable genes.
  facs_seurat <- FindVariableFeatures(facs_seurat)
  # as.loom does not know how to deal with NA
  Idents(facs_seurat) <- "None"
  # We need to have both normalized and scaled the data with Seurat, otherwise
  # Seurat does not write reducedDim to the loom object.
  facs_seurat <- NormalizeData(facs_seurat)
  facs_seurat <- ScaleData(facs_seurat)
  facs.loom <- as.loom(facs_seurat,
    filename = out_path,
    verbose = TRUE, overwrite = TRUE
  )
  facs.loom$close_all()
}

filter_by_variance <- function(sce, top = 100) {
  as.matrix(counts(sce)) %>%
    log1p() %>%
    rowVars() -> vars
  names(vars) <- rownames(sce)
  vars <- sort(vars, decreasing = TRUE)
  sce <- sce[names(vars)[1:top], ]
  counts(sce) <- as.matrix(counts(sce))
  sce
}

process_zinb <- function(input_path,
                         zinb_dim,
                         keep_variance,
                         gene_covariate,
                         epsilon,
                         out_path) {
  sce <- readRDS(input_path)

  sce <- filter_by_variance(sce, keep_variance)
  # We need to do this again, because some cells have zeroes for highly
  # variable genes
  sce <- calculateQCMetrics(sce)

  if (gene_covariate == 1) {
    covariates <- c("mean_counts", "n_cells_by_counts", "pct_dropout_by_counts")
    sce <- zinbwave(sce,
      K = zinb_dim,
      epsilon = epsilon,
      V = as.matrix(rowData(sce)[, covariates])
    )
  } else {
    sce <- zinbwave(sce, K = zinb_dim, epsilon = epsilon)
  }

  write_sce_to_loom(sce, out_path)
}

process_zinb(
  args$input_path,
  args$zinb_dim,
  args$keep_variance,
  args$gene_covariate,
  args$epsilon,
  args$output_loom
)
