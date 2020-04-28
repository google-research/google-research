#!/usr/bin/env Rscript
# Example use:
# time Rscript scran_process.R \
# --input_path='Zhengmix4eq.rds' \
# --output_loom='Zhengmix4eq.loom' \
# --use_ERCC=1 \
# --use_sum_factors=1 \
# --n_pcs=10 \
# --n_tops=500 \
# --assay_type='logcounts'

suppressPackageStartupMessages({
  library("argparse")
  library("SingleCellExperiment")
  library("scran")
  library("scater")
  library("Seurat")
  library("loomR")
})

parser <- ArgumentParser()

parser$add_argument("--input_path",
  type = "character",
  help = "Tissue to run on, cleaned SingleCellExperiment"
)
parser$add_argument("--output_loom",
  type = "character",
  help = "Output loom"
)
parser$add_argument("--use_sum_factors",
  type = "integer",
  help = "Whether to use sum factors for the normalization"
)
parser$add_argument("--use_ERCC",
  type = "integer",
  help = "Whether to use the ERCC normalization."
)
parser$add_argument("--assay_type",
  type = "character",
  help = "Which assay to run PCA on, must be one of `logcounts` or `counts`"
)
parser$add_argument("--n_pcs",
  type = "integer",
  help = "Number of PCs to compute"
)
parser$add_argument("--n_tops",
  type = "integer",
  help = "Number of genes to use for PCA"
)

args <- parser$parse_args()

write_sce_to_loom <- function(sce, out_path) {
  facs_seurat <- as.Seurat(sce, counts = "counts", data = "counts")
  # as.loom expects a layer for variable genes.
  facs_seurat <- FindVariableFeatures(facs_seurat)
  # as.loom does not know how to deal with NA.
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

process_sample <- function(sce,
                           use_sum_factors,
                           use_ercc,
                           assay_type,
                           n_pcs,
                           n_tops) {
  dat <- sce
  isSpike(dat, "ERCC") <- grepl("^ERCC", rownames(dat))

  if (use_sum_factors) {
    dat <- computeSumFactors(dat)
  }
  if (use_ercc) {
    dat <- computeSpikeFactors(dat, type = "ERCC", general.use = FALSE)
  }

  if (assay_type == "logcounts") {
    dat <- normalize(dat)
  } else if (assay_type == "counts") {
    dat <- normalize(dat, return_log = FALSE)
  }

  dat <- runPCA(dat, exprs_values = assay_type, ncomponents = n_pcs, ntop = n_tops)

  dat
}

sce <- readRDS(args$input_path)
use_sum_factors <- args$use_sum_factors == 1
use_ERCC <- args$use_ERCC == 1
processed.data <- process_sample(
  sce,
  use_sum_factors,
  use_ERCC,
  args$assay_type,
  args$n_pcs,
  args$n_tops
)
write_sce_to_loom(processed.data, out_path = args$output_loom)
