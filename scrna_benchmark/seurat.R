#!/usr/bin/env Rscript
# Example use:
# time Rscript seurat.R \
# --input_path='Marrow.rds' \
# --output_loom='Marrow_seurat.loom' \
# --n_pcs=10 \
# --n_features=200 \
# --normalization_method=LogNormalize \
# --variable_features=disp

suppressPackageStartupMessages({
  library("argparse")
  library("SingleCellExperiment")
  library("scran")
  library("scater")
  library("loomR")
  library("Seurat")
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
parser$add_argument("--normalization_method",
  type = "character",
  help = "Used in Seurat::NormalizeData. Must be one of 'LogNormalize' or 'CLR'"
)
parser$add_argument("--variable_features",
  type = "character",
  help = "Used in Seurat::FindVariableFeatures. Must be one of 'vst', 'mvp', or 'disp'"
)
parser$add_argument("--n_features",
  type = "integer",
  help = "Number of variable features to keep."
)
parser$add_argument("--n_pcs",
  type = "integer",
  help = "Number of PCs to compute."
)


args <- parser$parse_args()

save_loom_from_seurat <- function(dat, path) {
  facs_loom <- as.loom(dat, filename = path, verbose = TRUE, overwrite = TRUE)
  facs_loom$close_all()
}

process_seurat <- function(input_path,
                           normalization,
                           variable_features,
                           n_pcs,
                           n_features,
                           out_path) {
  sce <- readRDS(input_path)
  dat <- as.Seurat(sce, counts = "counts", data = "counts")

  dat <- NormalizeData(dat,
    normalization.method = normalization,
    scale.factor = 10000
  )
  dat <- FindVariableFeatures(dat,
    selection.method = variable_features,
    nfeatures = n_features
  )

  all.genes <- rownames(dat)
  dat <- ScaleData(dat, features = all.genes)

  dat <- RunPCA(dat, features = VariableFeatures(object = dat), npcs = n_pcs)

  save_loom_from_seurat(dat, out_path)
}

process_seurat(
  args$input_path,
  args$normalization_method,
  args$variable_features,
  args$n_pcs,
  args$n_features,
  args$output_loom
)
