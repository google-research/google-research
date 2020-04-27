suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scater)
  library(scran)
  library(DropletUtils)
  library(argparse)
  library(loomR)
  library(Seurat)
  library(purrr)
})

# create parser object
parser <- ArgumentParser()

# specify our desired options
# by default ArgumentParser will add an help option
parser$add_argument("--data_path",
  type = "character", required = TRUE,
  help = "Location of the matrices, where the script tries to read them."
)
parser$add_argument("--format",
  type = "character", required = TRUE,
  help = "Format to write the data in, must take a value in: 'SingleCellExperiment', 'Seurat', 'LoomR', and 'csv'."
)
parser$add_argument("--name",
  type = "character", required = TRUE,
  help = "Name of the dataset you want to create."
)
parser$add_argument("--seed",
  type = "integer", default = 1234,
  help = "Seed to use for the subsampling."
)

# QC related arguments.
parser$add_argument("--qc_count_mad_lower",
  type = "integer", default = 3,
  help = "Threshold for filtering cells based on the number of reads. Use a negative value if you don't want to filter"
)
parser$add_argument("--qc_feature_count_mad_lower",
  type = "integer", default = 3,
  help = "Threshold for filtering cells based on the number of expressed genes. Use a negative value if you don't want to filter"
)
parser$add_argument("--qc_mito_mad_upper",
  type = "integer", default = 3,
  help = "Threshold for filtering cells based on the number of mitochondrial RNA reads. Use a negative value if you don't want to filter"
)

# Cell mix
parser$add_argument("--b_cells",
  type = "integer", default = 0,
  help = "Number of CD19+ B cells in the mixture."
)
parser$add_argument("--naive_cytotoxic",
  type = "integer", default = 0,
  help = "Number of CD8+/CD45RA+ Naive Cytotoxic T Cells in the mixture,"
)
parser$add_argument("--cd14_monocytes",
  type = "integer", default = 0,
  help = "Number of CD14+ monocytes in the mixture,"
)
parser$add_argument("--regulatory_t",
  type = "integer", default = 0,
  help = "Number of CD4+/CD25+ Regulatory T Cells in the mixture,"
)
parser$add_argument("--cd56_nk",
  type = "integer", default = 0,
  help = "Number of CD56+ natural killer cells in the mixture,"
)
parser$add_argument("--cd4_t_helper",
  type = "integer", default = 0,
  help = "Number of CD4+ helper T cells in the mixture,"
)
parser$add_argument("--memory_t",
  type = "integer", default = 0,
  help = "Number of CD4+/CD45RO+ Memory T Cells in the mixture,"
)
parser$add_argument("--naive_t",
  type = "integer", default = 0,
  help = "Number of CD4+/CD45RA+/CD25- Naive T cells in the mixture,"
)
# Add new data here.

# Get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults,
args <- parser$parse_args()

generate_zheng <- function(dfs, qc_count_mad_lower, qc_feature_count_mad_lower, qc_mito_mad_upper, seed = 1234) {
  set.seed(seed)
  all <- map(dfs, function(arg) {
    arg$df[, sample(colnames(arg$df), arg$count, replace = FALSE)]
  })
  sce <- cbind(
    all$b_cells,
    all$naive_cytotoxic,
    all$cd14_monocytes,
    all$regulatory_t,
    all$cd4_t_helper,
    all$cd56_nk,
    all$memory_t,
    all$naive_t
    # Add new data here.
  )

  isSpike(sce, "ERCC") <- grepl("^ERCC", rowData(sce)$Symbol)
  sce <- calculateQCMetrics(sce, feature_controls = list(Mito = grepl("^MT-", rowData(sce)$Symbol)))

  libsize.drop <- rep(FALSE, length(colnames(sce)))
  if (qc_count_mad_lower >= 0) {
    libsize.drop <- isOutlier(sce$total_counts, nmads = qc_count_mad_lower, type = "lower", log = TRUE)
  }
  feature.drop <- rep(FALSE, length(colnames(sce)))
  if (qc_feature_count_mad_lower >= 0) {
    feature.drop <- isOutlier(sce$total_features_by_counts, nmads = qc_feature_count_mad_lower, type = "lower", log = TRUE)
  }
  mito.drop <- rep(FALSE, length(colnames(sce)))
  if (qc_mito_mad_upper >= 0) {
    mito.drop <- isOutlier(sce$pct_counts_Mito, nmads = qc_mito_mad_upper, type = "higher")
  }

  keep <- !(libsize.drop | feature.drop | mito.drop)
  table(keep)
  sce <- sce[, keep]


  num.cells <- nexprs(sce, byrow = TRUE)
  to.keep <- num.cells > 0
  table(!to.keep)
  sce <- sce[to.keep, ]
  sce
}


# Reading the data.
b_cells <- DropletUtils::read10xCounts(paste(args$data_path, "b_cells/filtered_matrices_mex/hg19", sep = "/"))
b_cells$label <- "b_cells"
memory_t <- DropletUtils::read10xCounts(paste(args$data_path, "memory_t/filtered_matrices_mex/hg19", sep = "/"))
memory_t$label <- "memory_t"
naive_t <- DropletUtils::read10xCounts(paste(args$data_path, "naive_t/filtered_matrices_mex/hg19", sep = "/"))
naive_t$label <- "naive_t"
cd56_nk <- DropletUtils::read10xCounts(paste(args$data_path, "cd56_nk/filtered_matrices_mex/hg19", sep = "/"))
cd56_nk$label <- "cd56_nk"
cd14_monocytes <- DropletUtils::read10xCounts(paste(args$data_path, "cd14_monocytes/filtered_matrices_mex/hg19", sep = "/"))
cd14_monocytes$label <- "cd14_monocytes"
cd4_t_helper <- DropletUtils::read10xCounts(paste(args$data_path, "cd4_t_helper/filtered_matrices_mex/hg19", sep = "/"))
cd4_t_helper$label <- "cd4_t_helper"
regulatory_t <- DropletUtils::read10xCounts(paste(args$data_path, "regulatory_t/filtered_matrices_mex/hg19", sep = "/"))
regulatory_t$label <- "regulatory_t"
naive_cytotoxic <- DropletUtils::read10xCounts(paste(args$data_path, "naive_cytotoxic/filtered_matrices_mex/hg19", sep = "/"))
naive_cytotoxic$label <- "naive_cytotoxic"
# Add new data here.

# Naming the Cells.
colnames(b_cells) <- paste0("b_cells", seq_len(ncol(b_cells)))
colnames(naive_cytotoxic) <- paste0("naive_cytotoxic", seq_len(ncol(naive_cytotoxic)))
colnames(cd14_monocytes) <- paste0("cd14_monocytes", seq_len(ncol(cd14_monocytes)))
colnames(regulatory_t) <- paste0("regulatory_t", seq_len(ncol(regulatory_t)))
colnames(cd4_t_helper) <- paste0("cd4_t_helper", seq_len(ncol(cd4_t_helper)))
colnames(cd56_nk) <- paste0("cd56_nk", seq_len(ncol(cd56_nk)))
colnames(memory_t) <- paste0("memory_t", seq_len(ncol(memory_t)))
colnames(naive_t) <- paste0("naive_t", seq_len(ncol(naive_t)))
# Add new data here.


sce_args <- list(
  b_cells = list(df = b_cells, count = args$b_cells),
  naive_cytotoxic = list(df = naive_cytotoxic, count = args$naive_cytotoxic),
  cd14_monocytes = list(df = cd14_monocytes, count = args$cd14_monocytes), # careful, only 2.6k
  regulatory_t = list(df = regulatory_t, count = args$regulatory_t),
  cd4_t_helper = list(df = cd4_t_helper, count = args$cd4_t_helper),
  cd56_nk = list(df = cd56_nk, count = args$cd56_nk),
  memory_t = list(df = memory_t, count = args$memory_t),
  naive_t = list(df = naive_t, count = args$naive_t)
  # Add new data here.
)
sce <- generate_zheng(sce_args,
  qc_count_mad_lower = args$qc_count_mad_lower,
  qc_feature_count_mad_lower = args$qc_feature_count_mad_lower,
  qc_mito_mad_upper = args$qc_mito_mad_upper,
  seed = args$seed
)

if (args$format == "SingleCellExperiment") {
  saveRDS(sce, paste(args$name, "rds", sep = "."))
}
if (args$format == "Seurat") {
  dat <- as.Seurat(sce, counts = "counts", data = "counts")
  saveRDS(dat, paste(args$name, "rds", sep = "."))
}
if (args$format == "LoomR") {
  dat <- as.Seurat(sce, counts = "counts", data = "counts")
  loom <- as.loom(dat, filename = paste(args$name, "rds", sep = "."), verbose = TRUE, overwrite = TRUE)
  loom$close_all()
}
if (args$format == "csv") {
  write.csv(as.matrix(counts(sce)), paste(args$name, "counts.csv", sep = "."))
  write.csv(colData(sce), paste(args$name, "metadata.csv", sep = "."))
  write.csv(rowData(sce), paste(args$name, "featuredata.csv", sep = "."))
}
