#!/usr/bin/env Rscript
# Example use:
# time Rscript paired_tag_analysis.R \
# --input_path='Paired-Tag_H3K27me3_DNA_filtered_matrix' \
# --output_path='Paired-Tag_H3K27me3_DNA_filtered_matrix'

suppressPackageStartupMessages({
  library(argparse)
  library(SingleCellExperiment)
  library(Matrix)
  library(irlba)
  library(cisTopic)
  library(Seurat)
  library(Signac)
  library(DropletUtils)
})

parser <- ArgumentParser()

parser$add_argument("--input_path",
  type = "character",
  help = "Input folder containing the MTX formated matrix."
)
parser$add_argument("--output_path",
  type = "character",
  help = "Output basepath for the embeddings."
)

# get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults,
args <- parser$parse_args()

create_sce <- function(path, data_source) {
  expression_matrix <- ReadMtx(
    mtx = file.path(path, "matrix.mtx"),
    features = file.path(path, "bins.tsv"),
    cells = file.path(path, "barcodes.tsv")
  )
  seurat_object <- CreateSeuratObject(counts = expression_matrix)
  return(as.SingleCellExperiment(seurat_object))
}

###################
#  Vallot LSI     #
###################

TFIDF <- function(scExp, scale = 10000, log = TRUE) {
  counts <- SingleCellExperiment::counts(scExp)
  n_features <- Matrix::colSums(counts)
  tf <- Matrix::t(Matrix::t(counts) / n_features)
  idf <- 1 + ncol(counts) / Matrix::rowSums(counts)
  normcounts <- Matrix::Diagonal(length(idf), idf) %*% tf
  if (log) {
    normcounts <- log1p(normcounts * scale)
  } else {
    normcounts <-
      normcounts * scale
  }
  SummarizedExperiment::assay(scExp, "normcounts",
    withDimnames = FALSE
  ) <- normcounts
  return(scExp)
}

pca <- function(x, n_comp, work = 3 * n_comp) {
  x.means <- Matrix::colMeans(x)
  svd.0 <- irlba::irlba(x, center = x.means, nv = n_comp, work = work)
  pca <- svd.0$u %*% diag(svd.0$d)
  return(pca)
}

tpm_norm <- function(scExp) {
  SummarizedExperiment::assay(scExp, "normcounts", withDimnames = FALSE) <-
    10^6 * Matrix::t(Matrix::t(SingleCellExperiment::counts(scExp)) /
      Matrix::colSums(SingleCellExperiment::counts(scExp)))
  return(scExp)
}

##################
# snapATAC       #
##################

calJaccard <- function(X_i, X_j) {
  A <- Matrix::tcrossprod(X_i, X_j)
  bi <- Matrix::rowSums(X_i)
  bj <- Matrix::rowSums(X_j)
  jmat <- as.matrix(A / (replicate(ncol(A), bi) + t(replicate(nrow(A), bj)) - A))
  return(jmat)
}

normJaccard <- function(jmat, b1, b2, method, k = 15) {
  # estimate the expected jaccard index using OVN
  #' @importFrom raster focal raster
  .normOVN <- function(o, p1, p2, k) {
    # sort the jaccard index matrix based on the coverage
    ind1 <- order(p1)
    ind2 <- order(p2)
    o_srt <- as.matrix(o[ind1, ind2, drop = FALSE])
    # calculate expected jaccard index
    mask_mat <- matrix(1, k, k)
    exp <- focal(raster(as.matrix(o_srt)), mask_mat, mean, na.rm = TRUE, pad = T)
    ee <- raster::as.matrix(exp)[order(ind1), order(ind2), drop = FALSE]
    return(ee)
  }

  # estimate the expected jaccard index using OVE
  .normOVE <- function(o, p1, p2, k) {
    pp <- tcrossprod(p1, p2)
    ss <- matrix(rep(p1, each = length(p2)), ncol = length(p2), byrow = TRUE) + matrix(rep(p2, each = length(p1)), ncol = length(p2), byrow = FALSE)
    ee <- pp / (ss - pp)
    return(ee)
  }

  jmat[jmat == 1] <- mean(jmat)
  x <- jmat
  emat <- eval(parse(text = paste0(".", method, "(x, b1, b2, k)")))
  if (method == "normOVE") {
    data <- data.frame(x = c(emat), y = c(jmat))
    model <- stats::lm(y ~ x, data)
    nmat <- matrix(model$residuals, nrow(emat), ncol(emat))
  } else if (method == "normOVN") {
    nmat <- jmat - emat
  }
  rm(jmat)
  rm(emat)
  return(nmat)
}

run_pca <- function(mat,
                    num_pcs = 50,
                    remove_first_PC = FALSE,
                    scale = FALSE,
                    center = FALSE) {
  set.seed(2019)
  SVD <- irlba(mat, num_pcs, num_pcs, scale = scale, center = center)
  sk_diag <- matrix(0, nrow = num_pcs, ncol = num_pcs)
  diag(sk_diag) <- SVD$d
  if (remove_first_PC) {
    sk_diag[1, 1] <- 0
    SVD_vd <- (sk_diag %*% t(SVD$v))[2:num_pcs, ]
  } else {
    SVD_vd <- sk_diag %*% t(SVD$v)
  }
  return(SVD_vd)
}

run_SnapATAC_normalize <- function(se) {
  binary_mat <- Matrix((assays(se)$counts > 0) + 0, sparse = TRUE)
  mat.use <- t(binary_mat)
  set.seed(2019)
  mat.ref <- mat.use
  jmat <- calJaccard(mat.use, mat.ref)
  p1 <- Matrix::rowMeans(mat.use)
  p2 <- Matrix::rowMeans(mat.ref)
  nmat <- normJaccard(
    jmat = jmat, b1 = p1, b2 = p2,
    method = "normOVE", k = 15
  )
  args <- list(A = nmat, nv = 50)
  args$center <- colMeans(nmat)
  x.norm <- sweep(args$A, 2, args$center, FUN = `-`)
  return(x.norm)
}

run_SnapATAC <- function(mat_SnapATAC, num_pcs = 10) {
  fm_SnapATAC <- run_pca(mat_SnapATAC, num_pcs = num_pcs)
  return(fm_SnapATAC)
}


#####################
# cisTopic          #
#####################

run_cisTopic <- function(se) {
  set.seed(2019)
  cistopics_counts <- assays(se)$counts
  rownames(cistopics_counts) <- lapply(strsplit(rownames(se), "_"),
    FUN = function(x) {
      paste(paste(x[1], x[2], sep = ":"), x[3], sep = "-")
    }
  )
  cisTopicObject <- createcisTopicObject(cistopics_counts,
    project.name = "cisTopic"
  )
  cisTopicObject <- runWarpLDAModels(cisTopicObject,
    topic = c(2, 5:15, 20, 25),
    seed = 2019,
    nCores = 2,
    iterations = 150,
    addModels = FALSE
  )
  cisTopicObject <- selectModel(cisTopicObject)
  fm_cisTopic <- modelMatSelection(cisTopicObject, "cell", "Probability")
  return(t(fm_cisTopic))
}

#####################
# Full method calls #
#####################

process_lsi <- function(data_path, data_source) {
  sce <- create_sce(data_path, data_source)
  pca_emb <- pca(t(assay(TFIDF(sce), "normcounts")), 10)[, -1]
  reducedDim(sce, "lsi") <- pca_emb
  return(sce)
}

process_pca <- function(data_path, data_source) {
  sce <- create_sce(data_path, data_source)
  pca_emb <- pca(t(assay(tpm_norm(sce), "normcounts")), 10)[, -1]
  reducedDim(sce, "pca") <- pca_emb
  return(sce)
}

process_snapatac <- function(data_path, data_source) {
  sce <- create_sce(data_path, data_source)
  tmp <- run_SnapATAC_normalize(sce)
  fm_SnapATAC <- run_SnapATAC(tmp, num_pcs = 10)
  reducedDim(sce, "SnapATAC") <- t(fm_SnapATAC)
  return(sce)
}

process_cistopic <- function(data_path, data_source) {
  sce <- create_sce(data_path, data_source)
  cis_emb <- run_cisTopic(sce)
  reducedDim(sce, "cisTopic") <- cis_emb
  return(sce)
}

process_signac <- function(data_path, data_source) {
  sce <- create_sce(data_path, data_source)
  dat <- as.Seurat(sce, counts = "counts", data = "counts")
  dat <- RunTFIDF(dat)
  dat <- FindTopFeatures(dat, min.cutoff = "q0")
  dat <- RunSVD(dat)
  sce <- as.SingleCellExperiment(dat)
  return(sce)
}

lsi <- process_lsi(args$input_path, args$data_source)
write.csv(
  reducedDim(lsi, "lsi"),
  file.path(args$output_path, "LSI.csv")
)
rm(lsi)
gc()
print("Done with LSI")

snap <- process_snapatac(args$input_path, args$data_source)
write.csv(
  reducedDim(snap, "SnapATAC"),
  file.path(args$output_path, "SnapATAC.csv")
)
rm(snap)
gc()
print("Done with SnapATAC")

signac <- process_signac(args$input_path, args$data_source)
write.csv(
  reducedDim(signac, "LSI"),
  file.path(args$output_path, "Signac.csv")
)
rm(signac)
gc()
print("Done with signac")

cis <- process_cistopic(args$input_path, args$data_source)
write.csv(
  reducedDim(cis, "cisTopic"),
  file.path(args$output_path, "cisTopic.csv")
)
rm(cis)
gc()
print("Done with cisTopic")

pca_tpm <- process_pca(args$input_path, args$data_source)
write.csv(
  reducedDim(pca_tpm, "pca"),
  file.path(args$output_path, "pca.csv")
)
rm(pca_tpm)
gc()
print("Done with PCA")
