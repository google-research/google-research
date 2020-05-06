# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

# Python dependencies
pip install -r cell_mixer/requirements.txt
pip install -r scrna_benchmark/requirements.txt

# Install R and R dependencies
sudo apt-get install r-base \
  libcurl4-openssl-dev \
  libhdf5-dev \
  libxml2-dev \
  libssl-dev \
  wget
sudo R -q -e 'install.packages("devtools")'
sudo R -q -e 'install.packages("argparse")'
sudo R -q -e 'install.packages("Seurat")'
sudo R -q -e 'install.packages("purrr")'
sudo R -q -e 'devtools::install_github(repo = "hhoeflin/hdf5r")'
sudo R -q -e 'devtools::install_github(repo = "mojaveazure/loomR", ref = "develop")'
sudo R -q -e 'install.packages("BiocManager")'
sudo R -q -e 'BiocManager::install("SingleCellExperiment")'
sudo R -q -e 'BiocManager::install("scater")'
sudo R -q -e 'BiocManager::install("zinbwave")'
sudo R -q -e 'BiocManager::install("scran")'
sudo R -q -e 'BiocManager::install("DropletUtils")'

DATA_PATH=/tmp/data
bash cell_mixer/fetch_data.sh $DATA_PATH

# We need to create some input data.
Rscript cell_mixer/cell_mixer.R \
--data_path=$DATA_PATH \
--name=mixture \
--format=SingleCellExperiment \
--b_cells=50 \
--naive_t=50

Rscript cell_mixer/cell_mixer.R \
--data_path=$DATA_PATH \
--name=mixture \
--format=csv \
--b_cells=50 \
--naive_t=50
python3 -m cell_mixer.converter \
--input_csv_prefix=mixture \
--format=anndata

# Check that scVI works.
python3 -m scrna_benchmark.scvi_process \
--epochs=1 \
--input_path=mixture.h5ad \
--output_csv=test.csv \
--log_path=scvi \
--n_layers=1 \
--n_hidden=32 \
--dispersion=gene \
--dropout_rate=0.1 \
--reconstruction_loss=zinb \
--n_latent=10 \
--lr=0.001

# Checks that DCA works.
python3 -m scrna_benchmark.dca_process \
--input_path=mixture.h5ad \
--output_csv=dca.csv \
--log_path=dca \
--ae_type=zinb \
--hidden_size=32,10,32 \
--hidden_dropout=0.1 \
--batch_size=32 \
--epochs=1 \
--seed=1 \
--log1p

# Common R parameters
SCE="mixture.rds"
LOOM="mixture.loom"
TISSUE="mixture"
DIMS=2
NTOPS=100

# scran parameters.
SUM_FACTOR=0
ERCC=0
ASSAY="logcounts"

# ZinbWAVE parameters.
EPSILON=2000
GENE_COVARIATE=1

# Seurat parameters.
NORM="LogNormalize"
VARIABLE="vst"

# Make sure that scran works.
Rscript scrna_benchmark/scran_process.R \
  --input_path="${SCE}" \
  --output_loom="${LOOM}"\
  --use_sum_factors="${SUM_FACTOR}" \
  --use_ERCC="${ERCC}" \
  --assay_type="${ASSAY}" \
  --n_pcs="${DIMS}" \
  --n_tops="${NTOPS}" \

python3 -m scrna_benchmark.compute_metrics \
  --input_loom="${LOOM}" \
  --output_csv="scran.csv" \
  --tissue="${TISSUE}" \
  --reduced_dim="PCA_cell_embeddings" \
  --source="scran" \
  --scran_sum_factor="${SUM_FACTOR}" \
  --scran_ercc="${ERCC}" \
  --scran_n_pcs="${DIMS}" \
  --scran_n_tops="${NTOPS}" \
  --scran_assay="${ASSAY}"

# Make sure that ZinbWAVE works.
Rscript scrna_benchmark/zinbwave.R \
  --input_path="${SCE}" \
  --output_loom="${LOOM}"\
  --zinb_dim="${DIMS}" \
  --epsilon="${EPSILON}" \
  --keep_variance="${NTOPS}" \
  --gene_covariate="${GENE_COVARIATE}"

python3 -m scrna_benchmark.compute_metrics \
  --input_loom="${LOOM}" \
  --output_csv="zinbwave.csv" \
  --tissue="${TISSUE}" \
  --reduced_dim="zinbwave_cell_embeddings" \
  --source="zinbwave" \
  --zinbwave_dims="${DIMS}" \
  --zinbwave_epsilon="${EPSILON}" \
  --zinbwave_keep_variance="${NTOPS}" \
  --zinbwave_gene_covariate="${GENE_COVARIATE}"

# Make sure that Seurat works.
Rscript scrna_benchmark/seurat.R \
  --input_path="${SCE}" \
  --output_loom="${LOOM}"\
  --n_pcs="${DIMS}" \
  --n_features="${NTOPS}" \
  --normalization_method="${NORM}" \
  --variable_features="${VARIABLE}"

python3 -m scrna_benchmark.compute_metrics \
  --input_loom="${LOOM}" \
  --output_csv="seurat.csv" \
  --tissue="${TISSUE}" \
  --reduced_dim="pca_cell_embeddings" \
  --source="seurat" \
  --seurat_norm="${NORM}" \
  --seurat_find_variable="${VARIABLE}" \
  --seurat_n_features="${NTOPS}" \
  --seurat_n_pcs="${DIMS}"

# Make sure that we can generate a dsub conf.
python3 -m  scrna_benchmark.generate_dsub_conf \
--cloud_input toto \
--cloud_output tata \
--conf_path .
