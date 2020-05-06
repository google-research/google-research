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
sudo R -q -e 'BiocManager::install("scran")'
sudo R -q -e 'BiocManager::install("DropletUtils")'

DATA_PATH=/tmp/data
bash cell_mixer/fetch_data.sh $DATA_PATH

Rscript cell_mixer/cell_mixer.R \
--data_path=$DATA_PATH \
--name=mixture \
--format=csv \
--b_cells=500 \
--naive_t=500

python3 -m cell_mixer.converter \
--input_csv_prefix=mixture \
--format=anndata
