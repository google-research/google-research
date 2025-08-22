#!/bin/bash
# Copyright 2025 The Google Research Authors.
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



# Set up virtual environment and install requirements.
virtualenv -p python3 .
source ./bin/activate
pip install -r requirements.txt

chmod +x scripts

# Download NQ dataset.
bash scripts/download-nq.sh

# Download FID model.
bash scripts/download-fid-model.sh

# Create a smaller subset of the data (n = 10) for this minimal example.
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar -P contriever_msmarco_embeddings/
tar xvf contriever_msmarco_embeddings/wikipedia_embeddings.tar -C contriever_msmarco_embeddings/
bash scripts/run-nq-embedding-and-retrieval.sh psgs_w100.tsv 100

# Generate embeddings for the NQ data subset and retrieve 1 document.
mkdir textgenshap-data-json
CUDA_VISIBLE_DEVICES=1 python saving_litm_interpretations_hf.py

# Look at the README to see other scripts that will run larger experiments.