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



# Download the english corpus (about 33M passages) and the english queries
# about 1K~2K questions for [‘train’,’dev’,’testA’,’testB’]. The queries take
# a long time to process.
python download_miracl.py

# Generate the embeddings from “contriever-msmarco” (6 GPU hours).
python contriever/generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever-msmarco \
    --output_dir contriever_msmarco_miracl_embeddings  \
    --passages miracl/queries_en/queries_en_train.jsonl \
    --shard_id 0 --num_shards 1 \
# For multi-GPU, use: --shard_id 4 --num_shards 8

# Run the query-X-document retrieval using FAISS (40 minutes).
python contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages miracl/queries_en/queries_en_train.jsonl \
    --passages_embeddings "contriever_msmarco_miracl_embeddings/*" \
    --data lost-in-the-middle/qa_data/nq-open-oracle.jsonl \
    --output_dir contriever_msmarco_miracl_retrievals \
    --n_docs 1000