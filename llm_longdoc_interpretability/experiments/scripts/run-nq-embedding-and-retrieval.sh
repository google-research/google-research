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



# Generate the embeddings from “contriever-msmarco” (4 GPU hours)
PASSAGES=$1

# Run the query-X-document retrieval using FAISS (40 minutes)
NUM_DOCS=$2
python contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages $PASSAGES \
    --passages_embeddings "contriever_msmarco_embeddings/wikipedia_embeddings/*" \
    --data lost-in-the-middle/qa_data/nq-open-oracle.jsonl \
    --output_dir contriever_msmarco_nq \
    --n_docs $NUM_DOCS