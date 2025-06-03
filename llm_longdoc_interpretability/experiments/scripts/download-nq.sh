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



# Download the LitM codebase.
git clone https://github.com/nelson-liu/lost-in-the-middle.git

# Decompress the data.
# gunzip lost-in-the-middle/qa_data/nq-open-oracle.jsonl.gz
gunzip lost-in-the-middle/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz

# The data to run the majority of the original experiments can be found in
# /qa_data/ with the prefetched relevant and filtered documents in the three files
# for 10, 20, and 30 documents. These latter three are only necessary if
# you're replicating the original paper.

# gunzip lost-in-the-middle/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_*
# gunzip lost-in-the-middle/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_*
# gunzip lost-in-the-middle/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_*

# Download the contriever repository
git clone https://github.com/facebookresearch/contriever.git

# Download the wikipedia passages (about 23M)
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip psgs_w100.tsv.gz