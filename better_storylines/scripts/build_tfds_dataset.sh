# Copyright 2022 The Google Research Authors.
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

ROOT_DIR='.'

wget -nc https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip -f cased_L-12_H-768_A-12.zip

python -m tensorflow_datasets.scripts.download_and_prepare \
  --checksums_dir="${ROOT_DIR}/tfds_datasets/url_checksums" \
  --datasets='roc_stories_embeddings/bert_mean_emb' \
  --module_import='src.rocstories_sentence_embeddings' \
  --register_checksums \
  --data_dir="${ROOT_DIR}/tfds_datasets"


