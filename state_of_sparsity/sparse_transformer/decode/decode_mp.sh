# Copyright 2024 The Google Research Authors.
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

output_dir=$1
base_dir="state_of_sparsity/sparse_transformer/decode"

python -m state_of_sparsity.sparse_transformer.decoder \
  --hparams_set="sparse_transformer_magnitude_pruning_tpu" \
  --output_dir=$output_dir \
  --model=sparse_transformer \
  --problem=translate_ende_wmt32k_packed \
  --data_dir=$base_dir/wmt_ende32k \
  --decode_from_file=$base_dir/newstest2014.en \
  --hparams="symbol_modality_num_shards=1"
