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

checkpoint_dir=$1
data_dir=$2
log_alpha_threshold=$3

python -m state_of_sparsity.sparse_rn50.imagenet_train_eval \
  --data_directory=$data_dir \
  --output_dir=$checkpoint_dir \
  --mode="eval_once" \
  --pruning_method="variational_dropout" \
  --eval_batch_size=100 \
  --log_alpha_threshold=$log_alpha_threshold
