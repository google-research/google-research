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

GIN_CONFIG='configs/linear_best.gin'
LEARNING_RATE="0.0001"
DATASET_NAME=bert_mean_emb
DATASET="roc_stories_embeddings/$DATASET_NAME"

python src/train.py \
--save_dir=saved_checkpoints \
--data_dir='tfds_datasets/' \
--gin_config="$GIN_CONFIG" \
--gin_bindings="dataset.dataset_name = '${DATASET}'" \
--gin_bindings="train.learning_rate = ${LEARNING_RATE}" \
--gin_bindings="LinearModel.hparams.small_context_loss_weight = 1.0" \
