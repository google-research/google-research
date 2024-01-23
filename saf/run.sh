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

set -e

mkdir -p logs

EXPERIMENT_VERSION=v0

SYN_OPTION=1
DATASET=synthetic_autoregressive
LEN_TOTAL=100
MODEL_NAME=lstm_seq2seq
python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=-1 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --num_trials=1 --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}
