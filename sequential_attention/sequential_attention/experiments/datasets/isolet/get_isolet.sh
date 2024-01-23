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

DATA_DIR='.'

ISOLET_TRAIN_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z
ISOLET_EVAL_URL=https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z
ISOLET_TRAIN_FILE=isolet1+2+3+4.data
ISOLET_EVAL_FILE=isolet5.data

curl "${ISOLET_TRAIN_URL}" --output "${ISOLET_TRAIN_FILE}.Z"
curl "${ISOLET_EVAL_URL}" --output "${ISOLET_EVAL_FILE}.Z"
uncompress "${ISOLET_TRAIN_FILE}.Z"
uncompress "${ISOLET_EVAL_FILE}.Z"