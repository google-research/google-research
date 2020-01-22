# Copyright 2020 The Google Research Authors.
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

# train or eval.
JOB=${1}

EXP="mnist"

# Specify data directory below.
DATA_DIR=""

if [[ "${JOB}" == "train" ]]
then
  python ../train.py \
  --experiment_type="${EXP}" \
  --batch_size=128 \
  --num_epochs=300 \
  --data_dir="${DATA_DIR}" \
  --logtostderr

elif [[ "${JOB}" == "eval" ]]
then
  python ../eval.py \
  --experiment_type="${EXP}" \
  --batch_size=10 \
  --data_dir="${DATA_DIR}" \
  --logtostderr
fi
