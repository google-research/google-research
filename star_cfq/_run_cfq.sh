# Copyright 2021 The Google Research Authors.
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

if [[ $# -ne 2 ]]; then
  echo "Usage: _run_cfq.sh <split> <steps>"
  exit 1
fi

SPLIT=$1
STEPS=$2
DATASET=star_cfq
MODEL=transformer
SAVE_PATH=t2t_data/${DATASET}/${SPLIT}/${MODEL}

# Run preprocessor first.
PATH="$HOME/.local/bin:$PATH" python3.7 \
  -m cfq.preprocess_main \
  --dataset="${DATASET}" --split="${SPLIT}" \
  --save_path="${SAVE_PATH}"

# Run the rest of the pipeline.
PATH="$HOME/.local/bin:$PATH" python3.7 \
  -m cfq.run_experiment \
  --dataset="${DATASET}" --split="${SPLIT}" \
  --model="${MODEL}" --hparams_set=cfq_transformer \
  --train_steps="${STEPS}"
