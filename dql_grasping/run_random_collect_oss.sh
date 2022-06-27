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

#!/bin/bash -eux

# Open-source version of random collection script.
ENV="env_procedural"
RANDOM_TYPE="random"
DIR_SUFFIX="100"

ROOT_DIR="${HOME}/qgrasping/${ENV}/${RANDOM_TYPE}_${DIR_SUFFIX}"
echo "ROOT_DIR=${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/policy_collect"

BASE_DIR="dql_grasping"

GINCONFIG="
$(cat "${BASE_DIR}/configs/${ENV}/grasping_env.gin")
$(cat "${BASE_DIR}/configs/${ENV}/run_${RANDOM_TYPE}.gin")
train_collect_eval.num_collect = 2  # Kept small for integration testing.
"

# RUN LOCAL OR LAUNCH TO BORG
python -m dql_grasping.run_train_collect_eval \
--gin_config "$GINCONFIG" \
--run_mode "collect_eval_once" \
--root_dir "${ROOT_DIR}" \
--task "0" \
--logtostderr
