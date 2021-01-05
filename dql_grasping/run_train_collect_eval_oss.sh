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

#!/bin/bash -eux

ENV="env_procedural"
RANDOM_DIR="random_100"
EXPT="dqn"
TRAIN_MODE="onpolicy"
SUFFIX=""
DATA_FORMAT="tfrecord"
TASK="0"

ROOT_DIR="${HOME}/qgrasping/${ENV}"

###############################################################################
BASE_DIR="dql_grasping"
# SET UP GIN CONFIG
GINCONFIG="
    $(cat "${BASE_DIR}/configs/${ENV}/grasping_env.gin")
    $(cat "${BASE_DIR}/configs/${ENV}/train_${EXPT}.gin")
    train_collect_eval.file_patterns = '${ROOT_DIR}/${RANDOM_DIR}/policy_collect/*.${DATA_FORMAT}'
    "
echo "$GINCONFIG"

EXPT_DIR="${ROOT_DIR}/${EXPT}_${TRAIN_MODE}_${SUFFIX}"

# SET UP PATHS
if [[ "${TRAIN_MODE}" == "onpolicy" ]]; then
  # Train on random policy data + any new data collected by the trained policy.
  GINCONFIG="
    ${GINCONFIG}
    run_env.replay_writer = @TFRecordReplayWriter()
    train_collect_eval.onpolicy = True
  "
else
  GINCONFIG="
    ${GINCONFIG}
    train_collect_eval.onpolicy = False
  "
fi # END PATHS

mkdir -p "${EXPT_DIR}"

python -m dql_grasping.run_train_collect_eval \
--gin_config "$GINCONFIG" \
--root_dir "${EXPT_DIR}" \
--logtostderr \
--task "${TASK}"

