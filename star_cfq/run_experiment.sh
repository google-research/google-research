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

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: run_experiment.sh <experiment_file> [steps]"
  exit 1
fi

EXPERIMENT=$1
# Default 500k steps as in *-CFQ paper.
STEPS=${2:-500000}

for SPLIT in $(cat split_sets/${EXPERIMENT}); do
  echo "Running ${SPLIT}"
  bash _run_cfq.sh ${SPLIT} ${STEPS}
done
