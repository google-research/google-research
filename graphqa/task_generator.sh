#!/bin/bash
# Copyright 2025 The Google Research Authors.
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
set -x

python3 -m venv graphqa
source graphqa/bin/activate

pip3 install -r graphqa/requirements.txt

# Fill in appropriate output path
GRAPHS_DIR="/usr/local/google/home/baharef/temp/graphqa"
TASK_DIR="/usr/local/google/home/baharef/temp/graphqa/tasks"
TASKS=("edge_existence" "node_degree" "node_count" "edge_count" "cycle_check" "connected_nodes")

# For experimenting with only erdos-reyni graph use `er``.
# For all graph generators, set to `all`.
ALGORITHM="er"

echo "The output path is set to: $TASK_DIR"

for  task in "${TASKS[@]}"
do
  echo "Generating examples for task $task"
  python3 -m graphqa.graph_task_generator \
                --task=$task \
                --algorithm=$ALGORITHM \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234
done
