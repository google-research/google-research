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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install numpy
python -m bisimulation_aaai2020.grid_world.compute_metric \
  --base_dir=/tmp/grid_world_test \
  --grid_file=bisimulation_aaai2020/grid_world/configs/2state.grid \
  --gin_files=bisimulation_aaai2020/grid_world/configs/2state.gin \
  --nosampled_metric \
  --nolearn_metric \
  --nosample_distance_pairs
