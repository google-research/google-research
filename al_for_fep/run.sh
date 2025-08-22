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

virtualenv -p python3 .
source ./bin/activate

pip install -r al_for_fep/requirements.txt
python3 -m al_for_fep.single_cycle_main --cycle_config al_for_fep/configs/simple_greedy_gaussian_process.py --cycle_config.cycle_dir ../cycle1 --cycle_config.training_pool al_for_fep/data/testdata/initial_training_set.csv --cycle_config.virtual_library=al_for_fep/data/testdata/virtual_library.csv