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

pip install -r requirements.txt

python compute_observation_error.py --caps 1 --repetitions 1 --output observation_error.csv
python compute_direct_error.py --caps 1 --repetitions 1 --output direct_error.csv
python compute_cumulative_error.py --caps 1 --window_sizes 1 --repetitions 1 --output cumulative_error.csv
python compute_sketches_error.py --caps 1 --window_sizes 1 --repetitions 1 --output sketches_error.csv
