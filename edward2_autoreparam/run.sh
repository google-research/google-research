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

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r edward2_autoreparam/requirements.txt
python -m edward2_autoreparam.run_experiments --method=baseline \
          --model=8schools --num_leapfrog_steps=1 \
          --num_mc_samples=1 --num_optimization_steps=5 --num_samples=5 \
          --burnin=2 --num_adaptation_steps=2 --results_dir=/tmp/results
python -m edward2_autoreparam.run_experiments --method=vip \
          --model=8schools --num_leapfrog_steps=1 \
          --num_mc_samples=1 --num_optimization_steps=5 --num_samples=5 \
          --burnin=2 --num_adaptation_steps=2 --results_dir=/tmp/results
python -m edward2_autoreparam.analyze_results --results_dir=/tmp/results \
          --model_and_dataset=8schools_na
