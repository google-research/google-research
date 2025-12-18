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

pip install -r f_divergence_tests/requirements.txt
python -m f_divergence_tests.test_two_samples \
  --experiment_name="kl_divergence_expo1d_test" \
  --divergence_class="kl_divergence" \
  --distribution="expo1d" \
  --distribution_params='{"n_samples": 500, "location": 4.0, "scale": 1.6, "multiplier": 90.0}' \
  --divergence_params='{}' \
  --divergence_sweep_params='{"lmbda": "[0.01, 0.1, 1.0]"}' \
  --kernel_type="gaussian" \
  --significance=0.05 \
  --num_permutations=200 \
  --num_bandwidths=5 \
  --seed_samples=42 \
  --seed_test=83