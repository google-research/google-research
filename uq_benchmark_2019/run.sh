# Copyright 2019 The Google Research Authors.
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

# To run:
# 1. Create a virtual environment:
#   virtualenv myenv && source myenv/bin/activate
# 2. Install dependencies:
#   pip install -r uq_benchmark_2019/requirements.txt
# 3. Run tests:
#   bash uq_benchmark_2019/run.sh

set -e  # exit on error

if [ ! -d robustness_dhtd ]
then
  git clone https://github.com/hendrycks/robustness.git robustness_dhtd
fi

python -m uq_benchmark_2019.array_utils_test
python -m uq_benchmark_2019.uq_utils_test
python -m uq_benchmark_2019.calibration_lib_test
python -m uq_benchmark_2019.mnist.experiment_test
python -m uq_benchmark_2019.cifar.data_lib_test
python -m uq_benchmark_2019.cifar.end_to_end_test
python -m uq_benchmark_2019.criteo.data_lib_test --criteo_dummy_path_for_test=uq_benchmark_2019/criteo/criteo_testdata.tfr
python -m uq_benchmark_2019.criteo.run_train_test --criteo_dummy_path_for_test=uq_benchmark_2019/criteo/criteo_testdata.tfr
