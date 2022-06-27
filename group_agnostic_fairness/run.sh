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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip3 install -r group_agnostic_fairness/requirements.txt

# Select from the following models or tests to run.
python -m group_agnostic_fairness.main_trainer
python -m group_agnostic_fairness.main_trainer_test
python -m group_agnostic_fairness.adversarial_reweighting_model_test
python -m group_agnostic_fairness.fairness_metrics_test
python -m group_agnostic_fairness.baseline_model_test
python -m group_agnostic_fairness.ips_reweighting_model_test
