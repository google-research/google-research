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

virtualenv -p python3 env
source env/bin/activate

pip3 install -r pruning_identified_exemplars/requirements.txt

output_dir="/tmp/"

python -m pruning_identified_exemplars.save_checkpoint.imagenet_train_eval --test_small_sample="True"
python -m pruning_identified_exemplars.pie_dataset_gen.imagenet_predictions --test_small_sample="True"
