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




# Get random ordering of strategies
python generate_random_model_order.py

# Train some number of randomly selected strategies (this example shows 100 distributed across 5 GPUs)
# Python script arguments:  GPU to use, dataset name, path to data, starting point in the random ordering, # of hyperparameter settings to iterate through
python train_models.py 0 physionet2012 data/physionet_data_processed/ 0 20 &
python train_models.py 1 physionet2012 data/physionet_data_processed/ 20 20 &
python train_models.py 2 physionet2012 data/physionet_data_processed/ 40 20 &
python train_models.py 3 physionet2012 data/physionet_data_processed/ 60 20 &
python train_models.py 4 physionet2012 data/physionet_data_processed/ 80 20

wait

# Get random ordering of strategies
python get_top_methods.py physionet2012

# Get test results for top method
# Python script arguments:  GPU to use, dataset name, path to data, starting point in the final ordering, # of hyperparameter settings to iterate through
python get_test_res.py 0 physionet2012 data/physionet_data_processed/ 0 1