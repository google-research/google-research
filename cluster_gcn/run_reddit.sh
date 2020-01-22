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

python train.py --dataset reddit --data_prefix ./data/ --nomultilabel --num_layers 4 --num_clusters 1500 --bsize 20 --hidden1 512 --dropout 0.2 --weight_decay 0  --early_stopping 200 --num_clusters_val 20 --num_clusters_test 1 --epochs 130 --save_name ./redditmodel --learning_rate 0.005 --diag_lambda 0.0001 --novalidation
