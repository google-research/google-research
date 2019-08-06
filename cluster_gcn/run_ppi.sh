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

python train.py --dataset ppi --data_prefix ./data/ --multilabel --num_layers 5 --num_clusters 50 --bsize 1 --layernorm --precalc --hidden1 2048 --dropout 0.2 --weight_decay 0  --early_stopping 1000 --num_clusters_val 2 --num_clusters_test 1 --epochs 400 --save_name ./ppimodel --diag_lambda 1
