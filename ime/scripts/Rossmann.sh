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
python main.py --model MLP --data Rossmann --learning_rate 0.001 --layers 5 --d_model 128 --batch_size 256
python main.py --model Linear --data Rossmann --learning_rate 0.001 --batch_size 256
python main.py --model SDT --data Rossmann --learning_rate 0.0001 --batch_size 256

python main.py --model IME_WW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 1 --smoothness_hp 0.01 --diversity_hp 0
python main.py --model IME_WW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 0.1 --smoothness_hp 0.1 --diversity_hp 0  --expert_type SDT

python main.py --model IME_BW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 1 --smoothness_hp 0.01 --diversity_hp 0
python main.py --model IME_BW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 0.1 --smoothness_hp 0.1 --diversity_hp 0 --expert_type SDT
