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



# Small-scale M3 example (20 series), with total cost objective
python -m main \
--dataset_name m3 \
--model_name naive_seasonal \
--optimization_objs total_cost \
--unit_holding_costs 1 \
--unit_stockout_costs 1 \
--unit_var_o_costs 0.000001 \
--nosweep \
--hidden_size 20 \
--max_steps 5 \
--learning_rate 0.0001 \
--num_workers 10 \
--no_safety_stock \
--project_name m3_example \
--N 20 \
--return_test_preds
