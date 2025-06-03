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



# Train all models on all datasets, with both forecasting and inventory objectives.

echo "M3 Naive Seasonal MSE"
python -m main --dataset_name m3 --model_name naive_seasonal --optimization_objs mse --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.00001 --nosweep --hidden_size 10 --max_steps 20 --learning_rate 0.01 --num_workers 2 --no_safety_stock --project_name m3_project --return_test_preds

echo "M3 Naive Seasonal Total Cost (trains separate models for each cost tradeoff)"
python -m main --dataset_name m3 --model_name naive_seasonal --optimization_objs total_cost --unit_holding_costs 1 --unit_holding_costs 2 --unit_holding_costs 10 --unit_stockout_costs 1 --unit_stockout_costs 2 --unit_stockout_costs 10 --unit_var_o_costs 0.00001 --unit_var_o_costs 0.000001 --nosweep --hidden_size 100 --max_steps 20 --learning_rate 0.0001 --num_workers 10 --no_safety_stock --project_name m3_project --return_test_preds

echo "M3 Naive Seasonal RRMS"
python -m main --dataset_name m3 --model_name naive_seasonal --optimization_objs rel_rms_stockout_2 --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.00001 --nosweep --hidden_size 10 --max_steps 10 --learning_rate 0.0001 --num_workers 10 --no_safety_stock --project_name m3_project --return_test_preds

echo "M3 LSTM MSE"
python -m main --dataset_name m3 --model_name lstm_windowed --optimization_objs mse --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.00001 --nosweep --hidden_size 20 --max_steps 10 --learning_rate 0.01 --num_workers 2 --no_safety_stock --project_name m3_project --return_test_preds

echo "M3 LSTM Total Cost (trains separate models for each cost tradeoff)"
python -m main --dataset_name m3 --model_name lstm_windowed --optimization_objs total_cost --unit_holding_costs 1 --unit_holding_costs 2 --unit_holding_costs 10 --unit_stockout_costs 1 --unit_stockout_costs 2 --unit_stockout_costs 10 --unit_var_o_costs 0.00001 --unit_var_o_costs 0.000001 --nosweep --hidden_size 20 --max_steps 20 --learning_rate 0.0001 --num_workers 10 --no_safety_stock --project_name m3_project --return_test_preds

echo "M3 LSTM RRMS"
python -m main --dataset_name m3 --model_name lstm_windowed --optimization_objs rel_rms_stockout_2 --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.00001 --nosweep --hidden_size 20 --max_steps 20 --learning_rate 0.0001 --num_workers 10 --no_safety_stock --project_name m3_project --return_test_preds

echo "Favorita (Full) Naive Seasonal MSE"
python -m main --dataset_name favorita --model_name naive_seasonal --optimization_objs mse --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.01 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.001 --batch_size 300 --num_batches 500 --num_workers 5 --project_name favorita_project --single_rollout --no_safety_stock --save latest

echo "Favorita (10K) Naive Seasonal Total Cost (trains separate models for each cost tradeoff)"
python -m main --dataset_name favorita --model_name naive_seasonal --optimization_objs total_cost --unit_holding_costs 1 --unit_holding_costs 2 --unit_holding_costs 10 --unit_stockout_costs 1 --unit_stockout_costs 2 --unit_stockout_costs 10 --unit_var_o_cost 0.01 --unit_var_o_cost 0.001 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.001 --batch_size 200 --num_batches 200 --num_workers 5 --project_name favorita_tradeoffs2 --N 10000 --single_rollout --no_safety_stock --save latest

echo "Favorita (Full) Naive Seasonal RRMS"
python -m main --dataset_name favorita --model_name naive_seasonal --optimization_objs rel_rms_stockout_2 --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.01 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.00001 --batch_size 300 --num_batches 500 --num_workers 5 --project_name favorita_project --single_rollout --no_safety_stock --save latest

echo "Favorita (Full) LSTM MSE"
python -m main --dataset_name favorita --model_name lstm_windowed --optimization_objs mse --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.01 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.001 --batch_size 300 --num_batches 500 --num_workers 5 --project_name favorita_project --single_rollout --no_safety_stock --save latest

echo "Favorita (10K) LSTM Total Cost (trains separate models for each cost tradeoff)"
python -m main --dataset_name favorita --model_name lstm_windowed --optimization_objs total_cost --unit_holding_costs 1 --unit_holding_costs 2 --unit_holding_costs 10 --unit_stockout_costs 1 --unit_stockout_costs 2 --unit_stockout_costs 10 --unit_var_o_cost 0.01 --unit_var_o_cost 0.001 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.001 --batch_size 200 --num_batches 200 --num_workers 5 --project_name favorita_project --N 10000 --single_rollout --no_safety_stock --save latest

echo "Favorita (Full) LSTM RRMS"
python -m main --dataset_name favorita --model_name lstm_windowed --optimization_objs rel_rms_stockout_2 --unit_holding_costs 1 --unit_stockout_costs 1 --unit_var_o_costs 0.0001 --nosweep --hidden_size 64 --max_steps 1 --learning_rate 0.00001 --batch_size 300 --num_batches 500 --num_workers 5 --project_name favorita_project --single_rollout --no_safety_stock --save latest

