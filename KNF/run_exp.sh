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
# M4-Weekly. To reproduce results in the paper, ensemble five KNF models with best hyperparameters trained with random seeds, e.g. 12, 20, 151, 155, 1312.
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=84 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 &
# Cryptos
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=3 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=256 --num_layers=5 --latent_dim=32 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=9 --num_steps=7 &
# NBA Player Trajectory
CUDA_VISIBLE_DEVICES=3 python3 run_koopman.py --seed=5 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=256 --latent_dim=64 --num_layers=6 --control_num_layers=5 --control_hidden_dim=128 --transformer_dim=64 --transformer_num_layers=4 --batch_size=128 --test_output_length=30 --num_steps=9 --jumps=1 --learning_rate=0.001 &
wait
