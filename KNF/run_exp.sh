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


# We already have well-trained models and their predictions saved here. https://drive.google.com/drive/folders/1N_wo1I7G62HglyML5yL4FTEzbfekh4vZ?usp=sharing

########################################################################################################
# On M4, we only report a single number which is sMAPE of the averaged predictions from multiple runs.
########################################################################################################

# M4-Weekly. The sMAPE of the average of predictions from following five models is 7.254.
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=901 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 -decay_rate=0.85 &
CUDA_VISIBLE_DEVICES=1 python3 run_koopman.py --seed=81 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 -decay_rate=0.85 &
CUDA_VISIBLE_DEVICES=2 python3 run_koopman.py --seed=21 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 -decay_rate=0.85 &
CUDA_VISIBLE_DEVICES=3 python3 run_koopman.py --seed=666 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 -decay_rate=0.85 &
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=82 --data_freq="Weekly" --dataset="M4" --data_dir="data/M4/" --train_output_length=10 --test_output_length=13 --input_dim=5 --input_length=45 --hidden_dim=256 --num_layers=5 --latent_dim=64 --learning_rate=0.005 --batch_size=128 --jumps=3 -decay_rate=0.85 &

# M4-Daily. Ensemble is not very helpful on Daily dataset, a single run already achieves SOTA performance. sMAPE: 2.997
CUDA_VISIBLE_DEVICES=1 python3 run_koopman.py --seed=6 --data_freq="Daily" --dataset="M4" --data_dir="data/M4/" --train_output_length=6 --test_output_length=14 --input_dim=3 --input_length=18 --hidden_dim=128 --num_layers=4 --transformer_dim=128 --control_hidden_dim=64 --latent_dim=8 --learning_rate=0.005 --batch_size=256 --jumps=5 -decay_rate=0.85 -num_steps=6 -num_sins=2 &

##########################################################################################
# On Cryptos and Traj datasets, we report the mean and standard deviations of five runs.
##########################################################################################
# Download Traj dataset here: https://drive.google.com/drive/folders/1N_wo1I7G62HglyML5yL4FTEzbfekh4vZ?usp=sharing
# Player Traj. RMSE: 1.163 ± 0.005
CUDA_VISIBLE_DEVICES=2 python3 run_koopman.py --seed=0 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=128 --latent_dim=32 --num_layers=4 --control_num_layers=3 --control_hidden_dim=64 --transformer_dim=64 --transformer_num_layers=3 --batch_size=128 --test_output_length=30 --num_steps=15 --jumps=2 --learning_rate=0.001 --num_sins=3 &
CUDA_VISIBLE_DEVICES=3 python3 run_koopman.py --seed=1 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=128 --latent_dim=32 --num_layers=4 --control_num_layers=3 --control_hidden_dim=64 --transformer_dim=64 --transformer_num_layers=3 --batch_size=128 --test_output_length=30 --num_steps=15 --jumps=2 --learning_rate=0.001 --num_sins=3 &
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=7 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=128 --latent_dim=32 --num_layers=4 --control_num_layers=3 --control_hidden_dim=64 --transformer_dim=64 --transformer_num_layers=3 --batch_size=128 --test_output_length=30 --num_steps=15 --jumps=2 --learning_rate=0.001 --num_sins=3 &
CUDA_VISIBLE_DEVICES=1 python3 run_koopman.py --seed=8 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=128 --latent_dim=32 --num_layers=4 --control_num_layers=3 --control_hidden_dim=64 --transformer_dim=64 --transformer_num_layers=3 --batch_size=128 --test_output_length=30 --num_steps=15 --jumps=2 --learning_rate=0.001 --num_sins=3 &
CUDA_VISIBLE_DEVICES=2 python3 run_koopman.py --seed=9 --dataset="Traj" --data_dir="data/PlayerTraj/" --num_feats=2 --train_output_length=15 --input_dim=3 --input_length=21 --hidden_dim=128 --latent_dim=32 --num_layers=4 --control_num_layers=3 --control_hidden_dim=64 --transformer_dim=64 --transformer_num_layers=3 --batch_size=128 --test_output_length=30 --num_steps=15 --jumps=2 --learning_rate=0.001 --num_sins=3 &

# Cryptos. Weighted RMSE: 0.00691 ± 2.87616e-05
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=162 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=64 --num_layers=5 --latent_dim=16 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=6 --num_steps=7 &
CUDA_VISIBLE_DEVICES=1 python3 run_koopman.py --seed=2 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=64 --num_layers=5 --latent_dim=16 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=6 --num_steps=7 &
CUDA_VISIBLE_DEVICES=2 python3 run_koopman.py --seed=28 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=64 --num_layers=5 --latent_dim=16 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=6 --num_steps=7 &
CUDA_VISIBLE_DEVICES=3 python3 run_koopman.py --seed=3 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=64 --num_layers=5 --latent_dim=16 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=6 --num_steps=7 &
CUDA_VISIBLE_DEVICES=0 python3 run_koopman.py --seed=43 --dataset="Cryptos" --data_dir="data/Cryptos/" --num_feats=8 --train_output_length=14 --test_output_length=15 --input_dim=7 --input_length=63 --hidden_dim=64 --num_layers=5 --latent_dim=16 --transformer_num_layers=3 --transformer_dim=256 --control_num_layers=3 --control_hidden_dim=128 --learning_rate=0.005 --batch_size=512 --jumps=100 --num_sins=6 --num_steps=7 &
wait




