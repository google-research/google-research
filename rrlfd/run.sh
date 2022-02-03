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

pip install --upgrade pip

pip install -r requirements.txt

python -m bc.collect_demos --logdir DEMOS_PATH --num_episodes 30

python -m rrlfd.bc.train \
--ckpt_dir=BC_EXP_DIR \
--binary_grip_action \
--early_closing \
--grip_action_from_state \
--action_norm=zeromean_unitvar \
--signals_norm=zeromean_unitvar \
--eval_seed=5000 \
--increment_eval_seed \
--visible_state=robot \
--max_demos_to_load=30 \
--noval_full_episodes \
--eval_task=Pick \
--demos_file=DEMOS_PATH/Pick/s0_e30.pkl

python -m rrlfd.residual.train \
--domain=mime \
--task=Pick \
--num_episodes=300000 \
--binary_grip_action \
--action_norm=zeromean_unitvar \
--signals_norm=zeromean_unitvar \
--noval_full_episodes \
--image_size=240 \
--rl_batch_size=256 \
--eval_seed=5000 \
--increment_eval_seed \
--critic_vmin=0 \
--critic_vmax=1 \
--network=resnet18_narrow32 \
--residual_action_norm=centered \
--seed=0 \
--policy_init_std=0.1 \
--policy_lr=0.00033 \
--critic_lr=0.00033 \
--max_demos_to_load=30 \
--bc_ckpt_to_load=BC_EXP_DIR/ckpt \
--original_demos_path=DEMOS_PATH/Pick/s0_e30.pkl \
--logdir=RL_EXP_DIR

