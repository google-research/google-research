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


source activate PPOIntrinsic

CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9987 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo-intrinsic-rewards-toxicity.py \
        --use_intrinsic_reward \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_toxicity_3shot_v2.txt \
        --positive_reward_value 0.1 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 1.0 \
        --perspective_api $PERSPECTIVE_API_KEY \
        --model_name gpt2-large \
        --log_with "wandb" \
        --model_save_path $CKPT_DIR/toxicity/ppo_intrinsic_epoch5_threshold1.0_bs64_mbs64 \
        --epochs 5 \
        --prompt_toxicity_level 0.6 \
        --output_min_length 10 \
        --output_max_length 14 \
        --learning_rate 1.41e-5 \
        --batch_size 64 \
        --mini_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4 \
        --tracker_project_name "trl-fgrlaif-toxicity-perspective";