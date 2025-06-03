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
    --main_process_port 8765 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo_summ_rouge.py \
        --use_instric_reward \
        --positive_reward_value 0.5 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 100;
        --use_score_scaling \
        --num_shared_layers 12 \
        --model_name $CKPT_DIR/summarization/sft_gpt2-medium/checkpoint-1000 \
        --model_save_path $CKPT_DIR/summarization/ppo-intrinsic-rouge \
        --epochs 5 \
        --output_min_length 30 \
        --output_max_length 50 \
        --learning_rate 1.41e-5 \
        --batch_size 8 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4 \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_summ_3shot_rouge_v2.txt;