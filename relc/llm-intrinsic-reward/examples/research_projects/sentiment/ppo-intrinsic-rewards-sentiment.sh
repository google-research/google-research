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


accelerate launch \
    --main_process_port 9876 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --multi_gpu \
    --dynamo_backend no ppo-intrinsic-rewards-sentiment.py \
        --use_instric_reward \
        --positive_reward_value 0.8 \
        --negative_reward_value -0.8 \
        --intrinsic_reward_threshold 10 \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_sentiment_3shot_v2.txt \
        --epochs 2 \
        --query_dataset "imdb" \
        --min_new_tokens 15 \
        --max_new_tokens 20 \
        --num_shared_layers 20 \
        --save_freq 50 \
        --model_save_path $SCRATCH/trl-fgrlaif/sentiment/ppo-intrinsic-sentiment-gpt2large-epoch2-bs16-mbs16-freeze20-pos0.8-neg0.8 \
        --ppo_config.model_name "gpt2-large" \
        --ppo_config.log_with "wandb" \
        --ppo_config.learning_rate 1.41e-5 \
        --ppo_config.batch_size 16 \
        --ppo_config.mini_batch_size 16 \
        --ppo_config.gradient_accumulation_steps 1 \
        --ppo_config.target_kl 6.0 \
        --ppo_config.kl_penalty "kl" \
        --ppo_config.ppo_epochs 4 \
        --ppo_config.tracker_project_name "trl-fgrlaif-sentiment-gpt2large";