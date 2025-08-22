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

# --model_name "lvwerra/gpt2-imdb" \
# --filter_negative \
# --multi_gpu 
accelerate launch \
    --main_process_port 29500 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no sentiment-refactored.py \
        --model_name "lvwerra/gpt2-imdb" \
        --log_with "wandb" \
        --learning_rate 1.41e-5 \
        --batch_size 128 \
        --mini_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --target_kl 6.0 \
        --kl_penalty "kl" \
        --ppo_epochs 4 \
        --tracker_project_name "trl-fgrlaif-sentiment" \
        --epoch 1;