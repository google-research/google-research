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

        # --use_intrinsic_rewad \
        # --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_summ_3shot_rlhf.txt \
        # --positive_reward_value 0.1 \
        # --negative_reward_value -0.1 \
        # --intrinsic_reward_threshold 10;
        # --use_score_scaling \
accelerate launch \
    --main_process_port 8989 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo_summ_rlhf.py \
        --log_with "wandb" \
        --num_shared_layers 12 \
        --model_name $SCRATCH/trlx/sft_gpt2-medium_bs64_epoch5 \
        --reward_model_ckpt_path $SCRATCH/trlx/reward_model_official/rm_checkpoint/pytorch_model.bin \
        --model_save_path $SCRATCH/trl-fgrlaif/summarization/ppo-baseline-pref \
        --epochs 3 \
        --output_min_length 30 \
        --output_max_length 50 \
        --learning_rate 1.41e-5 \
        --batch_size 8 \
        --mini_batch_size 8 \
        --rw_batch_size 8 \
        --length_penalty 0.5 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4 \
        --tracker_project_name "trl-fgrlaif-summ-rlhf";
