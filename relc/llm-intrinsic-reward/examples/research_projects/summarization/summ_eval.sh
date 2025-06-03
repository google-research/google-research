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

MODEL=$SCRATCH/trlx/sft_gpt2-medium/checkpoint-1000
echo $MODEL

accelerate launch \
    --main_process_port 8765 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no summ_eval.py \
        --model_name $MODEL \
        --save_path eval_results/sft_gpt2-medium_bs64_step1000.csv \
        --num_samples_to_eval 6000 \
        --reward_model_ckpt_path $SCRATCH/trlx/reward_model_official/rm_checkpoint/pytorch_model.bin \
        --output_max_length 50 \
        --batch_size 20 \
        --rw_batch_size 20 \
        --reward_model_device 0;