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

PROMPTS_FILE=$HOME/ppo-intrinsic-reward/dexpert_dataset/prompts/nontoxic_prompts-10k.jsonl
OUTPUT_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/toxicity/outputs/llama2_prompted_2shotv2_sample5000.jsonl
echo $OUTPUT_FILE

# --sample_size 100 \
CUDA_VISIBLE_DEVICES=0 python llama2_vllm.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --download_dir $HOME/huggingface/meta-llama/Llama-2-7b-chat-hf \
    --prompts_file $PROMPTS_FILE \
    --output_file $OUTPUT_FILE \
    --sample_size 5000 \
    --top_p 0.9 \
    --max_new_tokens 20 \
    --num_returns 25 \
    --temperature 1.0 \
    --batch_size 10 \
    --prompt_file $HOME/ppo-intrinsic-reward/prompts/llama2_prompt_2shot_v2.txt;