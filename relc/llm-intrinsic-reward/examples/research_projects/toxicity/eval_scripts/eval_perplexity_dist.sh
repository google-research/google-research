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

GENS_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/toxicity/outputs/ppo_self-intrinsic_llama2_bs32_mbs16_sample2k_step400.jsonl
OUTPUT_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/toxicity/outputs/ppo_self-intrinsic_llama2_bs32_mbs16_sample2k_step400_eval.txt
echo $OUTPUT_FILE

CUDA_VISIBLE_DEVICES=0 python eval_perplexity_dist.py \
    --generations_file $GENS_FILE \
    --output_file $OUTPUT_FILE;