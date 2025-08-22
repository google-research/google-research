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

DATA_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/toxicity/outputs/llama2_prompted_0shotv2_sample5000.jsonl
OUTPUT_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/toxicity/outputs/llama2_prompted_0shotv2_sample5000_API.jsonl

python eval_toxicity_perspective.py \
    --data_file $DATA_FILE \
    --output_file $OUTPUT_FILE \
    --sample_size 5000 \
    --num_thread 8 \
    --save_scores;