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


MODEL_NAME_OR_PATH=$SCRATCH/trl-fgrlaif/sentiment/ppo-baseline-sentiment-gpt2large-epoch2-bs16-mbs16-freeze20/step_100
# MODEL_NAME_OR_PATH=$SCRATCH/trl-fgrlaif/sentiment/ppo-intrinsic-sentiment-gpt2large-epoch2-bs16-mbs16-freeze20-pos0.5-neg0.5/step_100
PROMPT_PATH=$SCRATCH/DExperts/prompts/sentiment_prompts-10k/neutral_prompts.jsonl
# PROMPT_PATH=$SCRATCH/DExperts/prompts/sentiment_prompts-10k/negative_prompts.jsonl
OUTPUT_FILE=$HOME/ppo-intrinsic-reward/examples/research_projects/sentiment/outputs/ppo_neu_prompts_step_100.jsonl


python eval_sentiment_sst.py \
    $MODEL_NAME_OR_PATH \
    $PROMPT_PATH \
    --output_file $OUTPUT_FILE \
    --num_return 25 \
    --classifier_path $SCRATCH/huggingface/distilbert-base-uncased-finetuned-sst-2-english;