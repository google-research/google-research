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



# Generate the synthetic data for BLEURT finetuning.
python create_synthetic_negatives.py \
  --input_path=example.txt \
  --min_num_deletions=1 \
  --max_num_deletions=7 \
  --num_deletion_negatives=3 \
  --num_repetition_negatives=3 \
  --min_num_repetitions=1 \
  --max_num_repetitions=7 \
  --num_flip_negatives=2 \
  --num_random_negatives=1 \
  --num_digit_negatives=1 \
  --dev_frac=0.1 \
  --output_dir="" \
  --use_source_as_reference=True \
  --upsampling=True
