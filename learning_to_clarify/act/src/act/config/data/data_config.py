# coding=utf-8
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

import dataclasses
from typing import Optional


@dataclasses.dataclass
class DataConfig:
  """Configuration for a dataset."""
  train_path: Optional[str] = None
  validation_path: Optional[str] = None
  train_preference_filename: str = 'train_preference.jsonl'
  validation_preference_filename: str = 'validation_preference.jsonl'
  truncation_side: Optional[str] = 'left'
  has_context_metadata: bool = False
  eval_data: Optional[str] = None
  eval_result_output_path: Optional[str] = None
  eval_sample_output_path: Optional[str] = None
  eval_max_input_length: int = 4096
  preference_batch_generation: Optional[bool] = False
