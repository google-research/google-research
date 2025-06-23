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

import os
import time
from typing import Optional, Union

from act.config.base_config import BaseConfig
from act.config.model.model_config import ModelConfig
from act.models.generative_model import GoogleGenerativeModel
from transformers import PreTrainedTokenizerBase


PROMPT_FACTORY = {
    'CLARIFY': (
        "\n[Instruction]\nThe User's question is ambiguous so the System"
        ' should ask a clarifying question.\nAssistant:'
    )
}


class RejectedSampleModel(GoogleGenerativeModel):

  def __init__(self, config, model_config):
    super().__init__(config, model_config)
    self.label = config.training_config.target_label
    _ = self.get_icl_prompt(self.label)

  def get_icl_prompt(self, label = None):
    if not label:
      label = self.label
    if label in PROMPT_FACTORY:
      return PROMPT_FACTORY[label]
    else:
      raise ValueError(f"Label {label} is not supported.")

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
      'temperature': 0.1,
      'max_output_tokens': 64,
      'stop_token': "\n",
      'candidate_count': 1
    }
    stop_token = self.generation_kwargs.pop('stop_token', "\n")
    if stop_token:
      self.generation_kwargs['stop_sequences'] = stop_token
    else:
      self.generation_kwargs['stop_sequences'] = []
