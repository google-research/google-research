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

import json
import logging
import os
import time
from typing import Optional, Union
import uuid

from act.config.base_config import BaseConfig
from act.config.model.model_config import ModelConfig
from act.models.base_model import BaseModel
import torch
from google.generativeai import GenerativeModel
from google.generativeai.types import (
      HarmBlockThreshold,
      HarmCategory,
  )


logger = logging.getLogger(__name__)


class GoogleGenerativeModel(BaseModel):

  def __init__(self, config: BaseConfig, model_config: ModelConfig):
    super().__init__(config)
    self.model_config = model_config
    self.model = GoogleGenerativeModel.load_generative_model(
        self.model_config.model_id
    )
    self.construct_generation_kwargs()

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
      'temperature': 0.2,
      'top_p': 0.92,
      'top_k': 40,
      'max_output_tokens': 64,
      'stop_token': "\n",
      'candidate_count': 1
    }
    stop_token = self.generation_kwargs.pop('stop_token', "\n")
    if stop_token:
      self.generation_kwargs['stop_sequences'] = stop_token
    else:
      self.generation_kwargs['stop_sequences'] = []

  def generate_batch(self, inputs: list[str], **generation_kwargs):
      raise NotImplementedError("Batch prediction not yet supported.")

  def generate(self, inputs: Union[str, torch.Tensor], **generation_kwargs):
    sleep_attempts = 0
    sleep_time_seconds = 2
    while sleep_attempts <= 5:
      try:
        response = self.model.generate_content(
            contents=inputs,
            generation_config=generation_kwargs,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
                    HarmBlockThreshold.BLOCK_NONE
                ),
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
                    HarmBlockThreshold.BLOCK_NONE
                ),
                HarmCategory.HARM_CATEGORY_HARASSMENT: (
                    HarmBlockThreshold.BLOCK_NONE
                ),
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
                    HarmBlockThreshold.BLOCK_NONE
                ),
            },
        )
        response = response.candidates[0].content
        if len(response.parts) > 0:
          return response.parts[0].text
        else:
          return ''
      except Exception as re:
        print(f"Exception occurred while processing property: {re}")
        sleep_attempts += 1
        time.sleep(sleep_time_seconds)
        sleep_time_seconds *= 2

    return 'Error occurred while processing the request'

  @staticmethod
  def load_generative_model(model_name):
    return GenerativeModel(model_name)
