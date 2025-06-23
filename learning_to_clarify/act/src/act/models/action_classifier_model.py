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

from typing import Union

from act.config.base_config import BaseConfig
from act.config.model.model_config import ModelConfig
from act.models.generative_model import GoogleGenerativeModel
import torch


class ActionClassifierModel(GoogleGenerativeModel):

  def __init__(
      self,
      config,
      model_config,
  ):
    super().__init__(config, model_config)
    self.set_actions()

  def prepend_icl_examples(self, inputs):
    icl_examples = """
    Assistant: {}
    Is the Assistant's response a clarifying question? Yes or No.""".format(
        inputs
    )
    return icl_examples

  def set_actions(self):
    # TODO: This currently only supports two actions. Need to change to support
    # more.
    if hasattr(self.config.action_model_config, 'positive_action'):
      self.positive_action = self.config.action_model_config.positive_action
    else:
      self.positive_action = "CLARIFY"

    if hasattr(self.config.action_model_config, 'negative_action'):
      self.negative_action = self.config.action_model_config.negative_action
    else:
      self.negative_action = "ANSWER"

    if hasattr(self.config.action_model_config, 'positive_response'):
      self.positive_response = self.config.action_model_config.positive_response
    else:
      self.positive_response = "AskClarification"

    if hasattr(self.config.action_model_config, 'negative_response'):
      self.negative_response = self.config.action_model_config.negative_response
    else:
      self.negative_response = "DirectlyAnswerQuestion"

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
      'temperature': 0.1,
      'top_p': 0.95,
      'top_k': 40,
      'max_output_tokens': 3,
      'stop_token': "\n",
      'candidate_count': 1
    }
    stop_token = self.generation_kwargs.pop('stop_token', "\n")
    if stop_token:
      self.generation_kwargs['stop_sequences'] = stop_token
    else:
      self.generation_kwargs['stop_sequences'] = []

  def generate(self, inputs, **generation_kwargs):
    response = super().generate(self.prepend_icl_examples(inputs),
                                **self.generation_kwargs)
    return response

  def classify(self, result):
    if "yes" in result.lower():
      return self.positive_response
    else:
      return self.negative_response

  def mapper(self, policy):
    if (policy == self.positive_action) or (self.positive_action in policy):
      return "AskClarification"
    elif (policy == self.negative_action) or (self.negative_action in policy):
      return "DirectlyAnswerQuestion"
    else:
      print("Policy was odd: ", policy)
      return policy
