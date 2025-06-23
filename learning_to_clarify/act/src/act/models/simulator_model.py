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
from act.data.constants import _MODEL_ROLE, _USER_ROLE
from act.data.fewshot_examples import pacific_user_simulator_examples
from act.models.generative_model import GoogleGenerativeModel
from transformers import PreTrainedTokenizerBase


class SimulatorModel(GoogleGenerativeModel):

  def __init__(
      self,
      config,
      model_config,
      user_role = _USER_ROLE,
      assistant_role = _MODEL_ROLE,
  ):
    super().__init__(config, model_config)
    self.user_role = user_role
    self.assistant_role = assistant_role

  def simulation_prompt(self, inputs, intent):
    sim_prompt = (
        "The following is a conversation between a {} and an {}. The {} is"
        " asking some questions. {} Complete the conversation in a coherent"
        " way.[context]\n:{}".format(
            self.user_role, self.assistant_role, self.user_role, intent, inputs
        )
    )
    return sim_prompt

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
        "temperature": 0.1,
        "top_p": 0.92,
        "top_k": 40,
        "max_output_tokens": 64,
        "stop_token": "\n",
        "candidate_count": 1,
    }
    stop_token = self.generation_kwargs.pop("stop_token", "\n")
    if stop_token:
      self.generation_kwargs["stop_sequences"] = stop_token
    else:
      self.generation_kwargs["stop_sequences"] = []


class PACIFICSimulator(SimulatorModel):

  def simulation_prompt(self, inputs, intent):
    prompt_string = ""
    for i, example in enumerate(pacific_user_simulator_examples):
      prompt_string += "[Example {}]\n{}\n".format(i + 1, example)
    prompt_string += (
        "[Example {}] The following is a conversation between a {} and an {}."
        " The {} is asking some questions. {} Complete the conversation in a"
        " coherent way. Context: {}".format(
            len(pacific_user_simulator_examples) + 1,
            self.user_role,
            self.assistant_role,
            self.user_role,
            intent,
            "\n".join(inputs),
        )
    )
    if "?" in inputs[-1]:
      prompt_string += (
          "\nThe {} asked a clarification question, so the {} should clarify"
          " what they are asking.".format(self.assistant_role, self.user_role)
      )
    else:
      prompt_string += (
          "\nThe {} answered the {}'s question, so the {} can proceed with"
          " asking another question.".format(self.assistant_role,
                                             self.user_role, self.user_role)
      )
    prompt_string += "{}:".format(self.user_role)
    return prompt_string
