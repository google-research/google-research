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

import abc
from collections import defaultdict
import random
from typing import Any, Optional, Union

from datasets import arrow_dataset
from datasets import Dataset

from act.models.preference_model import RejectedSampleModel
from act.data.constants import (
    _MODEL_ROLE,
    _PREFERENCE_SYSTEM_INSTRUCTION,
    _SYSTEM_ROLE,
    _USER_ROLE,
)
from act.data.preference_generator import BaseGenerator


class ACTDatasetABC(abc.ABC):

  def __init__(
      self,
      dataset,
      target_label,
      icl_examples,
      preference_model,
      class_balance,
      system_instruction,
      system_role,
      user_role,
      assistant_role,
  ):
    raise NotImplementedError

  def build_prompt_datasets(self):
    raise NotImplementedError

  def rebalance_dataset(
      self, data
  ):
    raise NotImplementedError

  def prepare_datasets(self):
    raise NotImplementedError


class ACTDataset(ACTDatasetABC):

  def __init__(
      self,
      dataset,
      target_label,
      icl_examples,
      rejected_sample_model,
      class_balance = None,
      system_instruction = _PREFERENCE_SYSTEM_INSTRUCTION,
      system_role = _SYSTEM_ROLE,
      user_role = _USER_ROLE,
      assistant_role = _MODEL_ROLE,
      is_preference = False,
      has_context_metadata = False,
      preference_batch_generation = False,
  ):
    self.data = dataset
    self.system_instruction = system_instruction
    self.target_label = target_label
    self.class_balance = class_balance
    self.system_role = system_role
    self.user_role = user_role
    self.assistant_role = assistant_role
    self.is_preference = is_preference
    self.rejected_sample_model = rejected_sample_model
    self.icl_examples = icl_examples
    self.has_context_metadata = has_context_metadata
    self.preference_batch_generation = preference_batch_generation
    if self.is_preference and (len(self.data) == 1):
      self.data = self.data[0]
    if not self.is_preference:
      assert (
          self.rejected_sample_model
      ), "Preference model is needed for non-preference dataset."

  def rebalance_dataset(
      self, data
  ):
    if self.class_balance is None:
      return data

    pool = defaultdict(list)
    new_data = []
    for k in self.class_balance:
      pool[k] = self.rebalance_class(k, data)
      new_data.extend(pool[k])

    random.shuffle(new_data)
    return new_data

  def rebalance_class(
      self, class_label, data
  ):
    pool = []
    for instance in data:
      if instance["dialogue_policy"] == class_label:
        pool.append(instance)

    pool = random.choices(pool, k=self.class_balance[class_label] * len(pool))
    return pool

  def process_context(self, context):
    if self.has_context_metadata:
      prompt_header = "\n{}\n[conversation]\n".format(context)
    else:
      prompt_header = "[context]\n{}\n[conversation]\n".format(context)
    return prompt_header

  def assess_clarification(self, message):
    return message["requires_clarification"]

  def get_next_turn(
      self, i, messages, clarify
  ):
    if i + 1 >= len(messages):
      return "", None
    if not clarify:
      output_text = "Assistant: {}\n".format(messages[i + 1]["content"])
      answer = None
    else:
      output_text = "Assistant: {}\n".format(messages[i + 1]["content"])
      if i + 2 < len(messages):
        mess_next = messages[i + 2]
        temp = "User: {}\n".format(mess_next["content"].replace("\n", " "))
        output_text += temp
        if i + 3 < len(messages):
          turn_after = messages[i + 3]
          temp_after = "Assistant: {}\n".format(
              turn_after["content"].replace("\n", " ")
          )
          output_text += temp_after
          answer = temp_after
        else:
          answer = "This question was not answered by the Assistant."
      else:
        answer = "This question was not answered by the Assistant."

    return output_text, answer

  def build_prompt_dataset(self, data):
    instances = []
    for sample in data:
      input_text = "{}\n".format(self.system_instruction)
      output_text = ""
      for i, message in enumerate(sample["messages"]):
        if message["role"] == self.system_role:
          input_text += self.process_context(message["content"])
        elif message["role"] == self.user_role:
          input_text += "User: {}\n".format(
              message["content"].replace("\n", " ")
          )

          if self.assess_clarification(message):
            output_text, answer = self.get_next_turn(
                i, sample["messages"], clarify=True
            )
            dialogue_policy = "CLARIFY"
            gold_trajectory = input_text + output_text
            gold_target = answer
            chosen_policy = "CLARIFY"
            rejected_policy = "ANSWER"
            chosen_response = output_text
            rejected_response = gold_target
          else:
            output_text, answer = self.get_next_turn(
                i, sample["messages"], clarify=False
            )
            assert answer is None, "Something went wrong here."
            dialogue_policy = "ANSWER"
            gold_trajectory = input_text + output_text
            gold_target = output_text
            chosen_policy = "ANSWER"
            rejected_policy = "CLARIFY"
            chosen_response = output_text
            rejected_response = "PLACEHOLDER"
          instance = {
              "input_text": input_text,
              "output_text": output_text,
              "dialogue_policy": dialogue_policy,
              "gold_target": gold_target,
              "gold_trajectory": gold_trajectory,
              "chosen_policy": chosen_policy,
              "rejected_policy": rejected_policy,
              "chosen": chosen_response,
              "rejected": rejected_response,
          }
          instances.append(instance)
          input_text = input_text + output_text
    return instances

  def build_prompt_datasets(self):
    prompt_datasets = self.build_prompt_dataset(self.data)
    return prompt_datasets

  def build_preference_datasets(self, data):
    if self.is_preference:
      return data
    generator = BaseGenerator(
        data,
        self.target_label,
        self.icl_examples,
        self.rejected_sample_model,
        self.preference_batch_generation,
    )
    dat = generator.preference_dataset
    return dat

  def prepare_datasets(self):
    if not self.is_preference:
      prompt_datasets = self.build_prompt_datasets()
      prompt_datasets = self.build_preference_datasets(prompt_datasets)

      prompt_datasets = self.rebalance_dataset(prompt_datasets)

      final_dataset = defaultdict(list)
      for instance in prompt_datasets:
        for instance_k in instance.keys():
          final_dataset[instance_k].append(str(instance[instance_k]))
      final_dataset = Dataset.from_dict(final_dataset)
    else:
      final_dataset = Dataset.from_list(self.data)
    return final_dataset
