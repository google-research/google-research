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

import logging
from multiprocessing.pool import ThreadPool
import random
from typing import Any

from tqdm import tqdm

from act.models.preference_model import PROMPT_FACTORY, RejectedSampleModel
from act.data.base_preference_generator import BaseGeneratorABC

logger = logging.getLogger(__name__)

class BaseGenerator(BaseGeneratorABC):

  def __init__(
      self,
      dataset,
      target_label,
      icl_examples,
      preference_model,
      preference_batch_generation,
  ):
    self.dataset = dataset
    self.target_label = target_label
    self.icl_examples = icl_examples
    self.preference_model = preference_model
    self.preference_dataset = (
        self.prepare_batch_dataset()
        if preference_batch_generation
        else self.prepare_streaming_dataset()
    )

  def construct_icl_examples(self, label):
    icl_pool = []
    for instance in self.dataset:
      if instance["dialogue_policy"] == label:
        icl_example = instance["input_text"]
        icl_example += self.preference_model.get_icl_prompt(label)
        icl_example += instance["output_text"]
        icl_pool.append(icl_example)
    return icl_pool

  def create_prompt_input(
      self, instance, icl_pool
  ):
    examples = random.sample(icl_pool, self.icl_examples)
    prompt_input = "\n".join(examples)
    prompt_input = (
        prompt_input
        + "\n"
        + instance["input_text"]
        + self.preference_model.get_icl_prompt(self.target_label)
    )
    return prompt_input

  def prepare_batch_dataset(self, **generation_kwargs):
    icl_pool = self.construct_icl_examples(self.target_label)
    batch_requests = []

    for instance in self.dataset:
      if instance["dialogue_policy"] != self.target_label:
        batch_requests.append(self.create_prompt_input(instance, icl_pool))

    batch_results = self.preference_model.generate_batch(
        inputs=batch_requests,
        generation_kwargs=generation_kwargs,
    )

    instances = dict()
    batch_index = 0
    for i, instance in enumerate(self.dataset):
      if instance["dialogue_policy"] != self.target_label:
        response = batch_results[batch_index]
        response = response.split("?")[0] + "?"
        instance["rejected"] = response
        instances[i] = instance
        batch_index += 1
      else:
        instances[i] = instance

    batch_results_len = len(batch_results)
    assert (
        batch_index == batch_results_len
    ), f"Only {i} of {batch_results_len} batch results used!"

    return list([v for _, v in sorted(instances.items())])

  def prepare_streaming_dataset(self, **generation_kwargs):
    icl_pool = self.construct_icl_examples(self.target_label)
    instances = dict()

    with tqdm(total=len(self.dataset)) as pbar:
      for i, instance in enumerate(self.dataset):
        if instance["dialogue_policy"] != self.target_label:
          prompt_input = self.create_prompt_input(instance, icl_pool)

          while i not in instances:
            try:
              response = self.preference_model.generate(prompt_input,
                                                        **generation_kwargs)
              response = response.split("?")[0] + "?"
              instance["rejected"] = response
              instances[i] = instance
            except Exception as e:
              logging.warning(f"Failed to generate preference {i}, will retry: {e}")
        else:
          instances[i] = instance

        pbar.update(1)

    return list([v for _, v in sorted(instances.items())])
