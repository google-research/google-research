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

import torch

from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

  def __init__(self, stops=[], encounters=1, starting_idx=0):
    super().__init__()
    self.stops = stops
    self.ENCOUNTERS = encounters
    self.starting_idx = starting_idx

  def __call__(self, input_ids, scores):
    stop_count = 0
    input_ids = input_ids.squeeze()[self.starting_idx :]
    for stop in self.stops:
      stop_count = (stop == input_ids).sum().item()

    if stop_count >= self.ENCOUNTERS:
      return True
    return False
