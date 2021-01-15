# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Base class for all controllers in ES-ENAS."""

import abc
from typing import List

import pyglove as pg


class BaseController(abc.ABC):
  """Base class for all controllers in ES-ENAS."""

  def __init__(self, dna_spec, batch_size):
    """Initialization.

    Args:
      dna_spec: A search space definition for the controller to use.
      batch_size: Number suggestions in a current iteration.

    Returns:
      None.
    """
    self._dna_spec = dna_spec
    self._batch_size = batch_size
    self._controller = None

  def propose_dna(self):
    """Proposes a topology dna using stored template.

    Args: None.

    Returns:
      dna: A proposed dna.
    """
    return self._controller.propose()

  def collect_rewards_and_train(self, reward_vector,
                                dna_list):
    """Collects rewards to update the controller.

    Args:
      reward_vector: list of reward floats.
      dna_list: list of dna's from the proposal function.

    Returns:
      None.
    """

    for i, dna in enumerate(dna_list):
      dna.reward = reward_vector[i]
      self._controller.feedback(dna, dna.reward)
