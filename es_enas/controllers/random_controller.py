# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Random Controller that proposes random topologies."""

import pyglove as pg
from es_enas.controllers import base_controller


class RandomController(base_controller.BaseController):
  """Random Search Controller."""

  def __init__(self, dna_spec, batch_size,
               **kwargs):
    """Initialization. See base class for more details."""
    super().__init__(dna_spec, batch_size)
    del kwargs

  def propose_dna(self):
    """Proposes a topology dna using stored template.

    Args: None.

    Returns:
      dna: A proposed dna.
    """
    return pg.random_dna(self._dna_spec)

  def collect_rewards_and_train(self, reward_vector, dna_list):
    """Collects rewards and sends them to the replay buffer.

    Args:
      reward_vector: list of reward floats.
      dna_list: list of dna's from the proposal function.

    Returns:
      None.
    """

    del reward_vector
    del dna_list
    pass
