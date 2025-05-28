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

"""Concept PPO config."""
import dataclasses
from typing import Callable, Union

from acme.adders import reverb as adders_reverb
import numpy as np


@dataclasses.dataclass
class ConceptPPOConfig:
  """Configuration options for Concept PPO.

  Attributes:
    unroll_length: Length of sequences added to the replay buffer.
    num_minibatches: The number of minibatches to split an epoch into.
      i.e. minibatch size = batch_size / num_minibatches.
    num_epochs: How many times to loop over the set of minibatches.
    batch_size: int
    clip_value: bool
    replay_table_name: Replay table name.
    ppo_clipping_epsilon: float
    gae_lambda: float
    discount: float
    learning_rate:
    adam_epsilon: float
    entropy_cost: float
    value_cost: float
    concept_cost: float
    max_abs_reward: float
    max_gradient_norm: float
    variable_update_period: int
  """
  unroll_length: int = 16
  num_minibatches: int = 32
  num_epochs: int = 5
  batch_size: int = 128
  clip_value: bool = False
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  ppo_clipping_epsilon: float = 0.2
  gae_lambda: float = 0.95
  discount: float = 0.99
  learning_rate: Union[float, Callable[[int], float]] = 3e-4
  adam_epsilon: float = 1e-5
  entropy_cost: float = 0.01
  value_cost: float = 1.
  concept_cost: float = 0.1
  max_abs_reward: float = np.inf
  max_gradient_norm: float = 0.5
  variable_update_period: int = 1
