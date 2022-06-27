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

"""Aquadem config."""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class AquademConfig:
  """Configuration options for Aquadem."""

  # Number of actions to learn in the multiBC
  num_actions: int = 10

  # Learning rate for the multi BC.
  encoder_learning_rate: float = 3e-4
  encoder_batch_size: int = 256

  # Number of steps to train the multi BC
  encoder_num_steps: int = 50_000
  encoder_eval_every: int = 1_000

  # Scale
  temperature: float = 0.001

  # Ratio of batches from expert demonstrations vs agent interactions sampled
  demonstration_ratio: float = 0.25
  # If the transition comes from the expert, have a minimum reward value
  # which encourages to join the support of the expert
  min_demo_reward: Optional[float] = None
