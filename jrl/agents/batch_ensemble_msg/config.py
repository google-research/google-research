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

"""Config object for Batch Ensemble MSG."""
import gin
import dataclasses
import typing


@gin.configurable
@dataclasses.dataclass
class BatchEnsembleMSGConfig:
  """Configuration options for Batch Ensemble MSG."""
  policy_lr: float = 3e-5
  q_lr: float = 3e-4

  use_double_q: bool = True
  ensemble_size: int = 16
  beta: float = -1.0

  behavior_regularization_type: str = 'none'
  behavior_regularization_alpha: float = 1.0

  num_bc_iters: int = 50_000
  # num_bc_iters: int = 200_000

  use_random_weighting_in_critic_loss: bool = True
  target_entropy: float = 0.0

  actor_hidden_sizes: typing.Tuple = (256, 256)
  # actor_hidden_sizes: typing.Tuple = (300, 200)
  # q_hidden_sizes: typing.Tuple = (1024, 1024)
  # q_hidden_sizes: typing.Tuple = (400, 300)
  q_hidden_sizes: typing.Tuple = (256, 256)
  # q_hidden_sizes: typing.Tuple = (2048, 2048)
  # actor_hidden_sizes: typing.Tuple = (256, 256, 256)
  # q_hidden_sizes: typing.Tuple = (256, 256, 256)

  num_sgd_steps_per_step: int = 1
