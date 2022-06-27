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

"""Config object for MSG."""
import gin
import dataclasses
import typing

@gin.configurable
@dataclasses.dataclass
class MSGConfig:
  """Configuration options for MSG."""
  policy_lr: float = 3e-5
  # policy_lr: float = 1e-4
  q_lr: float = 3e-4

  ensemble_method: str = 'deep_ensembles'
  mimo_using_adamw: bool = False
  mimo_using_obs_tile: bool = False
  mimo_using_act_tile: bool = False

  use_double_q: bool = False
  ensemble_size: int = 16
  beta: float = -1.0
  td_target_method: str = 'independent'

  perform_sarsa_q_eval: bool = False
  num_q_repr_pretrain_iters: int = 0
  pretrain_temp: float = 1.
  use_sass: bool = False
  num_bc_iters: int = 50_000
  use_random_weighting_in_critic_loss: bool = True
  use_ema_target_critic_params: bool = True
  entropy_coefficient: typing.Optional[float] = None
  target_entropy: float = 0.0
  use_entropy_regularization: bool = True
  num_sgd_steps_per_step: int = 1

  actor_network_hidden_sizes: typing.Tuple = (256, 256)
  critic_network_hidden_sizes: typing.Tuple = (256, 256)
  networks_init_type: str = 'glorot_also_dist'
  critic_random_init: bool = False
  critic_init_scale: float = 1.0 # I dont think it is used anymore

  behavior_regularization_type: str = 'none'
  behavior_regularization_alpha: float = 1.0
  num_cql_actions: int = 2 # if using cql regularization type

  eval_with_q_filter: bool = False
  num_eval_samples: int = 10

  rem_mode: bool = False

  use_img_encoder: bool = False
  img_encoder_params_ckpt_path: str = ''
  img_encoder_fn: typing.Optional[typing.Callable] = None
