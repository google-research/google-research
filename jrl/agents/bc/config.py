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

"""Config object for BC."""
import dataclasses
import typing
import gin
from jrl.agents.bc import networks as bc_networks


@gin.configurable
@dataclasses.dataclass
class BCConfig:
  """Configuration options for BC."""
  build_actor_fn: typing.Callable = bc_networks.build_standard_actor_fn

  use_img_encoder: bool = False
  img_encoder_params_ckpt_path: str = ''
  img_encoder_fn: typing.Optional[typing.Callable] = None

  policy_lr: float = 1e-4

  loss_type: str = 'MLE'
  regularize_entropy: bool = False
  entropy_regularization_weight: float = 1.0

  num_sgd_steps_per_step: int = 1
