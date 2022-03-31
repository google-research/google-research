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

"""Config object for CQL."""
import dataclasses
import typing
import gin
from jrl.agents.snr.config import SNRKwargs


@gin.configurable
@dataclasses.dataclass
class CQLConfig:
  """Configuration options for CQL.

  For gin bindings on SNRKwargs to get applied to this dataclass\'s snr_kwargs,
  you should not forget to add the following gin binding when running:
  --gin_bindings='snr.config.SNRConfig.snr_kwargs=@snr.config.SNRKwargs()'"""
  policy_lr: float = 3e-5
  q_lr: float = 3e-4

  num_bc_iters: int = 50_000

  cql_alpha: float = 5.0
  num_importance_acts: int = 10

  target_entropy: float = 0.0
  num_sgd_steps_per_step: int = 1

  actor_network_hidden_sizes: typing.Tuple = (256, 256)
  critic_network_hidden_sizes: typing.Tuple = (256, 256, 256)

  num_critics: int = 2
  tau: float = 0.005

  eval_with_q_filter: bool = False
  num_eval_samples: int = 10

  snr_alpha: float = 0.0
  snr_kwargs: SNRKwargs = SNRKwargs()
