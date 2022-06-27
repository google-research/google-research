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

"""Config object for SNR."""
import dataclasses
import typing

import gin


@gin.configurable
@dataclasses.dataclass
class SNRKwargs:
  """For cleaner passing around of SNR keyword arguments."""
  snr_mode: str = 'ntk'
  snr_loss_type: str = 'full'
  snr_num_centroids: int = 1024
  snr_kmeans_iters: int = 100
  use_log_space_matrix: bool = True
  snr_matrix_tau: float = 0.01
  use_target_for_phi_prime: bool = False


@gin.configurable
@dataclasses.dataclass
class SNRConfig:
  """Configuration options for SNR.

  For gin bindings on SNRKwargs to get applied to this dataclass\'s snr_kwargs,
  you should not forget to add the following gin binding when running:
  --gin_bindings='snr.config.SNRConfig.snr_kwargs=@snr.config.SNRKwargs()'
  """
  policy_lr: float = 3e-5
  # policy_lr: float = 1e-4
  q_lr: float = 3e-4

  num_bc_iters: int = 50_000
  entropy_coefficient: typing.Optional[float] = None
  target_entropy: float = 0.0
  num_sgd_steps_per_step: int = 1

  actor_network_hidden_sizes: typing.Tuple = (256, 256, 256)
  critic_network_hidden_sizes: typing.Tuple = (256, 256, 256)
  num_critics: int = 1

  use_snr_in_bc_iters: bool = False
  snr_applied_to: str = 'policy'
  snr_alpha: float = 1.0
  snr_kwargs: SNRKwargs = SNRKwargs()
