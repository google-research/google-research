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

"""Contrastive RL config."""
import dataclasses
from typing import Any, Optional, Union, Tuple

from acme import specs
from acme.adders import reverb as adders_reverb
import numpy as onp


@dataclasses.dataclass
class ContrastiveConfig:
  """Configuration options for contrastive RL."""

  env_name: str = ''
  max_number_of_steps: int = 1_000_000
  num_actors: int = 4

  # Loss options
  batch_size: int = 256
  actor_learning_rate: float = 3e-4
  learning_rate: float = 3e-4
  reward_scale: float = 1
  discount: float = 0.99
  n_step: int = 1
  # Coefficient applied to the entropy bonus. If None, an adaptative
  # coefficient will be used.
  entropy_coefficient: Optional[float] = None
  target_entropy: float = 0.0
  # Target smoothing coefficient.
  tau: float = 0.005
  hidden_layer_sizes: Tuple[int, Ellipsis] = (256, 256)

  # Replay options
  min_replay_size: int = 10000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  num_parallel_calls: Optional[int] = 4
  samples_per_insert: float = 256
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1
  num_sgd_steps_per_step: int = 64  # Gradient updates to perform per step.

  repr_dim: Union[int, str] = 64  # Size of representation.
  use_random_actor: bool = True  # Initial with uniform random policy.
  repr_norm: bool = False
  use_cpc: bool = False
  local: bool = False  # Whether running locally. Disables eval.
  use_td: bool = False
  twin_q: bool = False
  use_gcbc: bool = False
  use_image_obs: bool = False
  random_goals: float = 0.5
  jit: bool = True
  add_mc_to_td: bool = False
  resample_neg_actions: bool = False
  bc_coef: float = 0.0

  # Parameters that should be overwritten, based on each environment.
  obs_dim: int = -1
  max_episode_steps: int = -1
  start_index: int = 0
  end_index: int = -1


def target_entropy_from_env_spec(
    spec,
    target_entropy_per_dimension = None,
):
  """A heuristic to determine a target entropy.

  If target_entropy_per_dimension is not specified, the target entropy is
  computed as "-num_actions", otherwise it is
  "target_entropy_per_dimension * num_actions".

  Args:
    spec: environment spec
    target_entropy_per_dimension: None or target entropy per action dimension

  Returns:
    target entropy
  """

  def get_num_actions(action_spec):
    """Returns a number of actions in the spec."""
    if isinstance(action_spec, specs.BoundedArray):
      return onp.prod(action_spec.shape, dtype=int)
    elif isinstance(action_spec, tuple):
      return sum(get_num_actions(subspace) for subspace in action_spec)
    else:
      raise ValueError('Unknown action space type.')

  num_actions = get_num_actions(spec.actions)
  if target_entropy_per_dimension is None:
    if not isinstance(spec.actions, specs.BoundedArray) or isinstance(
        spec.actions, specs.DiscreteArray):
      raise ValueError('Only accept BoundedArrays for automatic '
                       f'target_entropy, got: {spec.actions}')
    if not onp.all(spec.actions.minimum == -1.):
      raise ValueError(
          f'Minimum expected to be -1, got: {spec.actions.minimum}')
    if not onp.all(spec.actions.maximum == 1.):
      raise ValueError(
          f'Maximum expected to be 1, got: {spec.actions.maximum}')

    return -num_actions
  else:
    return target_entropy_per_dimension * num_actions
