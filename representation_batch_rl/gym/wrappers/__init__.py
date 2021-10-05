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

"""A collection of gym wrappers."""

from representation_batch_rl.gym.wrappers.absorbing_wrapper import AbsorbingWrapper
from representation_batch_rl.gym.wrappers.frame_stack_wrapper import FrameStackWrapper
from representation_batch_rl.gym.wrappers.frame_stack_wrapper import stack_expert_frames
from representation_batch_rl.gym.wrappers.normalize_action_wrapper import check_and_normalize_box_actions
from representation_batch_rl.gym.wrappers.normalize_action_wrapper import NormalizeBoxActionWrapper
from representation_batch_rl.gym.wrappers.normalize_state_wrapper import NormalizeStateWrapper
from representation_batch_rl.gym.wrappers.stochastic_wrappers import maybe_make_stochastic


def create_il_env(env_name, num_stack_frames, seed, shift, scale):
  """Create a gym environment for imitation learning.

  Args:
    env_name: an environment name.
    num_stack_frames: a number of consequtive observations to stack.
    seed: a random seed.
    shift: a numpy vector to shift observations.
    scale: a numpy vector to scale observations.

  Returns:
    An initialized gym environment.
  """
  env = maybe_make_stochastic(env_name)
  env = check_and_normalize_box_actions(env)
  env.seed(seed)

  if num_stack_frames != 1:
    env = FrameStackWrapper(env, num_stack_frames)

  if shift is not None:
    env = NormalizeStateWrapper(env, shift=shift, scale=scale)

  return AbsorbingWrapper(env)
