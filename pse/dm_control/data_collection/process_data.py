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

# Lint as: python3
r"""Data processing utilities."""

import numpy as np
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

from pse.dm_control import common_flags  # pylint: disable=unused-import
from pse.dm_control.utils import helper_utils

GAMMA = 0.99


def _get_episode(args, start_pos):
  new_args = {}
  for k, v in args.items():
    if k != 'policy_info':
      if k == 'observation':
        new_args[k] = {'pixels': None}
        new_args[k]['pixels'] = v['pixels'][start_pos::2]
      else:
        new_args[k] = v[start_pos::2]
    else:
      new_args[k] = v
  return trajectory.Trajectory(**new_args)


def get_trajs(episode):
  args = {field: getattr(episode, field) for field in episode._fields}
  return _get_episode(args, 0), _get_episode(args, 1)


def get_batched_spec(traj_arr_spec, max_episode_len):
  new_spec = {}
  for f in traj_arr_spec._fields:
    val = getattr(traj_arr_spec, f)
    if isinstance(val, dict) and 'pixels' in val:
      val, new_val = val['pixels'], {}
      new_val['pixels'] = val.replace(shape=[max_episode_len] + list(val.shape))
    elif isinstance(val, tensor_spec.array_spec.ArraySpec):
      new_val = val.replace(shape=[max_episode_len] + list(val.shape))
    elif not val:
      new_val = val
    new_spec[f] = new_val
  traj_arr_spec_with_time_dim = trajectory.Trajectory(**new_spec)
  return traj_arr_spec_with_time_dim


def get_episode_spec(traj_spec, max_episode_len):
  traj_arr_spec = tensor_spec.to_nest_array_spec(traj_spec)
  traj_batch_arr_spec = get_batched_spec(traj_arr_spec, max_episode_len)
  observation_spec = traj_batch_arr_spec.observation['pixels']
  metric_spec = array_spec.BoundedArraySpec(
      shape=(max_episode_len, max_episode_len), dtype=np.float32, minimum=0.)
  return tensor_spec.from_spec(
      (observation_spec, observation_spec, metric_spec))


def _get_pixels(episode):
  if isinstance(episode, list):
    return np.stack([x.observation['pixels'] for x in episode], axis=0)
  else:
    return episode.observation['pixels']


def process_episode(episode1, episode2, gamma):
  obs1, obs2 = _get_pixels(episode1), _get_pixels(episode2)
  metric = helper_utils.compute_metric(episode1, episode2, gamma)
  return (obs1, obs2, metric)
