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

"""Wrapper to flatten image observations.
"""

import collections
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class FlattenImageObservationsWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Env wrapper to Flatten all image observations.

  Essentially the same as FlattenObservationsWrapper but stacks along channel
  dim and returns rank 3 tensor.

  Images are optionally resized to a specified output resolution, else images
  must all be the same size.
  """

  def __init__(self,
               env,
               out_width_height = None):
    super(FlattenImageObservationsWrapper, self).__init__(env)

    self.wh = out_width_height

    obs_spec: array_spec.ArraySpec = self._env.observation_spec()
    if not isinstance(obs_spec, collections.OrderedDict):
      raise ValueError('Unsupported observation_spec %s' % str(obs_spec))

    o_shape = None
    o_dtype = None
    o_name = []
    for _, obs in obs_spec.items():
      if not isinstance(obs, array_spec.ArraySpec):
        raise ValueError('Unsupported observation_spec %s' % str(obs))

      if len(obs.shape) != 3:
        raise ValueError('All observations must be images (got shape %s).' % (
            str(obs.shape)))

      if self.wh:
        # The image size will be normalized.
        cur_shape = self.wh + (obs.shape[2],)
      else:
        cur_shape = obs.shape

      if o_shape is None:
        o_shape = list(obs.shape)
        o_dtype = obs.dtype
      else:
        if tuple(o_shape[0:2]) != cur_shape[0:2]:
          raise ValueError('All images must be the same shape.')
        if o_dtype != obs.dtype:
          raise ValueError('All images must be the same dtype.')
        o_shape[2] += obs.shape[2]
      o_name.append(obs.name)

    self._observation_spec = array_spec.ArraySpec(
        shape=o_shape,
        dtype=o_dtype,
        name='_'.join(o_name) + '_flattened')

  def _reset(self):
    return self._get_timestep(self._env.reset())

  def _step(self, action):
    return self._get_timestep(self._env.step(action))

  def _get_timestep(self, time_step):
    time_step = time_step._asdict()

    obs = []
    obs_spec: array_spec.ArraySpec = self._env.observation_spec()
    for key, _ in obs_spec.items():  # recall: obs_spec is an ordered_dict
      img = time_step['observation'][key]
      if self.wh:
        img = np.asarray(Image.fromarray(img).resize(self.wh))
        assert img.shape[0:2] == self.wh
      obs.append(img)
    time_step['observation'] = np.concatenate(obs, axis=-1)
    assert self.observation_spec().shape == time_step['observation'].shape
    return ts.TimeStep(**time_step)

  def observation_spec(self):
    return self._observation_spec
