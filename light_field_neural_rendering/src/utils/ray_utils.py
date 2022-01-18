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

"""Ray Utilities.

This code was adopted from parts of
https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py
"""

from jax import random
import jax.numpy as jnp


def cast_rays(z_vals, origins, directions):
  return origins[Ellipsis, None, :] + z_vals[Ellipsis, None] * directions[Ellipsis, None, :]


def sample_along_rays(key, origins, directions, num_samples, near, far,
                      randomized, lindisp):
  """Stratified sampling along the rays.

  Args:
    key: jnp.ndarray, random generator key.
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    num_samples: int.
    near: float, near clip.
    far: float, far clip.
    randomized: bool, use randomized stratified sampling.
    lindisp: bool, sampling linearly in disparity rather than depth.

  Returns:
    z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
    points: jnp.ndarray, [batch_size, num_samples, 3], sampled points.
  """
  batch_size = origins.shape[0]

  t_vals = jnp.linspace(0., 1., num_samples)
  if lindisp:
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  else:
    z_vals = near * (1. - t_vals) + far * t_vals

  if randomized:
    mids = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
    upper = jnp.concatenate([mids, z_vals[Ellipsis, -1:]], -1)
    lower = jnp.concatenate([z_vals[Ellipsis, :1], mids], -1)
    t_rand = random.uniform(key, [batch_size, num_samples])
    z_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast z_vals to make the returned shape consistent.
    z_vals = jnp.broadcast_to(z_vals[None, Ellipsis], [batch_size, num_samples])

  coords = cast_rays(z_vals, origins, directions)
  return z_vals, coords
