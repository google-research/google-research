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

"""Synthetic datasets for EIM experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

NINE_GAUSSIANS_DIST = "nine_gaussians"
TWO_RINGS_DIST = "two_rings"
CHECKERBOARD_DIST = "checkerboard"
TARGET_DISTS = [NINE_GAUSSIANS_DIST, TWO_RINGS_DIST, CHECKERBOARD_DIST]


class Ring2D(tfd.Distribution):
  """2D Ring distribution."""

  def __init__(self,
               radius_dist=None,
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name="Ring"):
    parameters = dict(locals())
    loc = tf.zeros([2], dtype=dtype)
    if radius_dist is None:
      radius_dist = tfd.Normal(loc=1., scale=0.1)
    self._loc = loc
    self._radius_dist = radius_dist
    super(Ring2D, self).__init__(
        dtype=dtype,
        reparameterization_type=tfd.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc],
        name=name)

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(self._loc)[:-1], self._radius_dist.batch_shape_tensor)

  def _batch_shape(self):
    return tf.broadcast_static_shape(self._loc.get_shape()[:-1],
                                     self._radius_dist.batch_shape)

  def _event_shape_tensor(self):
    return tf.constant([2], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([2])

  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    thetas = tf.random_uniform(
        new_shape, seed=seed, dtype=self.dtype) * 2. * math.pi
    rs = self._radius_dist.sample(new_shape, seed=seed)
    vecs = tf.stack([tf.math.sin(thetas), tf.math.cos(thetas)], axis=-1)

    sample = vecs * tf.expand_dims(rs, axis=-1)
    return tf.cast(sample, self.dtype)

  def _log_prob(self, event):
    radii = tf.norm(event, axis=-1, ord=2)
    return self._radius_dist.log_prob(radii) - tf.log(2 * math.pi * radii)


def two_rings_dist(scale=0.1):
  r_dist = tfd.Mixture(
      cat=tfd.Categorical(probs=[1., 1.]),
      components=[
          tfd.Normal(loc=0.6, scale=scale),
          tfd.Normal(loc=1.3, scale=scale)
      ])
  return Ring2D(radius_dist=r_dist)


def checkerboard_dist(num_splits=4):
  """Returns a checkerboard distribution."""
  bounds = np.linspace(-2., 2., num=(num_splits + 1), endpoint=True)
  uniforms = []
  for i in range(num_splits):
    for j in range(num_splits):
      if ((i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0)):
        low = tf.convert_to_tensor([bounds[i], bounds[j]], dtype=tf.float32)
        high = tf.convert_to_tensor([bounds[i + 1], bounds[j + 1]],
                                    dtype=tf.float32)
        u = tfd.Uniform(low=low, high=high)
        u = tfd.Independent(u, reinterpreted_batch_ndims=1)
        uniforms.append(u)
  return tfd.Mixture(
      cat=tfd.Categorical(probs=[1.] * len(uniforms)), components=uniforms)


def nine_gaussians_dist(variance=0.1):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-1., 0., 1.]:
    for j in [-1., 0., 1.]:
      loc = tf.constant([i, j], dtype=tf.float32)
      scale = tf.ones_like(loc) * tf.sqrt(variance)
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))
  return tfd.Mixture(
      cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32) / 9.),
      components=components)


def get_target_distribution(name, nine_gaussians_variance=0.01):
  if name == NINE_GAUSSIANS_DIST:
    return nine_gaussians_dist(variance=nine_gaussians_variance)
  elif name == TWO_RINGS_DIST:
    return two_rings_dist()
  elif name == CHECKERBOARD_DIST:
    return checkerboard_dist()
  else:
    raise ValueError("Invalid target name.")
