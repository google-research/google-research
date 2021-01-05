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

# python3
"""Defines the banana target spec."""

from typing import Text

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from hmc_swindles.targets import target_spec

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'banana',
]


def banana(ndims = 2,
           nonlinearity = 0.03,
           name = 'banana'):
  """Creates a banana-shaped distribution.

  This distribution was first described in [1]. The distribution is constructed
  by transforming a N-D gaussian with scale [10, 1, ...] by shifting the second
  dimension by `nonlinearity * (x0 - 100)` where `x0` is the value of the first
  dimension.

  This distribution is notable for having relatively narrow tails, while being
  derived from a simple, volume-preserving transformation of a normal
  distribution. Despite this simplicity, some inference algorithms have trouble
  sampling from this distribution.

  Args:
    ndims: Dimensionality of the distribution. Must be at least 2.
    nonlinearity: Controls the strength of the nonlinearity of the distribution.
    name: Name to prepend to ops created in this function, as well as to the
      `code_name` in the returned `TargetDensity`.

  Returns:
    target: `TargetDensity` specifying the banana distribution. The
      `distribution` attribute is an instance of `TransformedDistribution`.

  Raises:
    ValueError: If ndims < 2.

  #### References

  1. Haario, H., Saksman, E., & Tamminen, J. (1999). Adaptive proposal
      distribution for random walk Metropolis algorithm. Computational
      Statistics, 14(3), 375-396.
  """
  if ndims < 2:
    raise ValueError(f'ndims must be at least 2, saw: {ndims}')

  with tf.name_scope(name):

    def bijector_fn(x):
      """Banana transform."""
      batch_shape = tf.shape(x)[:-1]
      shift = tf.concat(
          [
              tf.zeros(tf.concat([batch_shape, [1]], axis=0)),
              nonlinearity * (tf.square(x[Ellipsis, :1]) - 100),
              tf.zeros(tf.concat([batch_shape, [ndims - 2]], axis=0)),
          ],
          axis=-1,
      )
      return tfb.Shift(shift)

    mg = tfd.MultivariateNormalDiag(
        loc=tf.zeros(ndims), scale_diag=[10.] + [1.] * (ndims - 1))
    dist = tfd.TransformedDistribution(
        mg, bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn))

    return target_spec.TargetDensity.from_distribution(
        distribution=dist,
        constraining_bijectors=tfb.Identity(),
        expectations=dict(
            params=target_spec.expectation(
                fn=tf.identity,
                human_name='Parameters',
                # The second dimension is a sum of scaled Chi2 and normal
                # distribution.
                # Mean of Chi2 with one degree of freedom is 1, but since the
                # first element has variance of 100, it cancels with the shift
                # (hence why the shift is there).
                ground_truth_mean=np.zeros(ndims),
                # Variance of Chi2 with one degree of freedom is 2.
                ground_truth_standard_deviation=np.array(
                    [10.] + [np.sqrt(1. + 2 * nonlinearity**2 * 10.**4)] +
                    [1.] * (ndims - 2)),
            ),),
        code_name=f'{name}_ndims_{ndims}_nonlinearity_{nonlinearity}',
        human_name='Banana',
    )
