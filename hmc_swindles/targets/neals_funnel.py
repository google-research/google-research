# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Defines the neals_funnel target spec."""

from typing import Text

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from hmc_swindles.targets import target_spec

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'neals_funnel',
]


def neals_funnel(ndims = 10,
                 name = 'neals_funnel'):
  """Creates a funnel-shaped distribution.

  This distribution was first described in [1]. The distribution is constructed
  by transforming a N-D gaussian with scale [3, 1, ...] by scaling all but the
  first dimensions by `exp(x0 / 2)` where  `x0` is the value of the first
  dimension.

  This distribution is notable for having a relatively very narrow "neck" region
  which is challenging for HMC to explore. This distribution resembles the
  posteriors of centrally parameterized hierarchical models.

  Args:
    ndims: Dimensionality of the distribution. Must be at least 2.
    name: Name to prepend to ops created in this function, as well as to the
      `code_name` in the returned `TargetDensity`.

  Returns:
    target: `TargetDensity` specifying the funnel distribution. The
      `distribution` attribute is an instance of `TransformedDistribution`.

  Raises:
    ValueError: If ndims < 2.

  #### References

  1. Neal, R. M. (2003). Slice sampling. Annals of Statistics, 31(3), 705-767.
  """
  if ndims < 2:
    raise ValueError(f'ndims must be at least 2, saw: {ndims}')

  with tf.name_scope(name):

    def bijector_fn(x):
      """Funnel transform."""
      batch_shape = tf.shape(x)[:-1]
      scale = tf.concat(
          [
              tf.ones(tf.concat([batch_shape, [1]], axis=0)),
              tf.exp(x[Ellipsis, :1] / 2) *
              tf.ones(tf.concat([batch_shape, [ndims - 1]], axis=0)),
          ],
          axis=-1,
      )
      return tfb.Scale(scale)

    mg = tfd.MultivariateNormalDiag(
        loc=tf.zeros(ndims), scale_diag=[3.] + [1.] * (ndims - 1))
    dist = tfd.TransformedDistribution(
        mg, bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn))

    return target_spec.TargetDensity.from_distribution(
        distribution=dist,
        constraining_bijectors=tfb.Identity(),
        expectations=dict(
            params=target_spec.expectation(
                fn=tf.identity,
                human_name='Parameters',
                # The trailing dimensions come from a product distribution of
                # independent standard normal and a log-normal with a scale of
                # 3 / 2.
                # See https://en.wikipedia.org/wiki/Product_distribution for the
                # formulas.
                # For the mean, the formulas yield zero.
                ground_truth_mean=np.zeros(ndims),
                # For the standard deviation, all means are zero and standard
                # deivations of the normals are 1, so the formula reduces to
                # `sqrt((sigma_log_normal + mean_log_normal**2))` which reduces
                # to `exp((sigma_log_normal)**2)`.
                ground_truth_standard_deviation=np.array([3.] +
                                                         [np.exp((3. / 2)**2)] *
                                                         (ndims - 1)),
            ),),
        code_name=f'{name}_ndims_{ndims}',
        human_name='Neal\'s Funnel',
    )
