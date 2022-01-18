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

# python3
"""Defines the ill conditioned gaussian target spec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from hmc_swindles.targets import target_spec

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'ill_conditioned_gaussian',
]


def ill_conditioned_gaussian(
    ndims = 100,
    gamma_shape_parameter = 0.5,
    max_eigvalue = None,
    seed = 10,
    name='ill_conditioned_gaussian'):
  """Creates a random ill-conditioned Gaussian.

  The covariance matrix has eigenvalues sampled from the inverse Gamma
  distribution with the specified shape, and then rotated by a random orthogonal
  matrix.

  Note that this function produces reproducible targets, i.e. the `seed`
  argument always needs to be non-`None`.

  Args:
    ndims: Dimensionality of the Gaussian.
    gamma_shape_parameter: The shape parameter of the inverse Gamma
      distribution.
    max_eigvalue: If set, will normalize the eigenvalues such that the maximum
      is this value.
    seed: Seed to use when generating the eigenvalues and the random orthogonal
      matrix.
    name: Name to prepend to ops created in this function, as well as to the
      `code_name` in the returned `TargetDensity`.

  Returns:
    target: `TargetDensity` specifying the requested Gaussian distribution. The
      `distribution` attribute is an instance of `MultivariateNormalTriL`.
  """
  with tf.name_scope(name):
    rng = np.random.RandomState(seed=seed & (2**32 - 1))
    eigenvalues = 1. / np.sort(
        rng.gamma(shape=gamma_shape_parameter, scale=1., size=ndims))
    if max_eigvalue is not None:
      eigenvalues *= max_eigvalue / eigenvalues.max()

    q, r = np.linalg.qr(rng.randn(ndims, ndims))
    q *= np.sign(np.diag(r))

    covariance = (q * eigenvalues).dot(q.T)

    gaussian = tfd.MultivariateNormalTriL(
        loc=tf.zeros(ndims),
        scale_tril=tf.linalg.cholesky(
            tf.convert_to_tensor(covariance, dtype=tf.float32)))

    # TODO(siege): Expose the eigenvalues directly.
    return target_spec.TargetDensity.from_distribution(
        distribution=gaussian,
        constraining_bijectors=tfb.Identity(),
        expectations=dict(
            first_moment=target_spec.expectation(
                fn=tf.identity,
                human_name='First moment',
                ground_truth_mean=np.zeros(ndims),
                ground_truth_standard_deviation=np.sqrt(np.diag(covariance)),
            ),
            second_moment=target_spec.expectation(
                fn=tf.square,
                human_name='Second moment',
                ground_truth_mean=np.diag(covariance),
                # The variance of the second moment is
                # E[x**4] - E[x**2]**2 = 3 sigma**4 - sigma**4 = 2 sigma**4.
                ground_truth_standard_deviation=(np.sqrt(2) *
                                                 np.diag(covariance)),
            )),
        code_name='{name}_ndims_{ndims}_gamma_shape_'
        '{gamma_shape}_seed_{seed}{max_eigvalue_str}'.format(
            name=name,
            ndims=ndims,
            gamma_shape=gamma_shape_parameter,
            seed=seed,
            max_eigvalue_str='' if max_eigvalue is None else
            '_max_eigvalue_{}'.format(max_eigvalue)),
        human_name='Ill-conditioned Gaussian',
    )
