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

#  python3
"""Defines a sparse logistic regression target spec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Callable

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from hmc_swindles.targets import data
from hmc_swindles.targets import joint_distribution_posterior
from hmc_swindles.targets import target_spec

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'sparse_logistic_regression',
]


def sparse_logistic_regression(
    dataset_fn,
    name='sparse_logistic_regression',
):
  """Bayesian logistic regression with a sparsity-inducing prior.

  Args:
    dataset_fn: A function to create a classification data set. The dataset must
      have binary labels.
    name: Name to prepend to ops created in this function, as well as to the
      `code_name` in the returned `TargetDensity`.

  Returns:
    target: `TargetDensity`.
  """
  with tf.name_scope(name) as name:
    dataset = dataset_fn()

    num_train_points = dataset.train_features.shape[0]
    num_test_points = dataset.test_features.shape[0]
    have_test = num_test_points > 0

    # Add bias.
    train_features = tf.concat(
        [dataset.train_features,
         tf.ones([num_train_points, 1])], axis=-1)
    train_labels = tf.convert_to_tensor(dataset.train_labels)
    test_features = tf.concat(
        [dataset.test_features,
         tf.ones([num_test_points, 1])], axis=-1)
    test_labels = tf.convert_to_tensor(dataset.test_labels)
    num_features = int(train_features.shape[1])

    root = tfd.JointDistributionCoroutine.Root
    zero = tf.zeros(num_features)
    one = tf.ones(num_features)
    half = tf.fill([num_features], 0.5)

    def model_fn(features):
      unscaled_weights = yield root(tfd.Independent(tfd.Normal(zero, one), 1))
      local_scales = yield root(tfd.Independent(tfd.Gamma(half, half), 1))
      global_scale = yield root(tfd.Gamma(0.5, 0.5))

      weights = unscaled_weights * local_scales * global_scale[Ellipsis, tf.newaxis]

      logits = tf.einsum('nd,...d->...n', features, weights)
      yield tfd.Independent(tfd.Bernoulli(logits=logits), 1)

    train_joint_dist = tfd.JointDistributionCoroutine(
        functools.partial(model_fn, features=train_features))
    test_joint_dist = tfd.JointDistributionCoroutine(
        functools.partial(model_fn, features=test_features))
    dist = joint_distribution_posterior.JointDistributionPosterior(
        train_joint_dist, (None, None, None, train_labels))

    expectations = {
        'params':
            target_spec.expectation(
                fn=lambda params: tf.concat(  # pylint: disable=g-long-lambda
                    params[:2] + (params[2][Ellipsis, tf.newaxis],),
                    axis=-1),
                human_name='Parameters',
            )
    }
    if have_test:
      expectations['test_nll'] = target_spec.expectation(
          fn=lambda params: (  # pylint: disable=g-long-lambda
              -test_joint_dist.sample_distributions(value=params)[0][-1].
              log_prob(test_labels)),
          human_name='Test NLL',
      )
      expectations['per_example_test_nll'] = target_spec.expectation(
          fn=lambda params: (  # pylint: disable=g-long-lambda
              -test_joint_dist.sample_distributions(value=params)
              [0][-1].distribution.log_prob(test_labels)),
          human_name='Per-example Test NLL',
      )

    return target_spec.TargetDensity.from_distribution(
        distribution=dist,
        constraining_bijectors=(tfb.Identity(), tfb.Exp(), tfb.Exp()),
        expectations=expectations,
        code_name='{}_{}'.format(dataset.code_name, name),
        human_name='{} Sparse Logistic Regression'.format(dataset.human_name),
    )
