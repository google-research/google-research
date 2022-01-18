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
"""Defines the 1PL item-response theory model target spec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Callable

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from hmc_swindles.targets import data
from hmc_swindles.targets import joint_distribution_posterior
from hmc_swindles.targets import target_spec

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'item_response_theory',
]


def item_response_theory(
    dataset_fn,
    name='item_response_theory',
):
  """One-parameter logistic item-response theory (IRT) model.

  Args:
    dataset_fn: A function to create an IRT data set.
    name: Name to prepend to ops created in this function, as well as to the
      `code_name` in the returned `TargetDensity`.

  Returns:
    target: `TargetDensity`.
  """
  with tf.name_scope(name) as name:
    dataset = dataset_fn()
    have_test = dataset.test_student_ids.shape[0] > 0

    num_students = dataset.train_student_ids.max()
    num_questions = dataset.train_question_ids.max()
    if have_test:
      num_students = max(num_students, dataset.test_student_ids.max())
      num_questions = max(num_questions, dataset.test_question_ids.max())

    # TODO(siege): Make it an option to use a sparse encoding, the choice
    # clearly depends on the dataset sparsity.
    def make_dense_encoding(student_ids, question_ids, correct):
      dense_y = np.zeros([num_students, num_questions], np.float32)
      y_mask = np.zeros_like(dense_y)
      dense_y[student_ids - 1, question_ids - 1] = (correct)
      y_mask[student_ids - 1, question_ids - 1] = 1.
      return dense_y, y_mask

    train_dense_y, train_y_mask = make_dense_encoding(
        dataset.train_student_ids,
        dataset.train_question_ids,
        dataset.train_correct,
    )
    test_dense_y, test_y_mask = make_dense_encoding(
        dataset.test_student_ids,
        dataset.test_question_ids,
        dataset.test_correct,
    )

    root = tfd.JointDistributionCoroutine.Root

    def model_fn(dense_y, y_mask):
      """Model definition."""
      mean_student_ability = yield root(tfd.Normal(0.75, 1.))
      student_ability = yield root(
          tfd.Independent(tfd.Normal(0., tf.ones([dense_y.shape[0]])), 1))
      question_difficulty = yield root(
          tfd.Independent(tfd.Normal(0., tf.ones([dense_y.shape[1]])), 1))
      logits = (
          mean_student_ability[Ellipsis, tf.newaxis, tf.newaxis] +
          student_ability[Ellipsis, tf.newaxis] -
          question_difficulty[Ellipsis, tf.newaxis, :])
      masked_logits = logits * y_mask - 1e10 * (1 - y_mask)
      yield tfd.Independent(tfd.Bernoulli(masked_logits), 2)

    train_joint_dist = tfd.JointDistributionCoroutine(
        functools.partial(model_fn, train_dense_y, train_y_mask))
    test_joint_dist = tfd.JointDistributionCoroutine(
        functools.partial(model_fn, test_dense_y, test_y_mask))
    dist = joint_distribution_posterior.JointDistributionPosterior(
        train_joint_dist, (None, None, None, train_dense_y))

    expectations = {
        'params':
            target_spec.expectation(
                fn=lambda params: tf.concat(  # pylint: disable=g-long-lambda
                    (params[0][Ellipsis, tf.newaxis],) + params[1:],
                    axis=-1),
                human_name='Parameters',
            )
    }
    if have_test:
      expectations['test_nll'] = target_spec.expectation(
          fn=lambda params: (  # pylint: disable=g-long-lambda
              -test_joint_dist.sample_distributions(value=params)[0][-1].
              log_prob(test_dense_y)),
          human_name='Test NLL',
      )

      def per_example_test_nll(params):
        """Computes per-example test NLL."""
        test_y_idx = np.stack(
            [dataset.test_student_ids - 1, dataset.test_question_ids - 1],
            axis=-1)

        dense_nll = (-test_joint_dist.sample_distributions(
            value=params)[0][-1].distribution.log_prob(test_dense_y))
        vectorized_dense_nll = tf.reshape(dense_nll,
                                          [-1, num_students, num_questions])
        # TODO(siege): Avoid using vmap here.
        log_prob_y = tf.vectorized_map(
            lambda nll: tf.gather_nd(nll, test_y_idx), vectorized_dense_nll)
        return tf.reshape(log_prob_y,
                          list(params[0].shape) + [test_y_idx.shape[0]])

      expectations['per_example_test_nll'] = target_spec.expectation(
          fn=per_example_test_nll,
          human_name='Per-example Test NLL',
      )

    return target_spec.TargetDensity.from_distribution(
        distribution=dist,
        constraining_bijectors=(tfb.Identity(), tfb.Identity(), tfb.Identity()),
        expectations=expectations,
        code_name='{}_{}'.format(dataset.code_name, name),
        human_name='{} 1PL Item-Response Theory'.format(dataset.human_name),
    )
