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

"""Optimizers based on scalarization.

One of the simplest approaches to optimizing multi-loss problems is to scalarize
to a real objective by combining the individual losses. Depending on how the
scalarization is performed, different optimization algorithms arise.
"""

import gin
import tensorflow.compat.v1 as tf


from yoto.optimizers import base as optimizers_base
from yoto.optimizers import distributions


@gin.configurable("LinearlyScalarizedOptimizer")
class LinearlyScalarizedOptimizer(optimizers_base.MultiLossOptimizer):
  r"""An optimizer that linearly scalarizes the losss.

  Namely, if the losses are loss_1, ..., loss_n, then it minimizes
    \sum_i loss_i * weight_i,
  for fixed weights. The weights can be either randomly drawn from one of the
  supported distributions, or fixed.
  """

  def __init__(self, problem, weights,
               batch_size=None, seed=17):
    """Initializes the optimizer.

    Args:
      problem: An instance of `problems.Problem`.
      weights: Either `distributions.DistributionSpec` class or a
        dictionary mapping the loss names to their corresponding
        weights.
      batch_size: Passed to the initializer of `MultiLossOptimizer`.
      seed: random seed to be used for sampling the weights.
    """
    super(LinearlyScalarizedOptimizer, self).__init__(
        problem, batch_size=batch_size)
    sampled_weights = distributions.get_samples_as_dicts(
        weights, names=self._losses_names, seed=seed)[0]
    self._check_weights_dict(sampled_weights)
    self._weights = sampled_weights

  def compute_train_loss_and_update_op(self, inputs, base_optimizer):
    losses, metrics = self._problem.losses_and_metrics(inputs, training=True)
    del metrics
    linearized_loss = 0.
    for loss_name, loss_value in losses.items():
      linearized_loss += tf.reduce_mean(loss_value * self._weights[loss_name])
    train_op = base_optimizer.minimize(
        linearized_loss, global_step=tf.train.get_or_create_global_step())
    self.normal_vars = tf.trainable_variables()
    return linearized_loss, train_op

  def compute_eval_loss(self, inputs):
    losses, metrics = self._problem.losses_and_metrics(inputs, training=False)
    del metrics
    linearized_loss = 0.
    for loss_name, loss_value in losses.items():
      linearized_loss += tf.reduce_mean(loss_value * self._weights[loss_name])
    return linearized_loss
