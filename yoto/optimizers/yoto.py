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

"""The You-Only-Train-Once (YOTO) multi-objective optimizer."""

import gin
import tensorflow.compat.v1 as tf

from yoto.optimizers import base
from yoto.optimizers import distributions


@gin.configurable("YotoOptimizer")
class YotoOptimizer(base.MultiLossOptimizerWithConditioning):
  """The You-Only-Train-Once optimizer.

  It conditions each input with the weights that will be used to combine the
  losses when evaluating it.
  """

  def __init__(self, problem, batch_size=None,
               train_distribution_spec_class=None,
               extra_inputs_preprocessing=distributions.TransformType.IDENTITY,
               weights_per_sample=True,
               normalize_loss_weights=False):
    """Initialize a YotoOptimizer object.

    YotoOptimizer trains a single model to minimize over a distribution of loss
    functions, not a single one. We assume the loss distribution is parametrized
    by a vector (loss weights), which are sampled randomly during training. The
    model is trained conditional on these loss weights.

    Args:
      problem: An instance of the Problem class, defining the learning problem
        to be solved.
      batch_size: Int. Batch size to be used for optimization.
      train_distribution_spec_class: DistributionSpec class (gin-injected),
        specifying the weight distribution.
      extra_inputs_preprocessing: TransformType enum value that specifies the
        pre-processing to be applied to the loss weights before they are fed
        to the model for conditioning.
      weights_per_sample: Bool. If True, the loss weights are sampled per
        training sample, otherwise - per training mini-batch.
      normalize_loss_weights: Bool. Whether to nomalize the loss weights by
        their geometric mean.
    """
    super(YotoOptimizer, self).__init__(problem=problem, batch_size=batch_size)
    train_distribution_spec = train_distribution_spec_class()
    shape = (self._batch_size, len(self._losses_names))
    if weights_per_sample:
      self._sampled_weights = distributions.get_sample(
          shape, train_distribution_spec, seed=17)
    else:
      weights = distributions.get_sample(
          (1, shape[1]), train_distribution_spec, seed=17)
      self._sampled_weights = tf.broadcast_to(weights, shape)
    self._preprocess = distributions.get_transform(extra_inputs_preprocessing)
    self._extra_inputs = self._preprocess(self._sampled_weights)
    if normalize_loss_weights:
      normalizer = tf.pow(
          tf.reduce_prod(self._sampled_weights, axis=1, keepdims=True),
          1. / float(len(self._losses_names)))
      self._sampled_weights = self._sampled_weights / normalizer

  def compute_train_loss_and_update_op(self, inputs, base_optimizer,
                                       base_optimizer_conditioning=None):
    """Returns the training loss and the update op."""
    losses, metrics = self._problem.losses_and_metrics(inputs,
                                                       self._extra_inputs,
                                                       training=True)
    del metrics
    losses_ordered = [losses[loss_name] for loss_name in self._losses_names]
    losses_stacked = tf.stack(losses_ordered, axis=1)
    loss_sum = tf.reduce_sum(self._sampled_weights * losses_stacked, axis=1)
    loss = tf.reduce_mean(loss_sum)

    # Store the "conditioning" and "normal" variables, so that they can be
    # optimized separately in main.py, if required.
    all_vars = tf.trainable_variables()
    self.conditioning_vars = [v for v in all_vars if "conditioning" in v.name]
    self.normal_vars = [v for v in all_vars if "conditioning" not in v.name]
    self.all_vars = all_vars
    assert set(self.all_vars) == set(self.normal_vars + self.conditioning_vars)

    if base_optimizer_conditioning:
      train_op_normal = base_optimizer.minimize(
          loss, global_step=tf.train.get_or_create_global_step(),
          var_list=self.normal_vars)
      train_op_conditioning = base_optimizer_conditioning.minimize(
          loss, var_list=self.conditioning_vars)
      train_op = tf.group([train_op_normal, train_op_conditioning])
    else:
      train_op = base_optimizer.minimize(
          loss, global_step=tf.train.get_or_create_global_step(),
          var_list=self.all_vars)
    return loss, train_op

  def compute_eval_loss(self, inputs):
    """Returns the train loss computed in the eval mode (training=False)."""
    losses, metrics = self._problem.losses_and_metrics(inputs,
                                                       self._extra_inputs,
                                                       training=False)
    del metrics
    losses_ordered = [losses[loss_name] for loss_name in self._losses_names]
    losses_stacked = tf.stack(losses_ordered, axis=1)
    loss_sum = tf.reduce_sum(self._sampled_weights * losses_stacked, axis=1)
    loss = tf.reduce_mean(loss_sum)
    return loss

  def compute_eval_losses_and_metrics_for_weights(self, inputs, weights_dict):
    """Returns losses and metrics computed by evaluation."""
    self._check_weights_dict(weights_dict)
    n_losses = len(self._losses_names)
    # TODO(josipd): Don't hard-code float32.
    weight_vector = tf.constant([weights_dict[loss_name]
                                 for loss_name in self._losses_names],
                                dtype=tf.float32)
    extra_inputs = tf.broadcast_to(self._preprocess(weight_vector),
                                   (self._batch_size, n_losses))
    losses, metrics = self._problem.losses_and_metrics(
        inputs, extra_inputs, training=False)
    return losses, metrics
