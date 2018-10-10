# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Large Margin Loss Function.

Let d be the estimated distance to boundary, and gamma and alpha
are margin loss parameters (gamma > 0 and alpha > 0).

The original margin loss can be written as:

loss = max(0, min(gamma - d, alpha * gamma))

The formulation written here can be obtained as:
min(gamma - d, alpha * gamma)
  = gamma + min(-d, alpha * gamma - gamma)
  = gamma - max(d, gamma - alpha * gamma)
  = gamma - max(d, gamma * (1-alpha))

One can see from here that the lower bound to distance to boundary is
distance_lower = gamma * (1-alpha).

loss = max(0, gamma - max(d, distance_lower))
Looking further:
loss = gamma + max(-gamma, -max(d, distance_lower))
     = gamma - min(gamma, max(d, distance_lower))

One can see from here that the distance is upper bounded by gamma.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_norm_fn(norm_type):
  norm_fn = lambda x: tf.norm(x, ord=norm_type)
  return norm_fn


def maximum_with_relu(a, b):
  return a + tf.nn.relu(b - a)


def _ensure_large_margin_args(name, sentinel, one_hot_labels, logits,
                              layers_list):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError(
        "Only call `%s` with "
        "named arguments (one_hot_labels=..., logits=..., ...)" % name)
  if one_hot_labels is None or logits is None or layers_list is None:
    raise ValueError("logits, one_hot_labels and layers_list must be provided.")


def large_margin(  # pylint: disable=invalid-name
    _sentinel=None,
    logits=None,
    one_hot_labels=None,
    layers_list=None,
    gamma=10,
    alpha_factor=4,
    top_k=1,
    dist_norm=2,
    epsilon=1e-8,
    use_approximation=True,
    loss_collection=tf.GraphKeys.LOSSES):
  """Creates a large margin loss.

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    one_hot_labels: `[batch_size, num_classes]` Target integer labels in `{0,
      1}`.
    layers_list: List of network layers to enforce large margin.
    gamma: Desired margin.
    alpha_factor: Factor to determine the lower bound of margin. both gamma and
      alpha_factor determine points to include in training the margin these
      points lie with distance to boundary of [gamma * (1 - alpha), gamma]
    top_k: Number of top classes to include in the margin loss.
    dist_norm: Distance to boundary defined on norm (options: be 1, 2, np.inf).
    epsilon: Dmall number to avoid division by 0.
    use_approximation: If true, use approximation of the margin gradient for
      less computationally expensive training.
    loss_collection: Collection to which the loss will be added.

  Returns:
    loss: Scalar `Tensor` of the same type as `logits`.
  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `one_hot_labels`.  Also if `one_hot_labels` or `logits` is None.
  """

  _ensure_large_margin_args("large_margin", _sentinel, one_hot_labels, logits,
                            layers_list)
  logits = tf.convert_to_tensor(logits)
  one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  assert top_k > 0

  dual_norm = {1: np.inf, 2: 2, np.inf: 1}
  norm_fn = get_norm_fn(dual_norm[dist_norm])
  with tf.name_scope("large_margin_loss"):
    class_prob = tf.nn.softmax(logits)
    # pick the correct class probability
    correct_class_prob = tf.reduce_sum(
        class_prob * one_hot_labels, axis=1, keepdims=True)

    # class probabilities except the correct
    other_class_prob = class_prob * (1. - one_hot_labels)
    if top_k > 1:
      # pick the top k class probabilities other than the correct.
      top_k_class_prob, _ = tf.nn.top_k(other_class_prob, k=top_k)
    else:
      top_k_class_prob = tf.reduce_max(other_class_prob, axis=1, keepdims=True)

    # difference between correct class probailities and top_k probabilities
    difference_prob = correct_class_prob - top_k_class_prob
    losses_list = []
    for layer in layers_list:
      difference_prob_grad = [
          tf.layers.flatten(tf.gradients(difference_prob[:, i], layer)[0])
          for i in range(top_k)
      ]

      difference_prob_gradnorm = tf.concat([
          tf.map_fn(norm_fn, difference_prob_grad[i])[:, tf.newaxis]
          for i in range(top_k)
      ], axis=1)

      if use_approximation:
        difference_prob_gradnorm = tf.stop_gradient(difference_prob_gradnorm)

      distance_to_boundary = difference_prob / (
          difference_prob_gradnorm + epsilon)

      # distances to consider between distance_upper and distance_lower bounds
      distance_upper = gamma
      distance_lower = gamma * (1 - alpha_factor)

      # enforce lower bound
      loss_layer = maximum_with_relu(distance_to_boundary, distance_lower)

      # enforce upper bound
      loss_layer = maximum_with_relu(0, distance_upper - loss_layer)

      # TODO(gamaleldin): add other loss types 'worst' etc.
      losses_list.append(tf.reduce_mean(loss_layer))

    loss = tf.reduce_mean(losses_list)
    # add loss to loss_collection
    tf.losses.add_loss(loss, loss_collection)
  return loss
