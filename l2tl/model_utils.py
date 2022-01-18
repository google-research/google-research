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

"""Model utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def metric_fn(labels, logits):
  """Metric function for evaluation."""
  predictions = tf.argmax(logits, axis=1)
  top_1_accuracy = tf.metrics.accuracy(labels, predictions)

  return {
      'top_1_accuracy': top_1_accuracy,
  }


def get_label(labels, params, num_classes, batch_size=-1):  # pylint: disable=unused-argument
  """Returns the label."""
  one_hot_labels = tf.one_hot(tf.cast(labels, tf.int64), num_classes)
  return one_hot_labels


def update_exponential_moving_average(tensor, momentum, name=None):
  """Returns an exponential moving average of `tensor`.

  We will update the moving average every time the returned `tensor` is
  evaluated. A zero-debias will be applied, so we will return unbiased
  estimates during the first few training steps.
  Args:
    tensor: A floating point tensor.
    momentum: A scalar floating point Tensor with the same dtype as `tensor`.
    name: Optional string, the name of the operation in the TensorFlow graph.

  Returns:
    A Tensor with the same shape and dtype as `tensor`.
  """
  with tf.variable_scope(name, 'update_exponential_moving_average',
                         [tensor, momentum]):
    numerator = tf.get_variable(
        'numerator', initializer=0.0, trainable=False, use_resource=True)
    denominator = tf.get_variable(
        'denominator', initializer=0.0, trainable=False, use_resource=True)
    update_ops = [
        numerator.assign(momentum * numerator + (1 - momentum) * tensor),
        denominator.assign(momentum * denominator + (1 - momentum)),
    ]
    with tf.control_dependencies(update_ops):
      return numerator.read_value() / denominator.read_value()
