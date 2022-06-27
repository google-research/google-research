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

"""Custom loss functions for TensorFlow."""

import tensorflow as tf


def weighted_cross_entropy_with_logits_with_masked_class(
    pos_weight = 1.0):
  """Wrapper function for masked weighted cross-entropy with logits.

  This loss function ignores the classes with negative class id.

  Args:
    pos_weight: A coefficient to use on the positive examples.

  Returns:
    A weighted cross-entropy with logits loss function that ignores classes
    with negative class id.
  """

  def masked_weighted_cross_entropy_with_logits(y_true,
                                                logits):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    loss = tf.math.reduce_mean(mask * tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=logits, pos_weight=pos_weight))
    return loss

  return masked_weighted_cross_entropy_with_logits
