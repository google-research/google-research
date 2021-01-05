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

# Lint as: python2, python3
"""Slow/long-running unit tests for mobile_model_v3.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import mobile_search_space_v3
from tunas import schema
from tunas.rematlib import mobile_model_v3


def _with_largest_possible_masks(oneof):
  """Add masks to enable all possible ops / filters in the search space."""
  if oneof.tag == basic_specs.OP_TAG:
    n = len(oneof.choices)
    mask = tf.constant([1 / n] * n, dtype=tf.float32)
  elif oneof.tag == basic_specs.FILTERS_TAG:
    largest_index = None
    for i, choice in enumerate(oneof.choices):
      if largest_index is None or choice > oneof.choices[largest_index]:
        largest_index = i
    mask = tf.one_hot(largest_index, len(oneof.choices), dtype=tf.float32)
  else:
    raise ValueError('Unrecognized tag: {!r}'.format(oneof.tag))
  return schema.OneOf(oneof.choices, oneof.tag, mask)


class MobileModelV3SlowTest(tf.test.TestCase):

  def test_mobilenet_v3_like_search_gradients(self):
    model_spec = mobile_search_space_v3.mobilenet_v3_like_search()
    model_spec = schema.map_oneofs(_with_largest_possible_masks, model_spec)
    model = mobile_model_v3.get_model(
        model_spec, num_classes=1001, force_stateless_batch_norm=True)

    inputs = tf.random_normal(shape=[8, 224, 224, 3], dtype=tf.float32)
    model.build(inputs.shape)

    logits, unused_endpoints = model.apply(inputs, training=True)
    labels = tf.one_hot([0]*8, 1001, dtype=tf.float32)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    variables = model.trainable_variables()
    grads = tf.gradients(loss, variables)
    for variable, grad in zip(variables, grads):
      self.assertIsNotNone(
          grad, msg='Gradient for {} is None'.format(variable.name))

    self.evaluate(tf.global_variables_initializer())
    for variable, array in zip(variables, self.evaluate(grads)):
      self.assertFalse(
          np.all(np.equal(array, 0)),
          msg='Gradient for {} is identically zero'.format(variable.name))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
