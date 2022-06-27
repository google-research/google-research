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

"""Tests for cost_model_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import cost_model_lib
from tunas import schema


class CostModelLibTest(tf.test.TestCase):

  def test_get_mask(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'OneOf must have a mask'):
      cost_model_lib.get_mask(
          schema.OneOf([6, 7, 8, 9],
                       basic_specs.OP_TAG))

    mask = cost_model_lib.get_mask(
        schema.OneOf([6, 7, 8, 9],
                     basic_specs.OP_TAG,
                     tf.constant([0, 0, 1, 0])))
    self.assertAllEqual(self.evaluate(mask), [0, 0, 1, 0])

  def test_estimate_cost(self):
    features = tf.zeros([144], tf.float32)
    output = cost_model_lib.estimate_cost(features, 'proxylessnas_search')
    self.assertEqual(output.shape.as_list(), [])
    self.assertEqual(output.dtype, tf.float32)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
