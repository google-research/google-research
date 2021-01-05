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

# Lint as: python3
"""Tests for storm optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import test

from storm_optimizer.storm_optimizer import StormOptimizer


class StormOptimizerTest(test.TestCase):

  def testRunsMinimize(self):
    storm = StormOptimizer()
    w = tf.Variable([3.0])
    loss = tf.square(w)
    update_op = storm.minimize(loss, var_list=[w])

    if not tf.executing_eagerly():
      self.evaluate(tf.initializers.global_variables())
      for _ in range(3):
        self.evaluate(update_op)


if __name__ == '__main__':
  test.main()
