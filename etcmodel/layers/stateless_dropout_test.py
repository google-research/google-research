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

"""Tests for stateless_dropout."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from etcmodel.layers import stateless_dropout as stateless_dropout_lib


class StatelessDropoutTest(tf.test.TestCase, parameterized.TestCase):

  def test_rate_zero(self):
    x = np.random.normal(size=(4, 8)).astype(np.float32)
    self.assertAllEqual(
        stateless_dropout_lib.stateless_dropout(x, 0, seed=[1, 0]), x)

  def test_rate_correctness(self):
    x = np.random.normal(size=(4, 8)).astype(np.float32)
    y = self.evaluate(
        stateless_dropout_lib.stateless_dropout(x, 0.8, seed=[2, 4]))
    self.assertBetween(np.sum(y == 0), 10, 31)

  def test_noise_shape_correctness(self):
    x = np.random.normal(size=(4, 16)).astype(np.float32)
    dropped_rows = self.evaluate(
        stateless_dropout_lib.stateless_dropout(
            x, 0.8, seed=[2, 4], noise_shape=[4, 1]))
    dropped_columns = self.evaluate(
        stateless_dropout_lib.stateless_dropout(
            x, 0.8, seed=[2, 4], noise_shape=[1, 16]))
    self.assertBetween(np.sum(np.sum(dropped_rows, axis=-1) == 0), 1, 3)
    self.assertBetween(np.sum(np.sum(dropped_columns, axis=0) == 0), 5, 15)

  def test_deterministic(self):
    x = np.random.normal(size=(4, 8)).astype(np.float32)
    y = self.evaluate(
        stateless_dropout_lib.stateless_dropout(x, 0.5, seed=[2, 4]))
    self.assertAllEqual(
        stateless_dropout_lib.stateless_dropout(x, 0.5, seed=[2, 4]), y)


if __name__ == '__main__':
  tf.test.main()
