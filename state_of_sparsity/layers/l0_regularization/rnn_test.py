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

"""Tests for l0 regularized recurrent cells.

TODO(tgale): There is a lot of redundancy in these tests and in the
nn tests, and between the l0 tests and the variational dropout tests.
Extract some common base classes and utilities we can use to make
these tests more succinct.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.testing.parameterized as parameterized
import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.l0_regularization as l0

# batch_size, seq_length, num_units, data_size
RNN_TEST_PARAMETERS = [(32, 25, 100, 33)]


@parameterized.parameters(RNN_TEST_PARAMETERS)
class RNNCellTest(l0.test_base.RNNTestCase):

  def testRNNCell_Train(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.assertSameResult(
        self.set_training(l0.rnn.RNNCell),
        tf.nn.rnn_cell.BasicRNNCell,
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])

  def testRNNCell_Eval(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.assertSameResult(
        self.set_evaluation(l0.rnn.RNNCell),
        tf.nn.rnn_cell.BasicRNNCell,
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])

  def testRNNCell_SameNoiseForAllTimeSteps(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertSameNoiseForAllTimesteps(
        self.set_training(l0.rnn.RNNCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])

  def testRNNCell_DifferentNoiseAcrossBatches(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertDifferentNoiseAcrossBatches(
        self.set_training(l0.rnn.RNNCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])

  def testRNNCell_DeterministicEval(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertDeterministic(
        self.set_evaluation(l0.rnn.RNNCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])


@parameterized.parameters(RNN_TEST_PARAMETERS)
class LSTMCellTest(l0.test_base.RNNTestCase):

  def testLSTMCell_Train(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.assertSameResult(
        self.set_training(l0.rnn.LSTMCell),
        tf.nn.rnn_cell.LSTMCell,
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])

  def testLSTMCell_Eval(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.assertSameResult(
        self.set_evaluation(l0.rnn.LSTMCell),
        tf.nn.rnn_cell.LSTMCell,
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])

  def testLSTMCell_SameNoiseForAllTimeSteps(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertSameNoiseForAllTimesteps(
        self.set_training(l0.rnn.LSTMCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])

  def testLSTMCell_DifferentNoiseAcrossBatches(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertDifferentNoiseAcrossBatches(
        self.set_training(l0.rnn.LSTMCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])

  def testLSTMCell_DeterministicEval(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    self.fix_random_seeds()
    self.assertDeterministic(
        self.set_evaluation(l0.rnn.LSTMCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])


if __name__ == "__main__":
  tf.test.main()
