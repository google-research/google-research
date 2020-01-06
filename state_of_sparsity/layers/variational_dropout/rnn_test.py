# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for variational dropout reccurrent cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.testing.parameterized as parameterized
import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.variational_dropout as vd


# batch_size, seq_length, num_units, data_size
RNN_TEST_PARAMETERS = [(32, 25, 100, 33)]


@parameterized.parameters(RNN_TEST_PARAMETERS)
class RNNCellTest(vd.test_base.RNNTestCase):

  def testRNNCell_Train(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    rnn_cell = self.set_no_epsilon(vd.rnn.RNNCell)
    self.assertSameResult(
        self.set_training(rnn_cell),
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
    rnn_cell = self.set_no_epsilon(vd.rnn.RNNCell)
    self.assertSameResult(
        self.set_evaluation(rnn_cell),
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
        self.set_training(vd.rnn.RNNCell),
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
        self.set_training(vd.rnn.RNNCell),
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
        self.set_evaluation(vd.rnn.RNNCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, num_units],
        [num_units])


@parameterized.parameters(RNN_TEST_PARAMETERS)
class LSTMCellTest(vd.test_base.RNNTestCase):

  def testLSTMCell_Train(
      self,
      batch_size,
      seq_length,
      num_units,
      data_size):
    lstm_cell = self.set_no_epsilon(vd.rnn.LSTMCell)
    self.assertSameResult(
        self.set_training(lstm_cell),
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
    lstm_cell = self.set_no_epsilon(vd.rnn.LSTMCell)
    self.assertSameResult(
        self.set_evaluation(lstm_cell),
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
        self.set_training(vd.rnn.LSTMCell),
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
        self.set_training(vd.rnn.LSTMCell),
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
        self.set_evaluation(vd.rnn.LSTMCell),
        num_units,
        [batch_size, seq_length, data_size],
        [data_size + num_units, 4 * num_units],
        [4 * num_units])


if __name__ == "__main__":
  tf.test.main()
