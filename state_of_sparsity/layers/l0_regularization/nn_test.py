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

"""Tests for l0 layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import absl.testing.parameterized as parameterized
import numpy as np
import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.l0_regularization as l0


# Parameters to test the matmul primitive on. First dimensions of
# first matrix, second dimension of first matrix/first dimension
# of second matrix, second dimension of second matrix.
MATMUL_TEST_PARAMETERS = [(32, 200, 100)]


@parameterized.parameters(MATMUL_TEST_PARAMETERS)
class MatmulTest(l0.test_base.TestCase):

  def testMatmulTrain(self, m, n, k):
    self.assertSameResult(
        l0.nn.matmul_train,
        tf.matmul,
        [m, n],
        [n, k])

  def testMatmulTrain_NonDeterministic(self, m, n, k):
    self.assertNonDeterministic(
        l0.nn.matmul_train,
        [m, n],
        [n, k])

  def testMatmulEval(self, m, n, k):
    self.assertSameResult(
        l0.nn.matmul_eval,
        tf.matmul,
        [m, n],
        [n, k])

  def testMatmulEval_Deterministic(self, m, n, k):
    self.assertDeterministic(
        l0.nn.matmul_eval,
        [m, n],
        [n, k])


# Parameters to test the batched matmul primitive on. First dimension
# of the first matrix, second dimension of the first matrix, third
# dimension of the first matrix/first dimenions of the second matrix,
# second dimension of the second matrix.
BROADCAST_MATMUL_TEST_PARAMETERS = [(32, 20, 200, 100),
                                    (1, 10, 100, 50)]


@parameterized.parameters(BROADCAST_MATMUL_TEST_PARAMETERS)
class BroadcastMatmulTest(l0.test_base.TestCase):

  def set_axes(self, ref_op):
    return functools.partial(ref_op, axes=[[2], [0]])

  def testBroadcastMatmulTrain(self, m, t, n, k):
    self.assertSameResult(
        l0.nn.broadcast_matmul_train,
        self.set_axes(tf.tensordot),
        [m, t, n],
        [n, k])

  def testBroadcastMatmulTrain_NonDeterministic(self, m, t, n, k):
    self.assertNonDeterministic(
        l0.nn.broadcast_matmul_train,
        [m, t, n],
        [n, k])

  def testBroadcastMatmulEval(self, m, t, n, k):
    self.assertSameResult(
        l0.nn.broadcast_matmul_eval,
        self.set_axes(tf.tensordot),
        [m, t, n],
        [n, k])

  def testBroadcastMatmulEval_Deterministic(self, m, t, n, k):
    self.assertDeterministic(
        l0.nn.broadcast_matmul_eval,
        [m, t, n],
        [n, k])

# Parameters to test the conv2d primitive with. Input tensor batch size,
# input channels, input height, input width, size of the convolutional
# filters, number of output channels.
CONV2D_TEST_PARAMETERS = [(32, 3, 224, 224, 3, 64)]


@parameterized.parameters(CONV2D_TEST_PARAMETERS)
class Conv2dTest(l0.test_base.TestCase):

  def testConv2dTrain(
      self,
      batch_size,
      in_channels,
      height,
      width,
      filter_size,
      out_channels):
    self.assertSameResult(
        self.fix_padding_and_strides(l0.nn.conv2d_train),
        self.fix_padding_and_strides(tf.nn.conv2d),
        [batch_size, height, width, in_channels],
        [filter_size, filter_size, in_channels, out_channels])

  def testConv2dTrain_NonDeterministic(
      self,
      batch_size,
      in_channels,
      height,
      width,
      filter_size,
      out_channels):
    self.assertNonDeterministic(
        self.fix_padding_and_strides(l0.nn.conv2d_train),
        [batch_size, height, width, in_channels],
        [filter_size, filter_size, in_channels, out_channels])

  def testConv2dEval(
      self,
      batch_size,
      in_channels,
      height,
      width,
      filter_size,
      out_channels):
    self.assertSameResult(
        self.fix_padding_and_strides(l0.nn.conv2d_eval),
        self.fix_padding_and_strides(tf.nn.conv2d),
        [batch_size, height, width, in_channels],
        [filter_size, filter_size, in_channels, out_channels])

  def testConv2dEval_Deterministic(
      self,
      batch_size,
      in_channels,
      height,
      width,
      filter_size,
      out_channels):
    self.assertDeterministic(
        self.fix_padding_and_strides(l0.nn.conv2d_eval),
        [batch_size, height, width, in_channels],
        [filter_size, filter_size, in_channels, out_channels])


# Parameters for the embedding lookup tests. Batch size, sequence length,
# vocabulary size, embedding vector size
EMBEDDING_TEST_PARAMETERS = [(32, 25, 10000, 512)]


@parameterized.parameters(EMBEDDING_TEST_PARAMETERS)
class EmbeddingLookupTest(l0.test_base.TestCase):

  def testEmbeddingLookupTrain(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    self.assertSameResult(
        self.flip_input_wrapper(l0.nn.embedding_lookup_train),
        self.flip_input_wrapper(tf.nn.embedding_lookup),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)

  def testEmbeddingLookupTrain_NonDeterministic(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    self.assertNonDeterministic(
        self.flip_input_wrapper(l0.nn.embedding_lookup_train),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)

  def testEmbeddingLookupEval(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    self.assertSameResult(
        self.flip_input_wrapper(l0.nn.embedding_lookup_eval),
        self.flip_input_wrapper(tf.nn.embedding_lookup),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)

  def testEmbeddingLookupEval_Deterministic(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    self.assertDeterministic(
        self.flip_input_wrapper(l0.nn.embedding_lookup_eval),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)


# Dimensions to calculate the regularization contribution over, and
# the beta, gamma, and zeta parameters.
L0_NORM_TEST_PARAMETERS = [(256, 128, 2.0 / 3.0, -0.1, 1.1)]


@parameterized.parameters(L0_NORM_TEST_PARAMETERS)
class TestL0Norm(l0.test_base.TestCase):

  def testL0Norm(self, d, k, beta, gamma, zeta):
    self.fix_random_seeds()

    log_alpha = tf.random_normal([d, k], dtype=tf.float32)

    output = l0.nn.l0_norm(log_alpha, beta, gamma, zeta)
    result, log_alpha = self.evaluate([output, log_alpha])

    # Verify the output shape
    self.assertEqual(result.shape, ())

    def expected_l0_norm(log_alpha, beta, gamma, zeta):
      def sigmoid(x):
        return 1.0 /(1.0 + np.exp(-x))
      return np.sum(sigmoid(log_alpha - beta * np.log(-gamma / zeta)))

    # Calculate the expected result and compare
    expected_result = expected_l0_norm(log_alpha, beta, gamma, zeta)
    self.assertAllClose(result, expected_result)


if __name__ == "__main__":
  tf.test.main()
