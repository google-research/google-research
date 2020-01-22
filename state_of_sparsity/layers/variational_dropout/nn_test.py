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

"""Tests for variational dropout layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import absl.testing.parameterized as parameterized
import numpy as np
import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.variational_dropout as vd


# Parameters to test the matmul primitive on. First dimensions of
# first matrix, second dimension of first matrix/first dimension
# of second matrix, second dimension of second matrix.
MATMUL_TEST_PARAMETERS = [(32, 200, 100)]


@parameterized.parameters(MATMUL_TEST_PARAMETERS)
class MatmulTest(vd.test_base.TestCase):

  def testMatmulTrain(self, m, n, k):
    self.assertSameResult(
        self.set_no_epsilon(vd.nn.matmul_train),
        tf.matmul,
        [m, n],
        [n, k])

  def testMatmulTrain_NonDeterministic(self, m, n, k):
    self.assertNonDeterministic(
        vd.nn.matmul_train,
        [m, n],
        [n, k])

  def testMatmulEval(self, m, n, k):
    self.assertSameResult(
        self.set_no_epsilon(vd.nn.matmul_eval),
        tf.matmul,
        [m, n],
        [n, k])

  def testMatmulEval_Deterministic(self, m, n, k):
    self.assertDeterministic(
        vd.nn.matmul_eval,
        [m, n],
        [n, k])


# Parameters to test the batched matmul primitive on. First dimension
# of the first matrix, second dimension of the first matrix, third
# dimension of the first matrix/first dimenions of the second matrix,
# second dimension of the second matrix.
BROADCAST_MATMUL_TEST_PARAMETERS = [(32, 20, 200, 100),
                                    (1, 10, 100, 50)]


@parameterized.parameters(BROADCAST_MATMUL_TEST_PARAMETERS)
class BroadcastMatmulTest(vd.test_base.TestCase):

  def set_axes(self, ref_op):
    return functools.partial(ref_op, axes=[[2], [0]])

  def testBroadcastMatmulTrain(self, m, t, n, k):
    self.assertSameResult(
        self.set_no_epsilon(vd.nn.broadcast_matmul_train),
        self.set_axes(tf.tensordot),
        [m, t, n],
        [n, k])

  def testBroadcastMatmulTrain_NonDeterministic(self, m, t, n, k):
    self.assertNonDeterministic(
        vd.nn.broadcast_matmul_train,
        [m, t, n],
        [n, k])

  def testBroadcastMatmulEval(self, m, t, n, k):
    self.assertSameResult(
        self.set_no_epsilon(vd.nn.broadcast_matmul_eval),
        self.set_axes(tf.tensordot),
        [m, t, n],
        [n, k])

  def testBroadcastMatmulEval_Deterministic(self, m, t, n, k):
    self.assertDeterministic(
        vd.nn.broadcast_matmul_eval,
        [m, t, n],
        [n, k])


# Parameters to test the conv2d primitive with. Input tensor batch size,
# input channels, input height, input width, size of the convolutional
# filters, number of output channels.
CONV2D_TEST_PARAMETERS = [(32, 3, 224, 224, 3, 64)]


@parameterized.parameters(CONV2D_TEST_PARAMETERS)
class Conv2dTest(vd.test_base.TestCase):

  def testConv2dTrain(
      self,
      batch_size,
      in_channels,
      height,
      width,
      filter_size,
      out_channels):
    conv2d_train = self.set_no_epsilon(vd.nn.conv2d_train)
    self.assertSameResult(
        self.fix_padding_and_strides(conv2d_train),
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
        self.fix_padding_and_strides(vd.nn.conv2d_train),
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
    conv2d_eval = self.set_no_epsilon(vd.nn.conv2d_eval)
    self.assertSameResult(
        self.fix_padding_and_strides(conv2d_eval),
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
        self.fix_padding_and_strides(vd.nn.conv2d_eval),
        [batch_size, height, width, in_channels],
        [filter_size, filter_size, in_channels, out_channels])


# Parameters for the embedding lookup tests. Batch size, sequence length,
# vocabulary size, embedding vector size
EMBEDDING_TEST_PARAMETERS = [(32, 25, 10000, 512)]


@parameterized.parameters(EMBEDDING_TEST_PARAMETERS)
class TestEmbeddingLookup(vd.test_base.TestCase):

  def testEmbeddingLookupTrain(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    embedding_lookup_train = self.set_no_epsilon(vd.nn.embedding_lookup_train)
    self.assertSameResult(
        self.flip_input_wrapper(embedding_lookup_train),
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
        self.flip_input_wrapper(vd.nn.embedding_lookup_train),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)

  def testEmbeddingLookupEval(
      self,
      batch_size,
      seq_length,
      vocab_size,
      embedding_size):
    embedding_lookup_eval = self.set_no_epsilon(vd.nn.embedding_lookup_eval)
    self.assertSameResult(
        self.flip_input_wrapper(embedding_lookup_eval),
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
        self.flip_input_wrapper(vd.nn.embedding_lookup_eval),
        [batch_size, seq_length, 1],
        [vocab_size, embedding_size],
        data_dtype=tf.int32)


# Dimensions of the parameters to calculate the KL divergence over.
DKL_TEST_PARAMETERS = [(256, 128)]


@parameterized.parameters(DKL_TEST_PARAMETERS)
class TestNegativeDKL(vd.test_base.TestCase):

  def testNegativeDKL(self, d, k):
    self.fix_random_seeds()

    theta = tf.random_normal([d, k], dtype=tf.float32)
    log_sigma2 = tf.random_normal([d, k], dtype=tf.float32)
    weights = (theta, log_sigma2)

    output = vd.nn.negative_dkl(weights)

    result, theta, log_sigma2 = self.evaluate([output, theta, log_sigma2])

    # Verify the output shape
    self.assertEqual(result.shape, ())

    # Compute the expected results
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -k1

    # Compute the log alpha values
    log_alpha = log_sigma2 - np.log(np.power(theta, 2) + 1e-8)

    def sigmoid(x):
      return 1.0 /(1.0 + np.exp(-x))

    term_1 = k1 * sigmoid(k2 + k3*log_alpha)
    term_2 = -0.5 * np.log1p(np.exp(-log_alpha))
    expected_result = -np.sum(term_1 + term_2 + c)

    self.assertAllClose(result, expected_result)


if __name__ == "__main__":
  tf.test.main()
