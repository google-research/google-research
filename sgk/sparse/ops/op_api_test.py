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

"""Tests for sparse operations."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from sgk.sparse import connectors
from sgk.sparse import initializers
from sgk.sparse import ops
from sgk.sparse import sparse_matrix
from sgk.sparse.ops import op_test

# TODO(tgale): Add more test cases once the generic spmm & sddmm kernel
# APIs relax the 4-element alignment constraint.
_BINARY_ARGUMENTS = [(4, 4, 4, 0.0, True), (4, 4, 4, 0.0, False),
                     (512, 512, 512, 0.8, True), (512, 512, 512, 0.8, False),
                     (1024, 4096, 256, 0.75, True),
                     (1024, 4096, 256, 0.8, False)]

# NOTE: Gradient tests are extremely slow, as the kernel must be run many
# times to estimate the jacobian. Only run smaller test sizes.
_BINARY_GRADIENT_ARGUMENTS = [(4, 4, 4, 0.0, True), (4, 4, 4, 0.0, False),
                              (32, 32, 32, .9, True), (32, 32, 32, .9, False)]

_UNARY_ARGUMENTS = [
    (4, 4, 0.0, True),
    (4, 4, 0.0, False),
    (128, 64, 0.5, True),
    (128, 64, 0.5, False),
    (319, 47, 0.75, True),
    (319, 47, 0.75, False),
]

# Disable TF2.
tf.disable_v2_behavior()


class SpmmTest(op_test.TestCase):

  @parameterized.parameters(*_BINARY_ARGUMENTS)
  def testSpmm(self, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = connector(initializer([m, k]))
    rhs_np = initializer([k, n])

    # TensorFlow graph.
    lhs = sparse_matrix.SparseMatrix("lhs", matrix=lhs_np)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.spmm(lhs, rhs)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(
          sess.run(output), np.dot(lhs_np, rhs_np), atol=1e-03, rtol=1e-05)

  @parameterized.parameters(*_BINARY_GRADIENT_ARGUMENTS)
  def testSpmmGradient(self, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = connector(initializer([m, k]))
    rhs_np = initializer([k, n])

    lhs = sparse_matrix.SparseMatrix("lhs", matrix=lhs_np)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.spmm(lhs, rhs)

    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      error = tf.test.compute_gradient_error(
          [lhs.values, rhs], [lhs.values.shape.as_list(), [k, n]], output,
          [m, n])
      self.assertLess(error, 1e-3)

  @parameterized.parameters((2, 4, 4, 4, 0.0, True), (2, 4, 4, 4, 0.0, False),
                            (8, 512, 512, 512, 0.8, True),
                            (8, 512, 512, 512, 0.8, False))
  def testSpmm_Replicated(self, r, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    mask = connector(initializer([m, k]))
    mask[mask != 0] = 1.0
    lhs_np = np.expand_dims(mask, axis=0) * initializer([r, m, k])
    rhs_np = initializer([r, k, n])

    # TensorFlow graph.
    topology = sparse_matrix.SparseTopology("topology", mask=mask)
    lhs = tf.Variable(
        np.reshape(lhs_np[lhs_np != 0], [r, -1]), dtype=tf.float32)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.replicated_spmm(lhs, topology, rhs)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      out = sess.run(output)
      for i in range(r):
        expected_out = np.dot(lhs_np[i, :, :], rhs_np[i, :, :])
        self.assertAllClose(out[i, :], expected_out, atol=1e-03, rtol=1e-05)

  @parameterized.parameters(*_BINARY_ARGUMENTS)
  def testSpmm_Fused(self, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = connector(initializer([m, k]))
    rhs_np = initializer([k, n])
    bias_np = np.random.uniform(size=m)

    # TensorFlow graph.
    lhs = sparse_matrix.SparseMatrix("lhs", matrix=lhs_np)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    bias = tf.Variable(bias_np, dtype=tf.float32)
    output = ops.fused_spmm(lhs, rhs, bias)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(
          sess.run(output),
          np.dot(lhs_np, rhs_np) + np.expand_dims(bias_np, axis=1),
          atol=1e-03,
          rtol=1e-05)


class SddmmTest(op_test.TestCase):

  @parameterized.parameters(*_BINARY_ARGUMENTS)
  def testSddmm(self, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([m, k])
    rhs_np = initializer([n, k])
    output_np = connector(np.ones([m, n]))

    # TensorFlow graph.
    output_topology = sparse_matrix.SparseMatrix("output", matrix=output_np)
    lhs = tf.Variable(lhs_np, dtype=tf.float32)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.sddmm(lhs, rhs, output_topology, transpose_rhs=True)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      expected_output = self.dense_to_scipy(
          output_np * np.dot(lhs_np, np.transpose(rhs_np)))
      actual_output = self.sparse_to_scipy(
          *sess.run([output.values, output.row_offsets, output.column_indices]),
          shape=expected_output.shape)

      self.assert_sparse_matrix_equal(
          actual_output, expected_output, atol=1e-03, rtol=1e-05)

  @parameterized.parameters((2, 4, 4, 4, 0.0, True), (2, 4, 4, 4, 0.0, False),
                            (8, 512, 512, 512, 0.8, True),
                            (8, 512, 512, 512, 0.8, False))
  def testSddmm_Replicated(self, r, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([r, m, k])
    rhs_np = initializer([r, n, k])
    output_np = connector(np.ones([m, n]))

    # TensorFlow graph.
    output_topology = sparse_matrix.SparseTopology(
        "output_topology", mask=output_np)
    lhs = tf.Variable(lhs_np, dtype=tf.float32)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.replicated_sddmm(lhs, rhs, output_topology, transpose_rhs=True)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())

      # Run the replicated sddmm.
      v, ro, ci = sess.run(
          [output, output_topology.row_offsets, output_topology.column_indices])

      for i in range(r):
        expected_output = self.dense_to_scipy(
            output_np * np.dot(lhs_np[i, :, :], np.transpose(rhs_np[i, :, :])))
        actual_output = self.sparse_to_scipy(
            v[i, :], ro, ci, shape=expected_output.shape)
        self.assert_sparse_matrix_equal(
            actual_output, expected_output, atol=1e-03, rtol=1e-05)

  @parameterized.parameters(*_BINARY_GRADIENT_ARGUMENTS)
  def testSddmmGradient(self, m, k, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([m, k])
    rhs_np = initializer([n, k])
    output_np = connector(np.ones([m, n]))

    # TensorFlow graph.
    output_topology = sparse_matrix.SparseMatrix("output", matrix=output_np)
    lhs = tf.Variable(lhs_np, dtype=tf.float32)
    rhs = tf.Variable(rhs_np, dtype=tf.float32)
    output = ops.sddmm(lhs, rhs, output_topology, transpose_rhs=True)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      error = tf.test.compute_gradient_error([lhs, rhs], [[m, k], [n, k]],
                                             output.values,
                                             output.values.shape.as_list())
      self.assertLess(error, 1e-3)


@parameterized.parameters(*_UNARY_ARGUMENTS)
class TransposeTest(op_test.TestCase):

  def testTranspose(self, m, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrix for verification.
    matrix_np = connector(initializer([m, n]))

    # TensorFlow graph.
    matrix = sparse_matrix.SparseMatrix("input", matrix=matrix_np)
    output = ops.transpose(matrix)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())
      expected_output = self.dense_to_scipy(np.transpose(matrix_np))
      actual_output = self.sparse_to_scipy(
          *sess.run([output.values, output.row_offsets, output.column_indices]),
          shape=expected_output.shape)

      self.assert_sparse_matrix_equal(
          actual_output, expected_output, atol=1e-03, rtol=1e-05)


@parameterized.parameters(*_UNARY_ARGUMENTS)
class Csr2IdxTest(op_test.TestCase):

  def testCsr2Idx(self, m, n, sparsity, force_gpu):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrix for verification.
    matrix_np = connector(initializer([m, n]))

    # TensorFlow graph.
    matrix = sparse_matrix.SparseMatrix("input", matrix=matrix_np)
    output = ops.csr2idx(matrix)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=force_gpu) as sess:
      sess.run(tf.global_variables_initializer())

      # Calculate the linear indices in numpy.
      x = self.dense_to_scipy(matrix_np)
      expected_output = np.concatenate(
          [x.indices[x.indptr[i]:x.indptr[i + 1]] + i * n for i in range(m)])
      self.assertAllEqual(sess.run(output), expected_output)


class SparseSoftmax(op_test.TestCase):

  @parameterized.parameters((4, 5, 0.0), (128, 256, 0.5), (1024, 4096, 0.85))
  def testSparseSoftmax(self, m, n, sparsity):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrix for verification.
    matrix_np = connector(initializer([m, n]))

    # TensorFlow graph.
    matrix = sparse_matrix.SparseMatrix("input", matrix=matrix_np)
    output = ops.sparse_softmax(matrix)

    with self.test_session(force_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())

      # Zero terms should not contribute to the softmax.
      matrix_np[matrix_np == 0] = -1e9

      def softmax(x):
        maxs = np.expand_dims(x.max(axis=1), axis=1)
        exps = np.exp(x - maxs)
        return exps / np.expand_dims(np.sum(exps, axis=1), axis=1)

      expected_output = self.dense_to_scipy(softmax(matrix_np))

      actual_output = self.sparse_to_scipy(
          *sess.run([output.values, output.row_offsets, output.column_indices]),
          expected_output.shape)

      self.assert_sparse_matrix_equal(
          actual_output, expected_output, atol=1e-03, rtol=1e-05)

  @parameterized.parameters((2, 4, 5, 0.0), (8, 128, 256, 0.5),
                            (16, 1024, 4096, 0.85))
  def testSparseSoftmax_Replicated(self, r, m, n, sparsity):
    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrix for verification.
    mask = connector(np.ones([m, n]))
    matrix_np = np.expand_dims(mask, axis=0) * initializer([r, m, n])

    # TensorFlow graph.
    topology = sparse_matrix.SparseTopology("topology", mask=mask)
    values = tf.Variable(
        np.reshape(matrix_np[matrix_np != 0], [r, -1]), dtype=tf.float32)
    output = ops.replicated_sparse_softmax(values, topology)

    with self.test_session(force_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      v, ro, ci = sess.run(
          [output, topology.row_offsets, topology.column_indices])

      # Zero terms should not contribute to the softmax.
      matrix_np[matrix_np == 0] = -1e9

      def softmax(x):
        maxs = np.expand_dims(x.max(axis=1), axis=1)
        exps = np.exp(x - maxs)
        return exps / np.expand_dims(np.sum(exps, axis=1), axis=1)

      for i in range(r):
        expected_output = self.dense_to_scipy(softmax(matrix_np[i, :, :]))

        actual_output = self.sparse_to_scipy(v[i, :], ro, ci,
                                             expected_output.shape)
        self.assert_sparse_matrix_equal(
            actual_output, expected_output, atol=1e-03, rtol=1e-05)


class FusedSoftmax(op_test.TestCase):

  @parameterized.parameters((4, 5), (128, 64), (1103, 971))
  def testFusedSoftmax(self, m, n):
    # Helpers to set up the matrices.
    initializer = initializers.Uniform()

    # Numpy matrix for verification.
    shape = [m, n]
    matrix_np = initializer(shape)

    # TensorFlow graph.
    matrix = tf.Variable(matrix_np, dtype=tf.float32)
    output = ops.fused_softmax(matrix)

    with self.test_session(force_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())

      def softmax(x):
        maxs = np.expand_dims(x.max(axis=1), axis=1)
        exps = np.exp(x - maxs)
        return exps / np.expand_dims(np.sum(exps, axis=1), axis=1)

      expected_output = softmax(matrix_np)

      actual_output = sess.run(output)
      self.assertAllClose(
          actual_output, expected_output, atol=1e-03, rtol=1e-05)


class DepthwiseConv2dTest(op_test.TestCase):

  @parameterized.parameters((1, 128, 32, 32), (1, 128, 64, 64), (4, 64, 16, 16),
                            (4, 64, 128, 128))
  def testDepthwiseConv2d(self, batch_size, in_channels, height, width):

    # Input data, random weights & bias.
    inputs = tf.ones([batch_size, in_channels, height, width])
    filters = tf.get_variable("filters", [3, 3, in_channels, 1])

    # TensorFlow graph.
    filters_t = tf.transpose(filters, (2, 0, 1, 3))
    out = ops.depthwise_conv2d(
        inputs, filters_t, strides=[1, 1, 1, 1], padding=[0, 0, 1, 1])
    expected_out = tf.nn.depthwise_conv2d(
        inputs,
        filters,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NCHW")

    # Execute the op and compare the results.
    with self.test_session(force_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      out_np, expected_out_np = sess.run([out, expected_out])
      self.assertAllClose(out_np, expected_out_np)

  @parameterized.parameters((1, 128, 32, 32), (1, 128, 64, 64), (4, 64, 16, 16),
                            (4, 64, 128, 128))
  def testFusedDepthwiseConv2d(self, batch_size, in_channels, height, width):

    # Input data, random weights & bias.
    inputs = tf.ones([batch_size, in_channels, height, width])
    filters = tf.get_variable("filters", [3, 3, in_channels, 1])
    bias = tf.get_variable("bias", [in_channels])

    # TensorFlow graph.
    filters_t = tf.transpose(filters, (2, 0, 1, 3))
    out = ops.fused_depthwise_conv2d(
        inputs, filters_t, bias, strides=[1, 1, 1, 1], padding=[0, 0, 1, 1])
    expected_out = tf.nn.depthwise_conv2d(
        inputs,
        filters,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NCHW")
    expected_out = tf.nn.bias_add(expected_out, bias, data_format="NCHW")
    expected_out = tf.nn.relu(expected_out)

    # Execute the op and compare the results.
    with self.test_session(force_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      out_np, expected_out_np = sess.run([out, expected_out])
      self.assertAllClose(out_np, expected_out_np)


if __name__ == "__main__":
  tf.test.main()
