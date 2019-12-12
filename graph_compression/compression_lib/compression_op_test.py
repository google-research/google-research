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

"""Tests for the key functions in compression library."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from graph_compression.compression_lib import compression_op


class CompressionOpInterfaceTest(tf.test.TestCase):

  def testLowRankDecompMatrixCompressorInterface(self):
    spec = compression_op.LowRankDecompMatrixCompressor.get_default_hparams()
    compressor = compression_op.LowRankDecompMatrixCompressor(spec)
    b_matrix = np.random.normal(0, 1, [10, 5])
    c_matrix = np.random.normal(0, 1, [5, 10])
    a_matrix = np.matmul(b_matrix, c_matrix)
    [b_matrix_out, c_matrix_out] = compressor.static_matrix_compressor(a_matrix)
    a_matrix_recovered = np.matmul(b_matrix_out, c_matrix_out)
    self.assertLess(np.linalg.norm(a_matrix - a_matrix_recovered), 0.01)

  def testCompressionOpInterface(self):
    with self.cached_session():
      compression_hparams = ("name=cifar10_compression,"
                             "begin_compression_step=1000,"
                             "end_compression_step=120000,"
                             "compression_frequency=100,"
                             "compression_option=1")
      global_step = tf.get_variable("global_step", initializer=30)
      c = compression_op.CompressionOp(
          spec=compression_op.CompressionOp.get_default_hparams().parse(
              compression_hparams),
          global_step=global_step)
      # Need to add initial value for a_matrix so that we would know what
      # to expect back.
      a_matrix_init = np.array([[1.0, 1.0, 1.0], [1.0, 0, 0], [1.0, 0, 0]])
      a_matrix = tf.get_variable(
          "a_matrix",
          initializer=a_matrix_init.astype(np.float32),
          dtype=tf.float32)
      matrix_compressor = compression_op.LowRankDecompMatrixCompressor(
          spec=compression_op.LowRankDecompMatrixCompressor.get_default_hparams(
          ).parse("num_rows=3,num_cols=3,rank=200"))

      [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
          a_matrix, matrix_compressor, scope="my_scope")

      tf.global_variables_initializer().run()
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.a_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          True)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          True)
      tf.assign(global_step, 1001).eval()
      a_matrix_update_op.eval()
      a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 1001)
      self.assertAlmostEqual(c.alpha.eval(), 0.99)
      self.assertEqual(c._last_alpha_update_step.eval(), 1001)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, True, True])

      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          False)

      [b_matrix,
       c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix_init)
      # since the matrices may match up to signs, we take absolute values.
      self.assertAllEqual(
          np.linalg.norm(np.abs(b_matrix) - np.abs(c.b_matrix_tfvar.eval())) <
          0.00001, True)
      self.assertAllEqual(
          np.linalg.norm(np.abs(c_matrix) - np.abs(c.c_matrix_tfvar.eval())) <
          0.00001, True)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          False)

      tf.assign(global_step, 1001).eval()
      a_matrix_update_op.eval()
      a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 1001)
      self.assertAlmostEqual(c.alpha.eval(), 0.99)
      self.assertEqual(c._last_alpha_update_step.eval(), 1001)
      self.assertAllEqual(
          np.all([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, True)

      tf.assign(global_step, 2000).eval()
      a_matrix_update_op.eval()
      a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 2000)
      self.assertAlmostEqual(c.alpha.eval(), 0.98)
      self.assertEqual(c._last_alpha_update_step.eval(), 2000)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, True, True])

  def testApplyCompression(self):
    with self.cached_session():
      compression_hparams = ("name=cifar10_compression,"
                             "begin_compression_step=1000,"
                             "end_compression_step=120000,"
                             "compression_frequency=100,"
                             "compression_option=1")
      compression_op_spec = (
          compression_op.CompressionOp.get_default_hparams().parse(
              compression_hparams))
      compressor_spec = (
          compression_op.LowRankDecompMatrixCompressor.get_default_hparams()
          .parse("num_rows=5,num_cols=5,rank=200"))
      matrix_compressor = compression_op.LowRankDecompMatrixCompressor(
          spec=compressor_spec)

      global_step = tf.get_variable("global_step", initializer=30)

      apply_comp = compression_op.ApplyCompression(
          scope="default_scope",
          compression_spec=compression_op_spec,
          compressor=matrix_compressor,
          global_step=global_step)
      # Need to add initial value for a_matrix so that we would know what
      # to expect back.
      a_matrix_init = np.outer(np.array([1., 2., 3.]), np.array([4., 5., 6.]))
      a_matrix = tf.get_variable(
          "a_matrix",
          initializer=a_matrix_init.astype(np.float32),
          dtype=tf.float32)
      a_matrix_compressed = apply_comp.apply_compression(
          a_matrix, scope="first_compressor")
      c = apply_comp._compression_ops[0]

      a_matrix2 = tf.get_variable(
          "a_matrix2",
          initializer=a_matrix_init.astype(np.float32),
          dtype=tf.float32)
      _ = apply_comp.apply_compression(a_matrix2, scope="second_compressor")
      c2 = apply_comp._compression_ops[1]

      _ = apply_comp.all_update_op()

      tf.global_variables_initializer().run()
      _ = a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 30)
      self.assertEqual(c.alpha.eval(), 1.0)
      self.assertEqual(c2.alpha.eval(), 1.0)
      self.assertEqual(c._last_alpha_update_step.eval(), -1)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, False, False])

      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.a_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          True)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          True)
      tf.assign(global_step, 1001).eval()
      # apply_comp_update_op.run()
      apply_comp._all_update_op.run()
      _ = a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 1001)
      self.assertAlmostEqual(c.alpha.eval(), 0.99)
      self.assertEqual(c._last_alpha_update_step.eval(), 1001)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, True, True])
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          False)

      [b_matrix,
       c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix_init)

      self.assertAllEqual(
          np.linalg.norm(np.abs(b_matrix) - np.abs(c.b_matrix_tfvar.eval())) <
          0.00001, True)
      self.assertAllEqual(
          np.linalg.norm(np.abs(c_matrix) - np.abs(c.c_matrix_tfvar.eval())) <
          0.00001, True)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
          False)
      self.assertAllEqual(
          np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
          False)

      tf.assign(global_step, 1001).eval()
      apply_comp._all_update_op.run()
      _ = a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 1001)
      self.assertAlmostEqual(c.alpha.eval(), 0.99)
      self.assertEqual(c._last_alpha_update_step.eval(), 1001)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, True, True])

      tf.assign(global_step, 2000).eval()
      apply_comp._all_update_op.run()
      _ = a_matrix_compressed.eval()
      self.assertEqual(c._global_step.eval(), 2000)
      self.assertAlmostEqual(c.alpha.eval(), 0.98)
      self.assertAlmostEqual(c2.alpha.eval(), 0.98)
      self.assertEqual(c._last_alpha_update_step.eval(), 2000)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
              np.linalg.norm(c.c_matrix_tfvar.eval())
          ]) > 0, [True, True, True])


if __name__ == "__main__":
  tf.test.main()
