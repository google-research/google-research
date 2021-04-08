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

"""Tests for the key functions in compression library."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow.compat.v1 as tf
from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import compression_wrapper
from graph_compression.compression_lib.keras_layers import layers as compression_layers

tf.enable_v2_behavior()


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
    with tf.Graph().as_default():
      with self.cached_session() as sess:
        compression_hparams = ("name=cifar10_compression,"
                               "begin_compression_step=1000,"
                               "end_compression_step=120000,"
                               "compression_frequency=10,"
                               "compression_option=1,"
                               "update_option=0")
        global_step = tf.compat.v1.get_variable("global_step", initializer=30)
        c = compression_op.CompressionOp(
            spec=compression_op.CompressionOp.get_default_hparams().parse(
                compression_hparams),
            global_step=global_step)
        # Need to add initial value for a_matrix so that we would know what
        # to expect back.
        a_matrix_init = np.array([[1.0, 1.0, 1.0], [1.0, 0, 0], [1.0, 0, 0]])
        a_matrix = tf.compat.v1.get_variable(
            "a_matrix",
            initializer=a_matrix_init.astype(np.float32),
            dtype=tf.float32)
        matrix_compressor = compression_op.LowRankDecompMatrixCompressor(
            spec=compression_op.LowRankDecompMatrixCompressor
            .get_default_hparams().parse("num_rows=3,num_cols=3,rank=200"))

        [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
            a_matrix, matrix_compressor, scope="my_scope")

        tf.compat.v1.global_variables_initializer().run()
        self.assertAllEqual(
            np.all(np.abs(np.linalg.norm(c.a_matrix_tfvar.eval())) < 0.00001),
            False)
        self.assertAllEqual(
            np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
            True)
        self.assertAllEqual(
            np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
            True)

        tf.compat.v1.assign(global_step, 1001).eval()
        sess.run(a_matrix_update_op)
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

        print("before 1002 step, c.alpha is ", c.alpha.eval())
        tf.compat.v1.assign(global_step, 1001).eval()
        sess.run(a_matrix_update_op)
        a_matrix_compressed.eval()
        print("after 1002 step, c.alpha is ", c.alpha.eval())
        self.assertEqual(c._global_step.eval(), 1001)
        self.assertAlmostEqual(c.alpha.eval(), 0.99)
        self.assertEqual(c._last_alpha_update_step.eval(), 1001)
        self.assertAllEqual(
            np.all([
                np.linalg.norm(c.a_matrix_tfvar.eval()),
                np.linalg.norm(c.b_matrix_tfvar.eval()),
                np.linalg.norm(c.c_matrix_tfvar.eval())
            ]) > 0, True)

        print("before 2000 step, alpha is ", c.alpha.eval())
        tf.compat.v1.assign(global_step, 2000).eval()
        a_matrix_update_op.eval()
        a_matrix_compressed.eval()
        print("after 2000 step, alpha is ", c.alpha.eval())
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
    with tf.Graph().as_default():
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

        global_step = tf.compat.v1.get_variable("global_step", initializer=30)

        apply_comp = compression_op.ApplyCompression(
            scope="default_scope",
            compression_spec=compression_op_spec,
            compressor=matrix_compressor,
            global_step=global_step)
        # Need to add initial value for a_matrix so that we would know what
        # to expect back.
        a_matrix_init = np.outer(np.array([1., 2., 3.]), np.array([4., 5., 6.]))
        a_matrix = tf.compat.v1.get_variable(
            "a_matrix",
            initializer=a_matrix_init.astype(np.float32),
            dtype=tf.float32)
        a_matrix_compressed = apply_comp.apply_compression(
            a_matrix, scope="first_compressor")
        c = apply_comp._compression_ops[0]

        a_matrix2 = tf.compat.v1.get_variable(
            "a_matrix2",
            initializer=a_matrix_init.astype(np.float32),
            dtype=tf.float32)
        _ = apply_comp.apply_compression(a_matrix2, scope="second_compressor")
        c2 = apply_comp._compression_ops[1]

        _ = apply_comp.all_update_op()

        tf.compat.v1.global_variables_initializer().run()
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
        tf.compat.v1.assign(global_step, 1001).eval()
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

        tf.compat.v1.assign(global_step, 1001).eval()
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

        tf.compat.v1.assign(global_step, 2001).eval()
        apply_comp._all_update_op.run()
        _ = a_matrix_compressed.eval()
        self.assertEqual(c._global_step.eval(), 2001)
        self.assertAlmostEqual(c.alpha.eval(), 0.98)
        self.assertAlmostEqual(c2.alpha.eval(), 0.98)
        self.assertEqual(c._last_alpha_update_step.eval(), 2001)
        self.assertAllEqual(
            np.array([
                np.linalg.norm(c.a_matrix_tfvar.eval()),
                np.linalg.norm(c.b_matrix_tfvar.eval()),
                np.linalg.norm(c.c_matrix_tfvar.eval())
            ]) > 0, [True, True, True])


class InputOutputCompressionOpTest(tf.test.TestCase):

  def test_get_apply_matmul(self):
    with tf.Graph().as_default():
      with self.cached_session():
        hparams = ("name=input_output_compression,"
                   "compression_option=9,"
                   "begin_compression_step=1000,"
                   "end_compression_step=120000,"
                   "compression_frequency=100,"
                   "compress_input=True,"
                   "compress_output=True,"
                   "input_compression_factor=2,"
                   "input_block_size=4,"
                   "output_compression_factor=2,"
                   "output_block_size=4,")
        compression_op_spec = (
            compression_op.InputOutputCompressionOp.get_default_hparams().parse(
                hparams))

        compressor_spec = (
            compression_op.LowRankDecompMatrixCompressor.get_default_hparams())
        matrix_compressor = compression_op.LowRankDecompMatrixCompressor(
            spec=compressor_spec)

        global_step = tf.compat.v1.get_variable("global_step", initializer=100)
        apply_comp = compression_op.ApplyCompression(
            scope="default_scope",
            compression_spec=compression_op_spec,
            compressor=matrix_compressor,
            global_step=global_step)

        # outer product - creates an 12x8 matrix
        a_matrix_init = np.outer(
            np.array([1., 2., 3., 7., 8., 9., 1., 2., 5., -2., -7., -1.]),
            np.array([4., 5., 6., 3., 1., 8., 3., 2.]))
        a_matrix = tf.compat.v1.get_variable(
            "a_matrix",
            initializer=a_matrix_init.astype(np.float32),
            dtype=tf.float32)
        _ = apply_comp.apply_compression(
            a_matrix, scope="compressor")
        # input is 1x12 vector
        left_operand_init = np.array(
            [1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.])
        left_operand = tf.compat.v1.get_variable(
            "left_operand",
            initializer=left_operand_init.astype(np.float32),
            dtype=tf.float32)
        c = apply_comp._compression_ops[-1]
        tf.compat.v1.global_variables_initializer().run()
        compressed_matmul = c.get_apply_matmul(left_operand)
        # check b, c and d matrices have the right shapes
        self.assertSequenceEqual(list(c.b_matrix_tfvar.eval().shape), [4, 2])
        self.assertSequenceEqual(list(c.c_matrix_tfvar.eval().shape), [6, 4])
        self.assertSequenceEqual(list(c.d_matrix_tfvar.eval().shape), [2, 4])

        # check that we get the expected output shape
        self.assertSequenceEqual(list(compressed_matmul.eval().shape), [8,])


class BlockCompressionOpTest(tf.test.TestCase):

  def test_get_apply_matmul(self):
    with tf.Graph().as_default():
      with self.cached_session():
        hparams = ("name=block_compression,"
                   "compression_option=10,"
                   "begin_compression_step=1000,"
                   "end_compression_step=120000,"
                   "compression_frequency=100,"
                   "block_method=mask,"
                   "block_compression_factor=2,")
        compression_op_spec = (
            compression_op.BlockCompressionOp.get_default_hparams().parse(
                hparams))

        compressor_spec = (
            compression_op.LowRankDecompMatrixCompressor.get_default_hparams())
        matrix_compressor = compression_op.LowRankDecompMatrixCompressor(
            spec=compressor_spec)

        global_step = tf.compat.v1.get_variable("global_step", initializer=100)
        apply_comp = compression_op.ApplyCompression(
            scope="default_scope",
            compression_spec=compression_op_spec,
            compressor=matrix_compressor,
            global_step=global_step)

        # outer product - creates an 12x8 matrix
        a_matrix_init = np.outer(
            np.array([1., 2., 3., 7., 8., 9., 1., 2., 5., -2., -7., -1.]),
            np.array([4., 5., 6., 3., 1., 8., 3., 2.]))
        a_matrix = tf.compat.v1.get_variable(
            "a_matrix",
            initializer=a_matrix_init.astype(np.float32),
            dtype=tf.float32)
        _ = apply_comp.apply_compression(
            a_matrix, scope="compressor")
        # input is 1x12 vector
        left_operand_init = np.array(
            [1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.])
        left_operand = tf.compat.v1.get_variable(
            "left_operand",
            initializer=left_operand_init.astype(np.float32),
            dtype=tf.float32)
        c = apply_comp._compression_ops[-1]
        tf.compat.v1.global_variables_initializer().run()
        compressed_matmul = c.get_apply_matmul(left_operand)
        # check c, c_mask matrices have the right shapes
        self.assertSequenceEqual(list(c.c_matrix_tfvar.eval().shape), [12, 8])
        self.assertSequenceEqual(list(c.c_mask_tfvar.eval().shape), [12, 8])
        # check we get the correct number of nonzero entries in the mask
        self.assertEqual(np.count_nonzero(c.c_mask_tfvar.eval()), 48)
        # check that we get the expected output shape
        self.assertSequenceEqual(list(compressed_matmul.eval().shape), [8,])


class CompressionLayersTest(tf.test.TestCase):

  def testCompressedDenseLayer(self):
    hparams = ("name=mnist_compression,"
               "compress_input=True,"
               "input_block_size=16,"
               "input_compression_factor=4,"
               "compression_option=9")

    compression_hparams = compression_op.InputOutputCompressionOp.get_default_hparams(
    ).parse(hparams)
    # compression_hparams = pruning.get_pruning_hparams().parse(hparams)
    # Create a compression object using the compression hyperparameters
    compression_obj = compression_wrapper.get_apply_compression(
        compression_hparams, global_step=0)
    val = np.random.random((10, 48))
    x = tf.Variable(val, dtype=tf.float32)
    y_compressed = compression_layers.CompressedDense(
        20, compression_obj=compression_obj)(x)
    y = tf.keras.layers.Dense(
        20)(x)

    self.assertAllEqual(y.shape.as_list(), y_compressed.shape.as_list())

  def testCompressedConv2DLayer(self):
    hparams = ("name=mnist_compression,"
               "compress_input=True,"
               "input_block_size=16,"
               "input_compression_factor=2,"
               "compression_option=9")

    compression_hparams = compression_op.InputOutputCompressionOp.get_default_hparams(
    ).parse(hparams)
    compression_obj = compression_wrapper.get_apply_compression(
        compression_hparams, global_step=0)

    val = np.random.random((10, 4, 10, 10))
    x = tf.Variable(val, dtype=tf.float32)
    y_compressed = compression_layers.CompressedConv2D(
        20, 3, padding="valid", data_format="channels_last",
        compression_obj=compression_obj)(x)
    y = tf.keras.layers.Conv2D(
        20, 3, padding="valid", data_format="channels_last")(x)

    self.assertAllEqual(y.shape.as_list(), y_compressed.shape.as_list())

if __name__ == "__main__":
  tf.test.main()
