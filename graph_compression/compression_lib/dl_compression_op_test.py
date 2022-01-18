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

# Lint as: python3
"""Tests for dl_compression_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import compression_op_utils
from graph_compression.compression_lib import dl_compression_op


class DlCompressionOpTest(tf.test.TestCase):

  def test_DLDecompMatrixCompressor_interface(self):
    compressor = dl_compression_op.DLMatrixCompressor(
        spec=compression_op.LowRankDecompMatrixCompressor.get_default_hparams())
    B = np.random.normal(0, 1, [20, 10])  # pylint: disable = invalid-name
    sub_thresh_indices = (B <= 0.5)
    B[sub_thresh_indices] = 0
    B[:, 0] = np.ones(shape=B[:, 0].shape)
    C = np.random.normal(0, 1, [10, 5])  # pylint: disable = invalid-name
    A = np.matmul(B, C)  # pylint: disable = invalid-name
    [B_out, C_out] = compressor.static_matrix_compressor(A, n_iterations=32)  # pylint: disable = invalid-name
    A_recovered = np.matmul(B_out, C_out)  # pylint: disable = invalid-name
    print("np.linalg.norm(A-A_recovered) / np.linalg.norm(A): ",
          np.linalg.norm(A - A_recovered) / np.linalg.norm(A))
    print("A: ", A)
    print("A_recovered: ", A_recovered)
    print("fraction error np.linalg.norm(A-A_recovered): ",
          np.linalg.norm(A - A_recovered) / np.linalg.norm(A))
    self.assertLessEqual(
        np.linalg.norm(A - A_recovered) / np.linalg.norm(A), 0.1)

  def test_dl_compression_op_interface(self):
    with self.cached_session() as session:
      self.check_dl_compression_op_interface(session)

  def test_dl_compression_op_interface_supervisor(self):
    with tf.Session() as session:
      session.graph._unsafe_unfinalize()
      self.check_dl_compression_op_interface_sparse(session, use_dl_op=True)

  def check_dl_compression_op_interface(self, session, use_dl_op=False):
    compression_hparams = ("name=cifar10_compression," +
                           "begin_compression_step=1000," +
                           "end_compression_step=120000," +
                           "compression_frequency=100," +
                           "rank=200,")
    update_style = compression_op_utils.UpdateOptions.PYTHON_UPDATE
    global_step = tf.get_variable("global_step", initializer=30)
    compression_op_spec = compression_op.CompressionOp.get_default_hparams(
    ).parse(compression_hparams)
    compression_op_spec.set_hparam("use_tpu", False)
    compression_op_spec.set_hparam(
        "compression_option",
        compression_op_utils.CompressionOptions.DL_MATRIX_COMPRESSION)
    compression_op_spec.set_hparam(
        "update_option", compression_op_utils.UpdateOptions.PYTHON_UPDATE)
    if use_dl_op:
      CompOp = dl_compression_op.DLCompressionOp  # pylint: disable = invalid-name
    else:
      CompOp = compression_op.CompressionOp  # pylint: disable = invalid-name

    c = CompOp(spec=compression_op_spec, global_step=global_step)
    # Need to add initial value for A so that we would know what to expect back.
    code = np.random.normal(0, 1, [20, 10])
    dictionary = np.random.normal(0, 1, [10, 5])
    A_init = np.matmul(code, dictionary)  # pylint: disable = invalid-name
    A = tf.get_variable(  # pylint: disable = invalid-name
        "A",
        initializer=A_init.astype(np.float32),
        dtype=tf.float32)

    MC = dl_compression_op.DLMatrixCompressor(  # pylint: disable = invalid-name
        spec=compression_op.LowRankDecompMatrixCompressor.get_default_hparams()
        .parse("num_rows=3,num_cols=3,rank=200,is_b_matrix_trainable=False"))

    [_, A_update_op] = c.get_apply_compression_op(  # pylint: disable = invalid-name
        A, MC, scope="my_scope")

    tf.global_variables_initializer().run()
    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())

    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          np.linalg.norm(c.b_matrix_tfvar.eval()),
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.a_matrix_tfvar.eval())) < 0.00001),
        False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001), True)
    tf.assign(global_step, 1001).eval()
    print("global_step.eval is ", global_step.eval())

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())
    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          np.linalg.norm(c.b_matrix_tfvar.eval()),
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
        False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
        False)

    [B, C] = MC.static_matrix_compressor(  # pylint: disable = invalid-name
        c.a_matrix_tfvar.eval())
    print("norm of error is ", np.linalg.norm(B - c.b_matrix_tfvar.eval()))
    self.assertAllEqual(
        np.all(np.abs(B - c.b_matrix_tfvar.eval()) < 0.00001), True)
    self.assertAllEqual(
        np.all(np.abs(C - c.c_matrix_tfvar.eval()) < 0.00001), True)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001),
        False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
        False)

    tf.assign(global_step, 1001).eval()

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())
    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          np.linalg.norm(c.b_matrix_tfvar.eval()),
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    tf.assign(global_step, 2000).eval()

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())

    self.assertAlmostEqual(c.alpha.eval(), 0.97)
    self.assertEqual(c._last_alpha_update_step.eval(), 2000)

  def check_dl_compression_op_interface_sparse(self, session, use_dl_op=False):
    compression_hparams = ("name=cifar10_compression," +
                           "begin_compression_step=1000," +
                           "end_compression_step=120000," +
                           "compression_frequency=100," +
                           "rank=200,")
    update_style = compression_op_utils.UpdateOptions.PYTHON_UPDATE
    global_step = tf.get_variable("global_step", initializer=30)
    compression_op_spec = compression_op.CompressionOp.get_default_hparams(
    ).parse(compression_hparams)
    compression_op_spec.set_hparam("use_tpu", False)
    compression_op_spec.set_hparam(
        "compression_option",
        compression_op_utils.CompressionOptions.DL_MATRIX_COMPRESSION)
    compression_op_spec.set_hparam(
        "update_option", compression_op_utils.UpdateOptions.PYTHON_UPDATE)
    if use_dl_op:
      CompOp = dl_compression_op.DLCompressionOp  # pylint: disable = invalid-name
    else:
      CompOp = compression_op.CompressionOp  # pylint: disable = invalid-name

    c = CompOp(spec=compression_op_spec, global_step=global_step)
    code = np.random.normal(0, 1, [20, 10])
    dictionary = np.random.normal(0, 1, [10, 5])
    A_init = np.matmul(code, dictionary)  # pylint: disable = invalid-name
    A = tf.get_variable(  # pylint: disable = invalid-name
        "A", initializer=A_init.astype(np.float32), dtype=tf.float32)
    MC = dl_compression_op.DLMatrixCompressor(  # pylint: disable = invalid-name
        spec=compression_op.LowRankDecompMatrixCompressor.get_default_hparams(
        ).parse("num_rows=3,num_cols=3,rank=200,is_b_matrix_trainable=False"))

    [_, A_update_op] = c.get_apply_compression_op(  # pylint: disable = invalid-name
        A, MC, scope="my_scope")

    tf.global_variables_initializer().run()
    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())

    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          c.b_matrix_indices_tfvar.eval().size,
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.a_matrix_tfvar.eval())) < 0.00001),
        False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001), True)
    tf.assign(global_step, 1001).eval()
    print("global_step.eval is ", global_step.eval())

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())
    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          c.b_matrix_indices_tfvar.eval().size,
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    self.assertAllEqual(
        np.all(c.b_matrix_indices_tfvar.eval().size < 0.00001), False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
        False)

    [B, C] = MC.static_matrix_compressor(c.a_matrix_tfvar.eval())  # pylint: disable = invalid-name
    print("B, B_tfvar :", B, c.b_matrix_tfvar.eval())
    print("norm of error is ",
          np.linalg.norm(B - tf.sparse.to_dense(c.b_matrix_tfvar).eval()))
    self.assertAllEqual(
        np.all(
            np.abs(B - tf.sparse.to_dense(c.b_matrix_tfvar).eval()) < 0.00001),
        True)
    self.assertAllEqual(
        np.all(np.abs(C - c.c_matrix_tfvar.eval()) < 0.00001), True)
    self.assertAllEqual(
        np.all(c.b_matrix_indices_tfvar.eval().size < 0.00001), False)
    self.assertAllEqual(
        np.all(np.abs(np.linalg.norm(c.c_matrix_tfvar.eval())) < 0.00001),
        False)

    tf.assign(global_step, 1001).eval()

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())
    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          c.b_matrix_indices_tfvar.eval().size,
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    tf.assign(global_step, 2000).eval()

    if update_style == compression_op_utils.UpdateOptions.TF_UPDATE:
      A_update_op.eval()
    else:
      c.run_update_step(session)

    print("global_step: ", c._global_step.eval())
    print("alpha: ", c.alpha.eval())
    print("last_alpha_update_step: ", c._last_alpha_update_step.eval())
    print("A,B,C norms are : ", np.linalg.norm(c.a_matrix_tfvar.eval()),
          c.b_matrix_indices_tfvar.eval().size,
          np.linalg.norm(c.c_matrix_tfvar.eval()))

    self.assertAlmostEqual(c.alpha.eval(), 0.97)
    self.assertEqual(c._last_alpha_update_step.eval(), 2000)


if __name__ == "__main__":
  tf.test.main()
