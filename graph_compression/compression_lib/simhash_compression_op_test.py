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

"""Tests for simhash_compression_op."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import simhash_compression_op as simhash


class SimhashCompressionOpTest(tf.test.TestCase):

  def testSimhashApplyCompression(self):
    with self.cached_session():
      hparams = ("name=cifar10_compression,"
                 "begin_compression_step=1000,"
                 "end_compression_step=2001,"
                 "compression_frequency=100,"
                 "compression_option=2")
      spec = simhash.SimhashCompressionOp.get_default_hparams().parse(hparams)

      matrix_compressor = simhash.SimhashMatrixCompressor(
          spec=compression_op.LowRankDecompMatrixCompressor.get_default_hparams(
          ).parse("num_rows=5,num_cols=5,rank=200"))

      global_step = tf.get_variable("global_step", initializer=30)

      apply_comp = simhash.SimhashApplyCompression(
          scope="default_scope",
          compression_spec=spec,
          compressor=matrix_compressor,
          global_step=global_step)

      # Need to add initial value for a_matrix so that we would know what to
      # expect back.
      a_matrix_init = np.outer(np.array([1., 2., 3.]), np.array([4., 5., 6.]))
      jitter = np.tile([0, 1e-1, 2e-2], (3, 1))
      a_matrix_init += jitter

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

      # Compression won't start until step 1000 + some random_shift amount.
      # Here we make sure output is as expected at step 30.
      self.assertEqual(c._global_step.eval(), 30)
      self.assertEqual(c.alpha.eval(), 1.0)
      self.assertEqual(c2.alpha.eval(), 1.0)
      self.assertEqual(c._last_alpha_update_step.eval(), -1)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
          ]) > 0, [True, False])
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
          ]) < 0.00001, [False, True])

      # At this point compression should have already started being applied;
      # verify at step 2000 all is as expected.
      tf.assign(global_step, 2000).eval()
      apply_comp._all_update_op.run()
      _ = a_matrix_compressed.eval()

      self.assertEqual(c._global_step.eval(), 2000)
      self.assertAlmostEqual(c.alpha.eval(), 0.99)
      self.assertEqual(c._last_alpha_update_step.eval(), 2000)
      self.assertAllEqual(
          np.array([
              np.linalg.norm(c.a_matrix_tfvar.eval()),
              np.linalg.norm(c.b_matrix_tfvar.eval()),
          ]) > 0, [True, True])
      self.assertFalse(
          np.all(np.abs(np.linalg.norm(c.b_matrix_tfvar.eval())) < 0.00001))

      # The static_matrix_compressor was configured with a rank spec of 200 --
      # meaning compression by half, i.e. new_rank = orig_rank / 2.
      self.assertEqual(
          np.linalg.matrix_rank(c.b_matrix_tfvar.eval()),
          np.linalg.matrix_rank(c.a_matrix_tfvar.eval()) / 2)

      b_matrix = matrix_compressor.static_matrix_compressor(a_matrix_init)
      self.assertAllEqual(
          np.linalg.norm(np.abs(b_matrix) - np.abs(c.b_matrix_tfvar.eval())) <
          0.00001, True)


if __name__ == "__main__":
  tf.test.main()
