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

"""Tests for non_semantic_speech_benchmark.distillation.layers."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import layers
from non_semantic_speech_benchmark.distillation.compression_lib import compression_op as compression
from non_semantic_speech_benchmark.distillation.compression_lib import compression_wrapper


class ModelsTest(parameterized.TestCase):

  def test_compressed_dense_inference(self):
    """Verify forward pass and removal of the uncompressed kernel."""
    compression_params = compression.CompressionOp \
      .get_default_hparams().parse("")
    compressor = compression_wrapper.get_apply_compression(
        compression_params, global_step=0)

    in_tensor = tf.zeros((1, 5), dtype=tf.float32)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((5,)),
        layers.CompressedDense(
            10, compression_obj=compressor, name="compressed")
    ])

    # remove uncompressed kernel
    m.get_layer("compressed").kernel = None
    m.get_layer("compressed").compression_op.a_matrix_tfvar = None

    out = m(in_tensor, training=False)
    self.assertEqual(out.shape[1], 10)

  def test_compressed_dense_training_failure(self):
    """Verify forward pass fails when training flag is True."""
    compression_params = compression.CompressionOp\
      .get_default_hparams().parse("")
    compressor = compression_wrapper.get_apply_compression(
        compression_params, global_step=0)

    in_tensor = tf.zeros((1, 5), dtype=tf.float32)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((5,)),
        layers.CompressedDense(
            10, compression_obj=compressor, name="compressed")
    ])

    # remove uncompressed kernel
    m.get_layer("compressed").kernel = None
    m.get_layer("compressed").compression_op.a_matrix_tfvar = None

    with self.assertRaises(ValueError) as exception_context:
      m(in_tensor, training=True)
    if not isinstance(exception_context.exception, ValueError):
      self.fail()


if __name__ == "__main__":
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
