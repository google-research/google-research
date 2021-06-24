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

"""Tests for non_semantic_speech_benchmark.distillation.models."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import models
from non_semantic_speech_benchmark.distillation.compression_lib import compression_op as compression
from non_semantic_speech_benchmark.distillation.compression_lib import compression_wrapper


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      {'frontend': True, 'bottleneck': 3, 'tflite': True},
      {'frontend': False, 'bottleneck': 3, 'tflite': True},
      {'frontend': True, 'bottleneck': 3, 'tflite': False},
      {'frontend': False, 'bottleneck': 3, 'tflite': False},
      {'frontend': True, 'bottleneck': 0, 'tflite': False},
  )
  def test_model_frontend(self, frontend, bottleneck, tflite):
    if frontend:
      input_tensor_shape = [1 if tflite else 2, 32000]  # audio signal.
    else:
      input_tensor_shape = [3, 96, 64, 1]  # log Mel spectrogram.
    input_tensor = tf.zeros(input_tensor_shape, dtype=tf.float32)
    output_dimension = 5

    m = models.get_keras_model(
        bottleneck, output_dimension, frontend=frontend, tflite=tflite)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    emb.shape.assert_has_rank(2)
    if bottleneck:
      self.assertEqual(emb.shape[1], bottleneck)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_invalid_mobilenet_size(self):
    invalid_mobilenet_size = 'huuuge'
    with self.assertRaises(ValueError) as exception_context:
      models.get_keras_model(3, 5, mobilenet_size=invalid_mobilenet_size)
    if not isinstance(exception_context.exception, ValueError):
      self.fail()

  def test_default_shape(self):
    self.assertEqual(models._get_frontend_output_shape(), [1, 96, 64])

  @parameterized.parameters(
      {'mobilenet_size': 'tiny'},
      {'mobilenet_size': 'small'},
      {'mobilenet_size': 'large'},
      {'mobilenet_size': 'tiny'},
  )
  def test_valid_mobilenet_size(self, mobilenet_size):
    input_tensor = tf.zeros([2, 32000], dtype=tf.float32)
    m = models.get_keras_model(3, 5, mobilenet_size=mobilenet_size)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    emb.shape.assert_has_rank(2)
    self.assertEqual(emb.shape[1], 3)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  @parameterized.parameters({'add_compression': True},
                            {'add_compression': False})
  def test_tflite_model(self, add_compression):
    compressor = None
    bottleneck_dimension = 3
    if add_compression:
      compression_params = compression.CompressionOp.get_default_hparams(
          ).parse('')
      compressor = compression_wrapper.get_apply_compression(
          compression_params, global_step=0)
    m = models.get_keras_model(
        bottleneck_dimension,
        5,
        frontend=False,
        mobilenet_size='small',
        compressor=compressor,
        tflite=True)

    input_tensor = tf.zeros([1, 96, 64, 1], dtype=tf.float32)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    emb.shape.assert_has_rank(2)
    self.assertEqual(emb.shape[0], 1)
    self.assertEqual(emb.shape[1], bottleneck_dimension)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[0], 1)
    self.assertEqual(o.shape[1], 5)

    if add_compression:
      self.assertIsNone(m.get_layer('distilled_output').kernel)
      self.assertIsNone(
          m.get_layer('distilled_output').compression_op.a_matrix_tfvar)


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
