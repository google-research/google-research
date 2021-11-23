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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import models


class ModelsTest(parameterized.TestCase):


  @parameterized.parameters(
      {'frontend': True, 'tflite': True},
      {'frontend': False, 'tflite': True},
      {'frontend': True, 'tflite': False},
      {'frontend': False, 'tflite': False},
  )
  def test_model_frontend(self, frontend, tflite):
    if frontend:
      input_tensor_shape = [1 if tflite else 2, 32000]  # audio signal.
    else:
      input_tensor_shape = [3, 96, 64, 1]  # log Mel spectrogram.
    input_tensor = tf.zeros(input_tensor_shape, dtype=tf.float32)
    output_dimension = 5

    m = models.get_keras_model(
        'mobilenet_debug_1.0_False', output_dimension,
        frontend=frontend, tflite=tflite)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    emb.shape.assert_has_rank(2)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_invalid_model(self):
    invalid_mobilenet_size = 'huuuge'
    with self.assertRaises(KeyError) as exception_context:
      models.get_keras_model(
          f'mobilenet_{invalid_mobilenet_size}_1.0_False', 5)
    if not isinstance(exception_context.exception, KeyError):
      self.fail()

  def test_truncation(self):
    m = models.get_keras_model(
        'efficientnetb0',
        frontend=False,
        output_dimension=5,
        truncate_output=True)

    input_tensor = tf.zeros([3, 96, 64, 1], dtype=tf.float32)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    # `embedding` is the original size, but `embedding_to_target` should be the
    # right size.
    self.assertEqual(emb.shape[1], 5)
    self.assertEqual(o.shape[1], 5)

  @parameterized.parameters(
      {'model_type': 'mobilenet_small_1.0_False'},
      {'model_type': 'efficientnetb0'},
      {'model_type': 'efficientnetb1'},
      {'model_type': 'efficientnetb2'},
      {'model_type': 'efficientnetb3'},
      {'model_type': 'efficientnetb4'},
      {'model_type': 'efficientnetb5'},
      {'model_type': 'efficientnetb6'},
      {'model_type': 'efficientnetb7'},
      {'model_type': 'efficientnetv2b0'},
      {'model_type': 'efficientnetv2b1'},
      {'model_type': 'efficientnetv2b2'},
      {'model_type': 'efficientnetv2b3'},
      {'model_type': 'efficientnetv2bL'},
      {'model_type': 'efficientnetv2bM'},
      {'model_type': 'efficientnetv2bS'},
  )
  @flagsaver.flagsaver
  def test_valid_model_type(self, model_type):
    # Frontend flags.
    flags.FLAGS.frame_hop = 5
    flags.FLAGS.num_mel_bins = 80
    flags.FLAGS.frame_width = 5

    input_tensor = tf.zeros([2, 16000], dtype=tf.float32)
    m = models.get_keras_model(
        model_type, 5, frontend=True, truncate_output=True)
    o_dict = m(input_tensor)
    o = o_dict['embedding_to_target']

    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_tflite_model(self):
    m = models.get_keras_model(
        'mobilenet_debug_1.0_False',
        5,
        frontend=False,
        tflite=True)

    input_tensor = tf.zeros([2, 96, 64, 1], dtype=tf.float32)
    o_dict = m(input_tensor)
    emb, o = o_dict['embedding'], o_dict['embedding_to_target']

    emb.shape.assert_has_rank(2)
    self.assertEqual(emb.shape[0], 2)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[0], 2)
    self.assertEqual(o.shape[1], 5)


if __name__ == '__main__':
  assert tf.executing_eagerly()
  absltest.main()
