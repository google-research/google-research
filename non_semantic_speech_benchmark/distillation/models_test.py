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

"""Tests for non_semantic_speech_benchmark.distillation.models."""

from absl.testing import absltest

import tensorflow as tf

from non_semantic_speech_benchmark.distillation import models


class ModelsTest(absltest.TestCase):

  def test_model_frontend(self):
    input_tensor = tf.zeros([2, 32000], dtype=tf.float32)  # audio signal
    m = models.get_keras_model(3, 5)
    o = m(input_tensor)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_model_no_frontend(self):
    input_tensor = tf.zeros([1, 96, 64, 1], dtype=tf.float32)  # log Mel spectrogram
    m = models.get_keras_model(3, 5, frontend=False)
    o = m(input_tensor)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_model_no_bottleneck(self):
    input_tensor = tf.zeros([2, 32000], dtype=tf.float32)
    m = models.get_keras_model(0, 5)
    o = m(input_tensor)
    o.shape.assert_has_rank(2)
    self.assertEqual(o.shape[1], 5)

  def test_invalid_mobilenet_size(self):
    invalid_mobilenet_size = "huuuge"
    with self.assertRaises(AssertionError) as exception_context:
      models.get_keras_model(3, 5, mobilenet_size=invalid_mobilenet_size)
    if not isinstance(exception_context.exception, AssertionError):
      self.fail()

  def test_valid_mobilenet_size(self):
    input_tensor = tf.zeros([2, 32000], dtype=tf.float32)
    for mobilenet_size in ("tiny", "small", "large"):
      m = models.get_keras_model(3, 5, mobilenet_size=mobilenet_size)
      o = m(input_tensor)
      o.shape.assert_has_rank(2)
      self.assertEqual(o.shape[1], 5)


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
