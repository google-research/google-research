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

"""Tests for augmentation."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from non_semantic_speech_benchmark.data_prep import augmentation


class AugmentationTest(tf.test.TestCase, parameterized.TestCase):

  def test_spec_augment_inference(self):
    """Verify inference does not do augmentation."""
    input_tensor_shape = [3, 96, 64, 1]  # log Mel spectrogram.
    input_tensor = tf.ones(input_tensor_shape, dtype=tf.float32)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((96, 64, 1)),
        augmentation.SpecAugment()
    ])

    out = m(input_tensor, training=False)
    self.assertListEqual(list(out.shape), input_tensor_shape)
    self.assertAllEqual(out, input_tensor)

  def test_spec_augment_training(self):
    """Verify augmentaion occurs during training."""
    input_tensor_shape = [3, 96, 64, 1]  # log Mel spectrogram.
    input_tensor = tf.ones(input_tensor_shape, dtype=tf.float32)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((96, 64, 1)),
        augmentation.SpecAugment()
    ])

    out = m(input_tensor, training=True)
    self.assertListEqual(list(out.shape), input_tensor_shape)
    self.assertNotAllEqual(out, input_tensor)

if __name__ == "__main__":
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
