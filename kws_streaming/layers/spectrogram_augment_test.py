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

"""Tests for spectrogram_augment."""

import numpy as np
from kws_streaming.layers import spectrogram_augment
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class SpecAugmentTest(tf.test.TestCase):

  def setUp(self):
    super(SpecAugmentTest, self).setUp()
    self.input_shape = [1, 5, 5]
    self.seed = 6

  def test_time_masking(self):
    test_utils.set_seed(self.seed)
    spectrogram = np.ones(self.input_shape)
    inputs = tf.keras.layers.Input(
        shape=self.input_shape[1:],
        batch_size=self.input_shape[0],
        dtype=tf.float32)
    outputs = spectrogram_augment.SpecAugment(
        time_masks_number=2,
        time_mask_max_size=3,
        frequency_masks_number=0,
        frequency_mask_max_size=3)(
            inputs, training=True)
    model = tf.keras.models.Model(inputs, outputs)
    prediction = model.predict(spectrogram)
    target = np.array([[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0.]]])
    self.assertAllEqual(prediction, target)

  def test_frequency_masking(self):
    test_utils.set_seed(self.seed)
    spectrogram = np.ones(self.input_shape)
    inputs = tf.keras.layers.Input(
        shape=self.input_shape[1:],
        batch_size=self.input_shape[0],
        dtype=tf.float32)
    outputs = spectrogram_augment.SpecAugment(
        time_masks_number=0,
        time_mask_max_size=3,
        frequency_masks_number=2,
        frequency_mask_max_size=3)(
            inputs, training=True)
    model = tf.keras.models.Model(inputs, outputs)
    prediction = model.predict(spectrogram)
    target = np.array([[[1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.],
                        [1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.],
                        [1., 0., 0., 1., 0.]]])
    self.assertAllEqual(prediction, target)


if __name__ == "__main__":
  tf.test.main()
