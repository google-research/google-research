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

"""Test for cutout spectrogram augmentation."""

import numpy as np
from kws_streaming.layers import spectrogram_cutout
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class CutoutTest(tf.test.TestCase):

  def setUp(self):
    super(CutoutTest, self).setUp()
    self.input_shape = [2, 7, 5]
    self.seed = 1

  def test_masking(self):
    test_utils.set_seed(self.seed)
    spectrogram = np.ones(self.input_shape)
    inputs = tf.keras.layers.Input(
        shape=self.input_shape[1:],
        batch_size=self.input_shape[0],
        dtype=tf.float32)
    outputs = spectrogram_cutout.SpecCutout(
        masks_number=2,
        time_mask_size=4,
        frequency_mask_size=2,
        seed=self.seed)(
            inputs, training=True)
    model = tf.keras.models.Model(inputs, outputs)
    prediction = model.predict(spectrogram)
    # confirm that every mask has different rects in different batch indexes
    target0 = np.array([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.],
                        [0., 1., 1., 1., 1.], [0., 1., 1., 0., 0.],
                        [0., 1., 1., 0., 0.], [0., 1., 1., 0., 0.],
                        [1., 1., 1., 0., 0.]])
    self.assertAllEqual(prediction[0, :, :], target0)

    target1 = np.array([[1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.],
                        [0., 1., 1., 1., 1.], [0., 1., 0., 0., 1.],
                        [0., 1., 0., 0., 1.], [1., 1., 0., 0., 1.],
                        [1., 1., 0., 0., 1.]])
    self.assertAllEqual(prediction[1, :, :], target1)


if __name__ == "__main__":
  tf.test.main()
