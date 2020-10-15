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

"""Test for RandomStretchSqueeze data augmentation."""
import numpy as np
from kws_streaming.layers import random_stretch_squeeze
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class RandomStretchSqueezeTest(tf.test.TestCase):

  def setUp(self):
    super(RandomStretchSqueezeTest, self).setUp()
    self.input_shape = [2, 7]  # [batch, audio_sequence]
    self.seed = 5

  def test_random_stretch_squeeze(self):
    test_utils.set_seed(self.seed)
    audio = np.zeros(self.input_shape)
    audio[:, 2:5,] = 1
    inputs = tf.keras.layers.Input(
        shape=self.input_shape[1:],
        batch_size=self.input_shape[0],
        dtype=tf.float32)
    outputs = random_stretch_squeeze.RandomStretchSqueeze(
        resample_offset=0.5,
        seed=self.seed)(
            inputs, training=True)
    model = tf.keras.models.Model(inputs, outputs)
    prediction = model.predict(audio)

    # confirm that data are squeezed
    target0 = np.array([0., 0., 1., 1., 0., 0., 0.])
    self.assertAllClose(prediction[0, :], target0)

    # confirm that data are stretched
    target1 = np.array([0., 0.44444, 1., 1., 1., 0.44444, 0.])
    self.assertAllClose(prediction[1, :], target1, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
