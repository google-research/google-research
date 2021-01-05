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

# Lint as: python3
"""Tests for m_layer.

We test that we can set up a model and run inference.
We are not trying to ensure that training works.
"""

import numpy
import tensorflow as tf
from m_layer import MLayer


class MLayerTest(tf.test.TestCase):

  def test_m_layer(self):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3,)),
        MLayer(dim_m=5, matrix_init='normal'),
        tf.keras.layers.ActivityRegularization(l2=1e-4),
        tf.keras.layers.Flatten()
    ])
    mlayer = model.layers[1]
    self.assertEqual(mlayer.trainable_weights[0].shape, [3, 5, 5])

    prediction = model.predict(tf.ones((1, 3)))
    self.assertFalse(numpy.isnan(prediction).any())

if __name__ == '__main__':
  tf.test.main()
