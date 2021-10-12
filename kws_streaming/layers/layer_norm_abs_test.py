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

"""Tests for kws_streaming.layers.layer_norm_abs."""

import numpy as np
from kws_streaming.layers import layer_norm_abs
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class LayerNormalizationAbsTest(tf.test.TestCase):

  def test(self):

    inputs = tf.keras.Input(
        shape=(2, 2), batch_size=1)
    outputs = layer_norm_abs.LayerNormalizationAbs(epsilon=1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    input_np = np.array([[[1.0, -1.0], [2.0, 4.0]]])
    output = model.predict(input_np)
    self.assertAllClose(
        output, np.array([[[0.5, -0.5], [-0.5, 0.5]]], dtype=np.float32))


if __name__ == "__main__":
  tf.test.main()
