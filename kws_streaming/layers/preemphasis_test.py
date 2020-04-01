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

"""Tests for kws_streaming.layers.preemphasis."""

import numpy as np
from kws_streaming.layers import preemphasis
from kws_streaming.layers.compat import tf
import kws_streaming.layers.test_utils as tu


class PreemphasisTest(tu.FrameTestBase):

  def test_derivative_calculation(self):
    # comapre TF implementation with numpy

    preemph = 0.97
    preemphasis_layer = preemphasis.Preemphasis(preemph=preemph)

    # it receives all data with size: data_size
    input1 = tf.keras.layers.Input(
        shape=(self.data_size,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output1 = preemphasis_layer(input1)
    model = tf.keras.models.Model(input1, output1)

    # generate frames for the whole signal (no streaming here)
    output_tf = model.predict(self.signal)

    output_np = []
    output_np.append(self.signal[0][0] * (1 - preemph))

    for i in range(1, self.data_size):
      derivative = self.signal[0][i] - preemph * self.signal[0][i - 1]
      output_np.append(derivative)

    self.assertAllClose(np.asarray(output_np), output_tf[0])


if __name__ == "__main__":
  tf.test.main()
