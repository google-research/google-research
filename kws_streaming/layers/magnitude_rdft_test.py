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

"""Tests for kws_streaming.layers.magnitude_rdft."""

import numpy as np
from kws_streaming.layers import magnitude_rdft
from kws_streaming.layers.compat import tf


class MagnitudeRDFTTest(tf.test.TestCase):

  def test_tf_fft_vs_rdft_direct(self):

    signal_size = 64
    # input signal
    signal = np.random.rand(1, signal_size)

    # build rfft model and run it
    input_signal = tf.keras.Input(shape=(signal_size,), batch_size=1)
    spectrum = tf.signal.rfft(input_signal)
    spectrum = tf.abs(spectrum)
    model = tf.keras.Model(input_signal, spectrum)
    model.summary()
    spectrum_output = model.predict(signal)

    # build rdft model and run it
    input_signal = tf.keras.Input(shape=(signal_size,), batch_size=1)
    output = magnitude_rdft.MagnitudeRDFT(magnitude_squared=False)(input_signal)
    model = tf.keras.Model(input_signal, output)
    model.summary()
    rdft_output = model.predict(signal)

    self.assertAllClose(rdft_output, spectrum_output)


if __name__ == "__main__":
  tf.test.main()
