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

"""Tests for kws_streaming.layers.mel_spectrogram."""

import numpy as np
from kws_streaming.layers import mel_spectrogram
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
import kws_streaming.layers.test_utils as tu
tf1.disable_eager_execution()


class MelSpectrogramTest(tu.TestBase):

  def test_tf_vs_tf_direct(self):
    # Compare TF implementation of Mel (based on FFT)
    # vs TF direct implementation (based on FT)
    feature_size = 257
    num_mel_bins = 80
    lower_edge_hertz = 125.0
    upper_edge_hertz = 7600.0
    sample_rate = 16000.0
    batch_size = 1

    np.random.seed(1)

    # generate input data
    frame = np.random.rand(batch_size, feature_size)

    # prepare model with TF implementation of Mel based on FFT
    input1 = tf.keras.layers.Input(
        shape=(feature_size,), batch_size=batch_size, dtype=tf.float32)
    mel_spectrum = mel_spectrogram.MelSpectrogram(
        mode=Modes.NON_STREAM_INFERENCE,
        use_tf=True,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        sample_rate=sample_rate)
    output1 = mel_spectrum(input1)
    model_tf = tf.keras.models.Model(input1, output1)
    # generate mel
    output_tf = model_tf.predict(frame)

    # prepare model with TF implementation of Mel based on direct FT
    input2 = tf.keras.layers.Input(
        shape=(feature_size,), batch_size=batch_size, dtype=tf.float32)
    mel_spectrum_direct = mel_spectrogram.MelSpectrogram(
        mode=Modes.STREAM_EXTERNAL_STATE_INFERENCE,
        use_tf=False,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        sample_rate=sample_rate)
    output2 = mel_spectrum_direct(input2)
    model_tf_direct = tf.keras.models.Model(input2, output2)
    # generate mel
    output_tf_direct = model_tf_direct.predict(frame)

    self.assertAllClose(output_tf, output_tf_direct, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
