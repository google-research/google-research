# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for kws_streaming.layers.magnitude_rdft_mel."""

from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import magnitude_rdft
from kws_streaming.layers import magnitude_rdft_mel
from kws_streaming.layers import mel_spectrogram
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class MagnitudeRDFTmelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(testcase_name='reduced rdft dim', mel_non_zero_only=True),
      dict(testcase_name='default rdft dim', mel_non_zero_only=False)
  ])
  def test_rdft_mel_vs_merged_rdft_mel(self, mel_non_zero_only):

    signal_size = 257
    # input signal
    signal = np.random.rand(1, signal_size)

    # model parameters
    mode = Modes.NON_STREAM_INFERENCE
    use_tf_fft = False
    magnitude_squared = False
    num_mel_bins = 40
    lower_edge_hertz = 20.0
    upper_edge_hertz = 4000.0
    sample_rate = 16000.0

    # build rdft mel model and run it
    input_signal = tf.keras.Input(shape=(signal_size,), batch_size=1)
    mag_rdft = magnitude_rdft.MagnitudeRDFT(
        mode=mode, use_tf_fft=use_tf_fft, magnitude_squared=magnitude_squared)(
            input_signal)
    mel_spectr = mel_spectrogram.MelSpectrogram(
        mode=mode,
        use_tf=False,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        sample_rate=sample_rate)(
            mag_rdft)
    model_rdft_mel = tf.keras.Model(input_signal, mel_spectr)
    model_rdft_mel.summary()
    rdft_mel_output = model_rdft_mel.predict(signal)

    # build merged rdft mel model and run it
    merged_rdft_mel = magnitude_rdft_mel.MagnitudeRDFTmel(
        use_tf_fft=use_tf_fft,
        magnitude_squared=magnitude_squared,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        sample_rate=sample_rate,
        mel_non_zero_only=mel_non_zero_only)(
            input_signal)
    model_merged_rdft_mel = tf.keras.Model(input_signal, merged_rdft_mel)
    model_merged_rdft_mel.summary()
    merged_rdft_mel_output = model_merged_rdft_mel.predict(signal)

    shape_rdft_mel = model_rdft_mel.layers[2].mel_weight_matrix.shape
    shape_rdft_melmerged = model_merged_rdft_mel.layers[
        1].mel_weight_matrix.shape
    if mel_non_zero_only:
      # shape of mel matrix with merged method is 2x smaller
      self.assertGreater(shape_rdft_mel[0] * shape_rdft_mel[1],
                         2 * shape_rdft_melmerged[0] * shape_rdft_melmerged[1])
    else:
      self.assertEqual(shape_rdft_mel[0] * shape_rdft_mel[1],
                       shape_rdft_melmerged[0] * shape_rdft_melmerged[1])
    self.assertAllClose(rdft_mel_output, merged_rdft_mel_output)


if __name__ == '__main__':
  tf.test.main()
