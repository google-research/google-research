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

"""Tests for kws_streaming.layers.mel_table."""

import math
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
import kws_streaming.layers.mel_table as mel_table
import kws_streaming.layers.test_utils as tu
tf1.disable_eager_execution()


class MelTableTest(tu.TestBase):

  def test_tf_np_mel_tables(self):
    frame_size = 400
    num_mel_bins = 80
    lower_edge_hertz = 125.0
    upper_edge_hertz = 7600.0
    sample_rate = 16000.0
    fft_size = 2**int(math.ceil(math.log(frame_size) / math.log(2.0)))

    # run numpy implementation
    mel_np = mel_table.SpectrogramToMelMatrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=fft_size // 2 + 1,
        audio_sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz)

    # run TF implementation
    mel_tf = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=fft_size // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        dtype=tf.float32)

    # compare numpy vs TF implementations
    self.assertAllClose(mel_np, mel_tf, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
