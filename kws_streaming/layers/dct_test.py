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

"""Tests for kws_streaming.layers.dct."""

import numpy as np
from kws_streaming.layers import dct
from kws_streaming.layers.compat import tf


class DCTTest(tf.test.TestCase):

  def test_tf_dct_vs_dct_direct(self):

    signal_size = 64
    # input signal
    signal = np.random.rand(1, 1, signal_size)

    # build mfcc model and run it
    input_signal = tf.keras.Input(
        shape=(
            1,
            signal_size,
        ), batch_size=1)
    output = tf.signal.mfccs_from_log_mel_spectrograms(input_signal)
    model = tf.keras.Model(input_signal, output)
    model.summary()
    mfcc_output = model.predict(signal)

    # build dct model and run it
    input_signal = tf.keras.Input(
        shape=(
            1,
            signal_size,
        ), batch_size=1)
    output = dct.DCT()(input_signal)
    model = tf.keras.Model(input_signal, output)
    model.summary()
    dct_output = model.predict(signal)

    self.assertAllClose(
        mfcc_output[0][0], dct_output[0][0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
