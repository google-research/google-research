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

"""Tests for kws_streaming.layers.stft."""
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import stft
from kws_streaming.layers import temporal_padding
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


class STFTTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(STFTTest, self).setUp()
    test_utils.set_seed(123)

    self.frame_size = 40
    self.frame_step = 10
    # layer definition
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal')

    if stft_layer.window_type == 'hann_tf':
      synthesis_window_fn = tf.signal.hann_window
    else:
      synthesis_window_fn = None

    # prepare input data
    self.input_signal = np.random.rand(1, 120)

    # prepare default tf stft
    padding_layer = temporal_padding.TemporalPadding(
        padding_size=stft_layer.frame_size - 1, padding=stft_layer.padding)
    # pylint: disable=g-long-lambda
    stft_default_layer = tf.keras.layers.Lambda(
        lambda x: tf.signal.stft(
            x,
            stft_layer.frame_size,
            stft_layer.frame_step,
            fft_length=stft_layer.fft_size,
            window_fn=synthesis_window_fn,
            pad_end=False))
    # pylint: enable=g-long-lambda
    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1)
    net = padding_layer(input_tf)
    net = stft_default_layer(net)

    model_stft = tf.keras.models.Model(input_tf, net)

    self.stft_out = model_stft.predict(self.input_signal)

  def testNonStreaming(self):
    # prepare non streaming model and compare it with default stft
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal')
    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1)
    net = stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)
    self.non_stream_out = model_non_stream.predict(self.input_signal)
    self.assertAllClose(self.non_stream_out, self.stft_out)

  @parameterized.named_parameters(
      {
          'testcase_name': 'streaming frame by frame',
          'input_samples': 1,
      }, {
          'testcase_name': 'streaming with 3 frames per call',
          'input_samples': 3,
      })
  def testStreaming(self, input_samples):
    # prepare non streaming model
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal')
    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1)
    net = stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)

    params = test_utils.Params([1])
    # shape of input data in the inference streaming mode (excluding batch size)
    params.data_shape = (input_samples*stft_layer.frame_step,)
    params.step = input_samples

    # convert it to streaming model
    model_stream = utils.to_streaming_inference(
        model_non_stream,
        params,
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run streaming inference and compare it with default stft
    stream_out = inference.run_stream_inference(params, model_stream,
                                                self.input_signal)
    stream_output_length = stream_out.shape[1]
    self.assertAllClose(stream_out, self.stft_out[:, 0:stream_output_length])


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
