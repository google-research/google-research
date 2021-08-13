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

"""Tests for kws_streaming.layers.inverse_stft."""
from absl.testing import parameterized
from kws_streaming.layers import inverse_stft
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


class InverseSTFTTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(InverseSTFTTest, self).setUp()
    test_utils.set_seed(123)

    self.frame_size = 32
    self.frame_step = 8
    # layer definition
    inverse_stft_layer = inverse_stft.InverseSTFT(
        self.frame_size, self.frame_step)

    # prepare input stft data
    input_audio = tf.random.uniform((1, 256), maxval=1.0)
    signal_stft_tf = tf.signal.stft(
        input_audio,
        inverse_stft_layer.frame_size,
        inverse_stft_layer.frame_step,
        inverse_stft_layer.fft_size,
        window_fn=inverse_stft_layer.synthesis_window_fn,
        pad_end=False)
    with tf1.Session() as sess:
      self.signal_stft = sess.run(signal_stft_tf)

    self.feature_size = self.signal_stft.shape[-1]

    # create istft model and run non stream inference
    input_tf = tf.keras.layers.Input(
        shape=self.signal_stft.shape[1:3], batch_size=1, dtype=tf.complex64)
    net = inverse_stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)
    self.non_stream_out = model_non_stream.predict(self.signal_stft)

  @parameterized.named_parameters(
      {
          'testcase_name': 'streaming frame by frame',
          'input_frames': 1,
      }, {
          'testcase_name': 'streaming with 3 input frames',
          'input_frames': 3,
      })
  def testStreaming(self, input_frames):
    params = test_utils.Params([1])

    # shape of input data in the inference streaming mode (excluding batch size)
    params.data_shape = (1, self.feature_size)
    params.step = input_frames

    # prepare non streaming model
    inverse_stft_layer = inverse_stft.InverseSTFT(
        self.frame_size, self.frame_step, use_one_step=(input_frames == 1))
    input_tf = tf.keras.layers.Input(
        shape=self.signal_stft.shape[1:3], batch_size=1, dtype=tf.complex64)
    net = inverse_stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)
    self.non_stream_out = model_non_stream.predict(self.signal_stft)

    # convert it to streaming model
    model_stream = utils.to_streaming_inference(
        model_non_stream,
        params,
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run streaming inference
    stream_out = inference.run_stream_inference(params, model_stream,
                                                self.signal_stft)

    # several samples in the end will be missing
    stream_output_length = stream_out.shape[1]
    self.assertAllClose(stream_out, self.non_stream_out[:,
                                                        0:stream_output_length])


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
