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

"""Tests for kws_streaming.layers.speech_features."""

import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import model_params
from kws_streaming.models import utils
tf1.disable_eager_execution()


class Params(model_params.Params):
  """Parameters for testing speech feature extractor.

     These parameters are compatible with command line flags
     and discribed in /train/base_parser.py
  """

  def __init__(self):
    super().__init__()
    self.window_size_ms = 25.0
    self.window_stride_ms = 10.0
    self.preemph = 0.97
    self.dct_num_features = 13
    self.use_spec_augment = 1
    self.use_spec_cutout = 1
    self.use_tf_fft = 0
    self.time_shift_ms = 0.0
    self.sp_time_shift_ms = 100.0
    self.resample = 0.0
    self.sp_resample = 0.2
    self.train = 0
    self.batch_size = 1
    self.mode = modes.Modes.NON_STREAM_INFERENCE
    self.data_stride = 1
    self.data_frame_padding = None


class SpeechFeaturesTest(tf.test.TestCase):
  """Speech feature extractor testing."""

  def setUp(self):
    super().setUp()

    self.inference_batch_size = 1
    self.params = Params()
    self.frame_size = int(
        round(self.params.sample_rate * self.params.window_size_ms / 1000.0))
    self.frame_step = int(
        round(self.params.sample_rate * self.params.window_stride_ms / 1000.0))

    # generate input signal
    test_utils.set_seed(1)
    self.data_size = 1024
    self.signal = np.random.rand(self.inference_batch_size, self.data_size)

  def test_tf_non_streaming_train(self):
    """Tests non stream inference with train flag."""
    params = Params()
    params.sp_time_shift_ms = 10.0
    speech_params = speech_features.SpeechFeatures.get_params(params)
    mode = modes.Modes.TRAINING
    # TF non streaming frame extraction based on tf.signal.frame
    mel_speech_tf = speech_features.SpeechFeatures(
        speech_params, mode, self.inference_batch_size)
    # it receives all data with size: data_size
    input1 = tf.keras.layers.Input(
        shape=(self.data_size,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output1 = mel_speech_tf(input1)
    model_tf = tf.keras.models.Model(input1, output1)

    # generate frames for the whole signal (no streaming here)
    self.assertNotEmpty(model_tf.predict(self.signal))

  def test_tf_non_streaming_vs_streaming_inference_internal_state(self):
    """Tests non stream inference vs stream inference with internal state."""
    speech_params = speech_features.SpeechFeatures.get_params(self.params)
    mode = modes.Modes.NON_STREAM_INFERENCE
    # TF non streaming frame extraction based on tf.signal.frame
    mel_speech_tf = speech_features.SpeechFeatures(
        speech_params, mode, self.inference_batch_size)
    # it receives all data with size: data_size
    input1 = tf.keras.layers.Input(
        shape=(self.data_size,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output1 = mel_speech_tf(input1)
    model_tf = tf.keras.models.Model(input1, output1)

    # generate frames for the whole signal (no streaming here)
    output_tf = model_tf.predict(self.signal)

    # streaming frame extraction
    # it receives input data incrementally with step: frame_step
    mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    mel_speech_stream = speech_features.SpeechFeatures(
        speech_params, mode, self.inference_batch_size)
    input2 = tf.keras.layers.Input(
        shape=(self.frame_step,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output2 = mel_speech_stream(input2)

    # initialize state of streaming model
    pre_state = self.signal[:, 0:mel_speech_stream.data_frame.frame_size -
                            mel_speech_stream.data_frame.frame_step]
    state_init = np.concatenate((np.zeros(
        shape=(1, mel_speech_stream.data_frame.frame_step),
        dtype=np.float32), pre_state),
                                axis=1)
    mel_speech_stream.data_frame.set_weights([state_init])
    model_stream = tf.keras.models.Model(input2, output2)

    # run streaming frames extraction
    start = self.frame_size - self.frame_step
    end = self.frame_size
    streamed_frames = []
    while end <= self.data_size:
      # next data update
      stream_update = self.signal[:, start:end]

      # get new frame from stream of data
      output_frame = model_stream.predict(stream_update)
      streamed_frames.append(output_frame)

      # update indexes of streamed updates
      start = end
      end = start + self.frame_step

    self.assertNotEmpty(streamed_frames)
    # compare streaming vs non streaming frames extraction
    for i in range(len(streamed_frames)):
      self.assertAllClose(
          streamed_frames[i][0][0], output_tf[0][i], rtol=1e-4, atol=1e-4)

  def test_tf_non_streaming_vs_streaming_inference_external_state(self):
    """Tests non stream inference vs stream inference with external state."""
    speech_params = speech_features.SpeechFeatures.get_params(self.params)
    mode = modes.Modes.NON_STREAM_INFERENCE
    # TF non streaming frame extraction based on tf.signal.frame
    mel_speech_tf = speech_features.SpeechFeatures(
        speech_params, mode, self.inference_batch_size)
    # it receives all data with size: data_size
    input1 = tf.keras.layers.Input(
        shape=(self.data_size,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output1 = mel_speech_tf(input1)
    model_tf = tf.keras.models.Model(input1, output1)

    # generate frames for the whole signal (no streaming here)
    output_tf = model_tf.predict(self.signal)

    # input data for streaming mode
    input_tensors = [
        tf.keras.layers.Input(
            shape=(self.frame_step,),
            batch_size=self.inference_batch_size,
            dtype=tf.float32)
    ]

    # convert non streaming trainable model to
    # streaming inference with external state
    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    model_stream = utils.convert_to_inference_model(model_tf, input_tensors,
                                                    mode)

    # initialize state of streaming model
    pre_state = self.signal[:, 0:self.frame_size - self.frame_step]
    state2 = np.concatenate(
        (np.zeros(shape=(1, self.frame_step), dtype=np.float32), pre_state),
        axis=1)

    # run streaming frames extraction
    start = self.frame_size - self.frame_step
    end = self.frame_size
    streamed_frames = []
    while end <= self.data_size:
      # next data update
      stream_update = self.signal[:, start:end]

      # get new frame from stream of data
      output_frame, output_state = model_stream.predict([stream_update, state2])
      state2 = output_state
      streamed_frames.append(output_frame)

      # update indexes of streamed updates
      start = end
      end = start + self.frame_step

    # compare streaming vs non streaming frames extraction
    for i in range(len(streamed_frames)):
      self.assertAllClose(
          streamed_frames[i][0][0], output_tf[0][i], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
