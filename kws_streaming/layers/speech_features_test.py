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

"""Tests for kws_streaming.layers.speech_features."""

import numpy as np
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils
tf1.disable_eager_execution()


class MelSpeechFeaturesTest(tf.test.TestCase):
  """Mel speech feature extractor testing."""

  def setUp(self):
    super(MelSpeechFeaturesTest, self).setUp()

    self.inference_batch_size = 1
    self.preemph = 0.97
    self.window_type = 'hann'
    self.frame_size_ms = 25.0
    self.frame_step_ms = 10.0
    self.mel_num_bins = 80
    self.mel_lower_edge_hertz = 125.0
    self.mel_upper_edge_hertz = 7600.0
    self.sample_rate = 16000.0
    self.noise_scale = 0.0
    self.mean = None
    self.stddev = None

    self.frame_size = int(round(self.sample_rate * self.frame_size_ms / 1000.0))
    self.frame_step = int(round(self.sample_rate * self.frame_step_ms / 1000.0))

    # generate input signal
    np.random.seed(1)
    self.data_size = 1024
    self.signal = np.random.rand(self.inference_batch_size, self.data_size)

  def test_tf_non_streaming_vs_streaming_inference_internal_state(self):
    use_tf_fft = False
    mode = Modes.NON_STREAM_INFERENCE
    # TF non streaming frame extraction based on tf.signal.frame
    mel_speech_tf = speech_features.SpeechFeatures(
        mode=mode,
        use_tf_fft=use_tf_fft,
        inference_batch_size=self.inference_batch_size,
        preemph=self.preemph,
        window_type=self.window_type,
        frame_size_ms=self.frame_size_ms,
        frame_step_ms=self.frame_step_ms,
        mel_num_bins=self.mel_num_bins,
        mel_lower_edge_hertz=self.mel_lower_edge_hertz,
        mel_upper_edge_hertz=self.mel_upper_edge_hertz,
        sample_rate=self.sample_rate,
        noise_scale=self.noise_scale,
        mean=self.mean,
        stddev=self.stddev)
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
    mode = Modes.STREAM_INTERNAL_STATE_INFERENCE
    mel_speech_stream = speech_features.SpeechFeatures(
        mode=mode,
        use_tf_fft=use_tf_fft,
        inference_batch_size=self.inference_batch_size,
        preemph=self.preemph,
        window_type=self.window_type,
        frame_size_ms=self.frame_size_ms,
        frame_step_ms=self.frame_step_ms,
        mel_num_bins=self.mel_num_bins,
        mel_lower_edge_hertz=self.mel_lower_edge_hertz,
        mel_upper_edge_hertz=self.mel_upper_edge_hertz,
        sample_rate=self.sample_rate,
        noise_scale=self.noise_scale,
        mean=self.mean,
        stddev=self.stddev)
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

    # compare streaming vs non streaming frames extraction
    for i in range(len(streamed_frames)):
      self.assertAllClose(
          streamed_frames[i][0][0], output_tf[0][i], rtol=1e-4, atol=1e-4)

  def test_tf_non_streaming_vs_streaming_inference_external_state(self):
    use_tf_fft = False
    mode = Modes.NON_STREAM_INFERENCE
    # TF non streaming frame extraction based on tf.signal.frame
    mel_speech_tf = speech_features.SpeechFeatures(
        mode=mode,
        use_tf_fft=use_tf_fft,
        inference_batch_size=self.inference_batch_size,
        preemph=self.preemph,
        window_type=self.window_type,
        frame_size_ms=self.frame_size_ms,
        frame_step_ms=self.frame_step_ms,
        mel_num_bins=self.mel_num_bins,
        mel_lower_edge_hertz=self.mel_lower_edge_hertz,
        mel_upper_edge_hertz=self.mel_upper_edge_hertz,
        sample_rate=self.sample_rate,
        noise_scale=self.noise_scale,
        mean=self.mean,
        stddev=self.stddev)
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
    mode = Modes.STREAM_EXTERNAL_STATE_INFERENCE
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


if __name__ == '__main__':
  tf.test.main()
