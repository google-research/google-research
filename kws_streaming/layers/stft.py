# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Streaming aware Short-time Fourier Transform(STFT)."""
import math
from kws_streaming.layers import data_frame
from kws_streaming.layers import modes
from kws_streaming.layers import windowing
from kws_streaming.layers.compat import tf


class STFT(tf.keras.layers.Layer):
  """Streaming aware STFT layer.

  Computes stft in streaming or non-streaming mode.

  Attributes:
    frame_size: Sliding window/frame size in samples.
    frame_step: Number of samples to jump between frames. Also called hop size
    window_type: None or hann_tf are supported.
    inverse_stft_window_fn: If True window_fn=tf.signal.inverse_stft_window_fn
      else window_fn=synthesis_window_fn which is defined by window_type.
    fft_size: If None then closed to frame_size power of 2 will be used.
    mode: Inference or training mode.
    **kwargs: Additional layer arguments.
  """

  def __init__(
      self,
      frame_size,
      frame_step,
      mode=modes.Modes.TRAINING,
      inference_batch_size=1,
      padding='causal',
      window_type='hann_tf',
      fft_size=None,
      **kwargs):
    super(STFT, self).__init__(**kwargs)
    self.frame_size = frame_size
    self.frame_step = frame_step

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.padding = padding
    self.window_type = window_type
    if fft_size:
      self.fft_size = fft_size
    else:
      self.fft_size = 2**int(math.ceil(math.log(self.frame_size, 2)))

  def build(self, input_shape):
    super(STFT, self).build(input_shape)

    self.data_frame = data_frame.DataFrame(
        mode=self.mode,
        inference_batch_size=self.inference_batch_size,
        frame_size=self.frame_size,
        frame_step=self.frame_step,
        use_one_step=False,
        padding=self.padding)

    if self.window_type:
      self.windowing = windowing.Windowing(
          window_size=self.frame_size, window_type=self.window_type)
    else:
      self.windowing = tf.keras.layers.Lambda(lambda x: x)

    self.rfft = tf.keras.layers.Lambda(
        lambda x: tf.signal.rfft(x, fft_length=[self.fft_size]))

  def call(self, inputs):
    outputs = self.data_frame(inputs)
    outputs = self.windowing(outputs)
    outputs = self.rfft(outputs)
    return outputs

  def get_config(self):
    config = {
        'frame_size': self.frame_size,
        'frame_step': self.frame_step,
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'padding': self.padding,
        'window_type': self.window_type,
        'fft_size': self.fft_size,
    }
    base_config = super(STFT, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.data_frame.get_input_state()

  def get_output_state(self):
    return self.data_frame.get_output_state()
