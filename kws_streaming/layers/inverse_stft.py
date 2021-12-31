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

"""Streaming aware inverse_stft layer."""
import functools
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


class InverseSTFT(tf.keras.layers.Layer):
  """Streaming aware InverseSTFT layer.

  Computes inverse_stft in streaming or non-streaming mode.

  Attributes:
    frame_size: Sliding window/frame size in samples.
    frame_step: Number of samples to jump between frames. Also called hop size
    window_type: None or hann_tf are supported.
    inverse_stft_window_fn: If True window_fn=tf.signal.inverse_stft_window_fn
      else window_fn=synthesis_window_fn which is defined by window_type.
    fft_size: If None then closed to frame_size power of 2 will be used.
    mode: Inference or training mode.
    use_one_step: If True, model will run one sample per one inference step;
      if False, model will run multiple per one inference step. It is useful
      for strided streaming.
    input_frames: Number of the input frames in streaming mode, it will be
      estimated automatically in build method.
    state_name_tag: Tag appended to the state's name.
    **kwargs: Additional layer arguments.
  """

  def __init__(self,
               frame_size,
               frame_step,
               inverse_stft_window_fn=True,
               window_type='hann_tf',
               fft_size=None,
               inference_batch_size=1,
               mode=modes.Modes.TRAINING,
               use_one_step=False,
               input_frames=None,
               state_name_tag='ExternalState',
               **kwargs):
    super(InverseSTFT, self).__init__(**kwargs)
    self.frame_size = frame_size
    self.frame_step = frame_step
    self.window_type = window_type
    self.inverse_stft_window_fn = inverse_stft_window_fn
    self.fft_size = fft_size
    self.inference_batch_size = inference_batch_size
    self.mode = mode
    self.use_one_step = use_one_step
    self.state_name_tag = state_name_tag

    if self.window_type not in [None, 'hann_tf']:
      raise ValueError('Usupported window_type', self.window_type)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state varaible for inference streaming with internal state
      self.states = self.add_weight(
          name=self.name + 'frame_states',
          shape=[self.inference_batch_size, self.frame_size],
          trainable=False,
          initializer=tf.zeros_initializer,
          dtype=tf.float32)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state,
      # state becomes an input output placeholders
      self.input_state = tf.keras.layers.Input(
          shape=(self.frame_size,),
          batch_size=self.inference_batch_size,
          name=self.name + 'frame_states',
          dtype=tf.float32)
      self.output_state = None

    self.window_fn = None
    self.synthesis_window_fn = None
    if self.window_type == 'hann_tf':
      self.synthesis_window_fn = functools.partial(
          tf.signal.hann_window, periodic=True)
      if self.inverse_stft_window_fn:
        self.window_fn = tf.signal.inverse_stft_window_fn(
            self.frame_step, forward_window_fn=self.synthesis_window_fn)
      else:
        self.window_fn = self.synthesis_window_fn
    else:
      self.window_fn = None

  def build(self, input_shape):
    super(InverseSTFT, self).build(input_shape)
    self.input_frames = input_shape.as_list()[1]

  def get_config(self):
    config = super(InverseSTFT, self).get_config()
    config.update({
        'frame_size': self.frame_size,
        'frame_step': self.frame_step,
        'window_type': self.window_type,
        'inverse_stft_window_fn': self.inverse_stft_window_fn,
        'fft_size': self.fft_size,
        'inference_batch_size': self.inference_batch_size,
        'mode': self.mode,
        'use_one_step': self.use_one_step,
        'state_name_tag': self.state_name_tag,
        'input_frames': self.input_frames,
    })
    return config

  def get_input_state(self):
    # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def get_output_state(self):
    # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def _streaming_internal_state(self, inputs):
    inversed_frames, new_states = self._streaming_external_state(
        inputs, self.states)
    assign_states = self.states.assign(new_states)
    with tf.control_dependencies([assign_states]):
      # use tf.identity to ensure that assign_states is executed
      return tf.identity(inversed_frames)

  def _streaming_external_state(self, inputs, state):
    state = [] if state is None else state

    # compute inversed FT of any number of input frames
    inversed_frame = tf.signal.inverse_stft(
        inputs,
        self.frame_size,
        self.frame_step,
        self.fft_size,
        window_fn=self.window_fn)
    inversed_frame = tf.cast(inversed_frame, tf.float32)

    # if there is no overlap between frames then
    # there is no need in streaming state processing
    if self.frame_size - self.frame_step <= 0:
      return inversed_frame, state

    if self.use_one_step:  # streaming with input frame by frame
      # update frame state
      new_frame_state = state + inversed_frame[:, 0:self.frame_size]

      # get output hop before frame shifting
      inversed_frames = new_frame_state[:, 0:self.frame_step]

      # shift frame samples by frame_step to the left: ring buffer
      new_frame_state = tf.concat(
          [new_frame_state, tf.zeros([1, self.frame_step])], axis=1)
      new_frame_state = new_frame_state[:, -self.frame_size:]
    else:  # streaming with several input frames
      previous_state = state + inversed_frame[:, 0:self.frame_size]

      new_frame_state = tf.concat(
          [previous_state, inversed_frame[:, self.frame_size:]], axis=1)

      # get output hops before frame shifting
      inversed_frames = new_frame_state[:,
                                        0:self.frame_step * self.input_frames]

      # shift frame samples by frame_step to the left: ring buffer
      new_frame_state = tf.concat(
          [new_frame_state, tf.zeros([1, self.frame_step])], axis=1)
      new_frame_state = new_frame_state[:, -self.frame_size:]

    return inversed_frames, new_frame_state

  def _non_streaming(self, inputs):
    # note that if not rectangular window_fn is used then,
    # the first and last reconstructed frames will be numerically different
    # from the original audio frames
    output = tf.signal.inverse_stft(
        inputs,
        self.frame_size,
        self.frame_step,
        self.fft_size,
        window_fn=self.window_fn)
    return tf.cast(output, tf.float32)
