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

"""Residual wrapper."""
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class Residual(tf.keras.layers.Layer):
  """Residual wrapper for streaming residual connections.

  Residual connection is a special case for streaming.
  It should be used together with TemporalPadding and convolutional layers.
  It needs their parameters, as shown in the example:
    residual_model() in residual_test.py

  Attributes:
    mode: Training or inference modes: non streaming, streaming.
    padding: padding applied by TemporalPadding layer
    kernel_size_time: kernel size of conv layer in time dim
    step_size: step size of data fed into the model
    **kwargs: additional layer arguments
  """

  def __init__(self,
               mode=Modes.TRAINING,
               padding=None,
               kernel_size_time=0,
               step_size=None,
               **kwargs):
    super(Residual, self).__init__(**kwargs)
    self.mode = mode
    self.padding = padding
    self.kernel_size_time = kernel_size_time
    self.step_size = step_size

  def build(self, input_shape):
    super(Residual, self).build(input_shape)

    # if step size was not provided in the constructor, then
    # model has to be built by calling model.build(input_shape))
    # so that step_size is inferred
    self.step_size = input_shape.as_list()[1] - self.kernel_size_time + 1

  def call(self, inputs):
    if not self.step_size:
      raise ValueError('step_size is not initialized: model should be built '
                       'with static time input shape, or model.build should be '
                       'called with static time input shape')

    if self.mode in [
        Modes.STREAM_INTERNAL_STATE_INFERENCE,
        Modes.STREAM_EXTERNAL_STATE_INFERENCE
    ]:
      if self.padding == 'causal':
        return inputs[:, -self.step_size:, :]  # pylint: disable=invalid-unary-operand-type
      elif self.padding == 'same':
        return inputs[:, (self.time_kernel_size //
                          2):(self.time_kernel_size // 2) + self.step_size, :]
      else:
        raise ValueError('wrong padding ', self.padding)
    elif self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      return inputs
    else:
      raise ValueError('wrong mode', self.mode)

  def get_config(self):
    config = super(Residual, self).get_config()
    config.update({
        'mode': self.mode,
        'padding': self.padding,
        'kernel_size_time': self.kernel_size_time,
        'step_size': self.step_size,
    })
    return config

  def get_input_state(self):
    return []

  def get_output_state(self):
    return []
