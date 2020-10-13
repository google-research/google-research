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

"""TemporalPadding layer."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf

SUPPORTED_PADDINGS = ['valid', 'causal', 'same']


class TemporalPadding(tf.keras.layers.Layer):
  """Padding in time dimension of tensor with rank >= 2.

  It is convenient for models with streaming support: in streaming mode
  it will disable padding; and in non streaming mode it will pad input data
  in time dimension [batch, time, ...].

  Attributes:
    mode: Training or inference modes: non streaming, streaming.
    padding: padding mode supports 'causal' or 'same'. 'valid' - not padding
    padding_size: how much to pad
  """

  def __init__(self,
               mode=modes.Modes.TRAINING,
               padding_size=None,
               padding=None,
               **kwargs):
    super(TemporalPadding, self).__init__(**kwargs)

    if padding not in SUPPORTED_PADDINGS:
      raise ValueError('wrong padding ', padding)

    if mode not in [modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE,
                    modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
                    modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE]:
      raise ValueError('wrong mode ', mode)

    self.mode = mode
    self.padding = padding
    self.padding_size = padding_size

  def call(self, inputs):

    if inputs.shape.rank < 2:
      raise ValueError('inputs.shape.rank: %d must be >= 2' % inputs.shape.rank)

    if self.mode in [
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    ] or self.padding == 'valid':
      # padding is not applied in streaming mode or on valid
      return inputs

    pad = [[0, 0]] * inputs.shape.rank

    if self.padding == 'causal':
      pad[1] = [self.padding_size, 0]
    elif self.padding == 'same':
      half = self.padding_size // 2
      pad[1] = [half, half]

    inputs = tf.pad(inputs, pad, 'constant')
    return inputs

  def get_config(self):
    config = super(TemporalPadding, self).get_config()
    config.update({
        'mode': self.mode,
        'padding': self.padding,
        'padding_size': self.padding_size,
    })
    return config

  def get_input_state(self):
    return []

  def get_output_state(self):
    return []
