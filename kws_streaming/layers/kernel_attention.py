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

"""KernelAttention streaming aware layer."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from official.nlp.modeling import layers as nlp_layers


class KernelAttention(nlp_layers.kernel_attention.KernelAttention):
  """Streaming aware KernelAttention layer."""

  def __init__(self,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               **kwargs):
    super().__init__(**kwargs)

    if not self.use_causal_windowed:
      raise ValueError('Streaming of KernelAttention is not supported for '
                       'use_causal_windowed = False.')

    self.mode = mode
    self.inference_batch_size = inference_batch_size

  def build(self, input_shape):
    super().build(input_shape)

    if self.mode in [
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    ]:
      if len(input_shape) != 2:
        raise ValueError('input_shape must be a list with 2 elements')
      if input_shape[0] != input_shape[1]:
        raise ValueError(f'input_shapes of the input tensors must have '
                         f'the same dimensions but get'
                         f' input_shape[0]: {input_shape[0]} and '
                         f' input_shape[1]: {input_shape[1]}')
      if input_shape[0].as_list()[1] is None:
        raise ValueError('in streaming mode time dimension of input packet '
                         'should not be dynamic: TFLite limitation')

      if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
        self.kv_cache = self.add_weight(
            name='kv_cache',
            shape=(self.inference_batch_size, self._num_heads, self._key_dim,
                   self._key_dim),
            trainable=False,
            initializer=tf.zeros_initializer)
        self.k_sum_cache = self.add_weight(
            name='k_sum_cache',
            shape=(self.inference_batch_size, self._num_heads, self._key_dim),
            trainable=False,
            initializer=tf.zeros_initializer)
      elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
        raise ValueError('Streaming with external states is not implemented.')

  def call(self, inputs, training=None):
    if len(inputs) != 2:
      raise ValueError('inputs has to have 2 elements')
    if inputs[0].shape.rank != inputs[1].shape.rank:
      raise ValueError(f'Input tensors must have the same rank, but get'
                       f' inputs[0].shape: {inputs[0].shape} and '
                       f' inputs[1].shape: {inputs[1].shape}')

    if inputs[0].shape.rank != 4:
      raise ValueError(f'Input has to have rank 4, but get {inputs[0].shape}')

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      return self._streaming_internal_state(inputs)

    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      raise ValueError('Streaming with external states is not implemented.')

    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      return self._non_streaming(inputs)
    else:
      raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

  def get_config(self):
    config = super().get_config()
    config.update({
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
    })
    return config

  def _streaming_internal_state(self, inputs, training=None):
    if None in inputs[0].shape[1:]:
      raise ValueError(f'In streaming mode dynamic shape is not supported, '
                       f'get input tensor with shape: {inputs[0].shape} ')

    query = inputs[0]
    value = inputs[1]

    stream_output = []
    cache = {'kv': self.kv_cache, 'k_sum': self.k_sum_cache}
    time_size = query.shape[1]
    causal_chunk_length = 1
    # TODO(rybakov) Check why there is numerical difference without for loop.
    for i in range(time_size):
      query_step = query[:, i * causal_chunk_length:(i + 1) *
                         causal_chunk_length, :]

      query_step_shape = tf.shape(query_step)
      query_step_1d = tf.reshape(query_step,
                                 (query_step_shape[0], -1, query_step_shape[3]))

      value_step = value[:, i * causal_chunk_length:(i + 1) *
                         causal_chunk_length, :]
      value_step_shape = tf.shape(value_step)
      value_step_1d = tf.reshape(value_step,
                                 (value_step_shape[0], -1, value_step_shape[3]))

      step_residual1d = super().call(
          query=query_step_1d,
          value=value_step_1d,
          cache=cache,
          training=training)

      residual_step = tf.reshape(step_residual1d, query_step_shape)
      stream_output.append(residual_step)

    stream_output = tf.concat(stream_output, axis=1)

    assign_kv_states = self.kv_cache.assign(cache['kv'])
    assign_k_sum_cache_states = self.k_sum_cache.assign(cache['k_sum'])

    with tf.control_dependencies([assign_kv_states, assign_k_sum_cache_states]):
      return tf.identity(stream_output)

  def _streaming_external_state(self, inputs, states, training=None):
    raise ValueError('Streaming with external states is not implemented.')

  def _non_streaming(self, inputs, training=None):

    query = inputs[0]
    value = inputs[1]

    query_shape = tf.shape(query)
    query_1d = tf.reshape(query, (query_shape[0], -1, query_shape[3]))

    value_shape = tf.shape(value)
    value_1d = tf.reshape(value, (value_shape[0], -1, value_shape[3]))

    output_1d = super().call(query=query_1d, value=value_1d, training=training)
    output_1d = tf.reshape(output_1d, query_shape)
    return output_1d

  def get_input_state(self):
    raise ValueError('Streaming with external states is not implemented.')

  def get_output_state(self):
    raise ValueError('Streaming with external states is not implemented.')
