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

"""Base model definition."""

import tensorflow as tf
from kws_streaming.models_sub import utils


class BaseModel(tf.keras.Model):
  """BaseModel for streaming aware models based on keras subclass API.

  It can be used to wrap Keras layer for streaming inference mode.
  It is similar with Stream wrapper, but it is applicable only
  for subclass modesl with external state:
  user will have to manage extrenal state

  Attributes:
    inference_batch_size: batch size in inference mode
    state_type: type of the states used by the model
    state_shape: shape of the state/ring_buffer specified by user
    ring_buffer_size_in_time_dim: size of ring buffer in time dim
    use_one_step: True - model will run one sample per one inference step;
      False - model will run multiple per one inference step.
      It is useful for strided streaming
    **kwargs: additional layer arguments

  Raises:
    NotImplementedError: if call() or stream_inference are not implemented
    ValueError: if model is not built
  """

  def __init__(self,
               inference_batch_size=1,
               state_type=tf.dtypes.float32,
               input_name='input_0',
               output_name='output_0',
               **kwargs):
    super(BaseModel, self).__init__(**kwargs)
    self.inference_batch_size = inference_batch_size
    self.state_type = state_type
    self._input_tensor_name = 'input_0'
    self._output_tensor_name = 'output_0'

  def build(self, input_shape):
    super(BaseModel, self).build(input_shape)

    self._stride = 1
    self.states_shape = {}
    for layer in self.layers:
      if utils.method_exists(layer, 'get_input_state_shape'):
        self.states_shape[
            layer.get_core_layer().name] = layer.get_input_state_shape()
      if hasattr(layer, 'stride'):
        self._stride *= layer.stride

  def states(self, return_np=False):
    if self.built:
      states = {}
      for key, state_shape in self.states_shape.items():
        state_shape[0] = self.inference_batch_size
        states[key] = tf.zeros(state_shape, dtype=self.state_type, name=key)
        if return_np:
          states[key] = states[key].numpy()
      return states
    else:
      raise ValueError('model is not built')

  @property
  def input_tensor_name(self):
    return self._input_tensor_name

  @property
  def output_tensor_name(self):
    return self._output_tensor_name

  @property
  def stride(self):
    if self.built:
      return self._stride
    else:
      raise ValueError('model is not built')

  def call(self, inputs):
    raise NotImplementedError('call is not implemented.')

  @tf.function
  def stream_inference(self, inputs, states):
    raise NotImplementedError('stream_inference is not implemented.')
