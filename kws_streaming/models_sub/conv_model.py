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

"""Toy convolutional model."""

import tensorflow as tf
from kws_streaming.layers import quantize
from kws_streaming.layers import ring_buffer
from kws_streaming.models_sub import base_model
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import AllValuesQuantizer


class ConvModel(base_model.BaseModel):
  """Streaming and quantization aware toy conv model.

  It assumes that speech features already precomputed and feed into the model

  Attributes:
    label_count: number of labels in the model
    apply_quantization: True quantize all layers, otherwise not
    **kwargs: additional layer arguments
  """

  def __init__(self, label_count, apply_quantization, **kwargs):
    super(ConvModel, self).__init__(**kwargs)

    # create layers
    self.input_quant = quantize_layer.QuantizeLayer(
        AllValuesQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))

    self.conv1 = quantize.quantize_layer(
        tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=[1, 3],
            padding='SAME'))
    self.bn1 = quantize.quantize_layer(tf.keras.layers.BatchNormalization())
    self.relu1 = quantize.quantize_layer(tf.keras.layers.ReLU())

    self.conv2 = ring_buffer.RingBuffer(
        quantize.quantize_layer(
            tf.keras.layers.Conv2D(
                filters=2,
                kernel_size=(3, 1),
                dilation_rate=1,
                strides=2,
                use_bias=False), apply_quantization,
            quantize.NoOpActivationConfig(['kernel'], ['activation'], False)),
        use_one_step=False,
        inference_batch_size=self.inference_batch_size,
        pad_time_dim='causal')
    self.bn2 = quantize.quantize_layer(
        tf.keras.layers.BatchNormalization(),
        default_8bit_quantize_configs.NoOpQuantizeConfig())
    self.relu2 = quantize.quantize_layer(tf.keras.layers.ReLU())

    self.flatten = ring_buffer.RingBuffer(
        quantize.quantize_layer(tf.keras.layers.Flatten(), apply_quantization),
        use_one_step=True,
        inference_batch_size=self.inference_batch_size)

    self.dense = quantize.quantize_layer(
        tf.keras.layers.Dense(
            label_count, activation='softmax', use_bias=False),
        apply_quantization)

  def call(self, inputs):
    net = inputs
    net = self.input_quant(net)
    net = self.conv1(net)
    net = self.bn1(net)
    net = self.relu1(net)
    net = self.conv2(net)
    net = self.bn2(net)
    net = self.relu2(net)
    net = self.flatten(net)
    net = self.dense(net)
    return net

  @tf.function
  def stream_inference(self, inputs, states):
    net = inputs
    outputs = {}

    net = self.input_quant(net)
    net = self.conv1(net)
    net = self.bn1(net)
    net = self.relu1(net)

    # create inference graph, it is the same as in call()
    # but now every streaming aware layer will return its state
    # all states will bereturned together with the output of the model
    net, output_state = self.conv2(
        net, state=states[self.conv2.get_core_layer().name])
    output_state = tf.identity(
        output_state, name=self.conv2.get_core_layer().name)
    outputs[self.conv2.get_core_layer().name] = output_state

    net = self.bn2(net)
    net = self.relu2(net)

    net, output_state = self.flatten(
        net, state=states[self.flatten.get_core_layer().name])
    output_state = tf.identity(
        output_state, name=self.flatten.get_core_layer().name)
    outputs[self.flatten.get_core_layer().name] = output_state

    net = self.dense(net)
    outputs[self.output_tensor_name] = tf.identity(
        net, name=self.output_tensor_name)
    return outputs
