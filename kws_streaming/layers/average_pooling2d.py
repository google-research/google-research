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

"""Convolutional AveragePooling2D."""
import numpy as np
from kws_streaming.layers.compat import tf


class AveragePooling2D(tf.keras.layers.Layer):
  """AveragePooling2D layer.

  It is convolutional AveragePooling2D based on depthwise_conv2d.
  It can be useful for cases where AveragePooling2D has to run in streaming mode

  The input data with shape [batch_size, time1, feature1, feature2]
  are processed by depthwise conv with fixed weights, all weights values
  are equal to 1.0/(size_in_time_1*size_in_feature1).
  Averaging is done in 'time1' and 'feature1' dims.
  Conv filter has size [size_in_time_1, size_in_feature1, feature2],
  where first two dims are specified by user and
  feature2 is defiend by the last dim of input data.

  So if kernel_size = [time1, feature1]
  output will be [batch_size, time1, 1, feature2]

  Attributes:
    kernel_size: 2D kernel size - defines the dims
      which will be eliminated/averaged.
    strides: stride for each dim, with size 4
    padding: defiens how to pad
    dilation_rate: dilation rate in which we sample input values
      across the height and width
    **kwargs: additional layer arguments
  """

  def __init__(self,
               kernel_size,
               strides=None,
               padding='valid',
               dilation_rate=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.dilation_rate = dilation_rate
    if not self.strides:
      self.strides = [1, 1, 1, 1]

    if not self.dilation_rate:
      self.dilation_rate = [1, 1]

  def build(self, input_shape):
    super(AveragePooling2D, self).build(input_shape)
    # expand filters shape with the last dimension
    filter_shape = self.kernel_size + (input_shape[-1],)
    self.filters = self.add_weight('kernel', shape=filter_shape)

    init_weight = np.ones(filter_shape) / np.prod(self.kernel_size)
    self.set_weights([init_weight])

  def call(self, inputs):
    # inputs [batch_size, time1, feature1, feature2]
    time_kernel_exp = tf.expand_dims(self.filters, -1)
    # it can be replaced by AveragePooling2D with temporal padding
    # and optimized for streaming mode
    # output will be [batch_size, time1, feature1, feature2]
    return tf.nn.depthwise_conv2d(
        inputs,
        time_kernel_exp,
        strides=self.strides,
        padding=self.padding.upper(),
        dilations=self.dilation_rate,
        name=self.name + '_averPool2D')

  def get_config(self):
    config = super(AveragePooling2D, self).get_config()
    config.update({
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'dilation_rate': self.dilation_rate,
    })
    return config
