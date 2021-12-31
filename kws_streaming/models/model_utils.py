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

"""Utility functions for models."""
import ast
from kws_streaming.layers.compat import tf


def parse(text):
  """Parse model parameters.

  Args:
    text: string with layer parameters: '128,128' or "'relu','relu'".

  Returns:
    list of parsed parameters
  """
  if not text:
    return []
  res = ast.literal_eval(text)
  if isinstance(res, tuple):
    return res
  else:
    return [res]


def conv2d_bn(x,
              filters,
              kernel_size,
              padding='same',
              strides=(1, 1),
              activation='relu',
              use_bias=False,
              scale=False):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    kernel_size: size of convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    activation: activation function applied in the end.
    use_bias: use bias for convolution.
    scale: scale batch normalization.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """

  x = tf.keras.layers.Conv2D(
      filters, kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias)(x)
  x = tf.keras.layers.BatchNormalization(scale=scale)(x)
  x = tf.keras.layers.Activation(activation)(x)
  return x
