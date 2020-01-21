# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Implements keras layers producing ranks and sorted values."""

from typing import Tuple
import gin
import tensorflow.compat.v2 as tf

from soft_sort import ops


@gin.configurable
class SoftSortLayer(tf.keras.layers.Layer):
  """A layer to sort values according to an axis.

  The output can be the whole values, sorted in ascending or descending order,
  or optionally the top K values or a quantization of the sorted values.
  """

  def __init__(self, axis = -1, topk = None, **kwargs):
    """Initializes the layer.

    Args:
     axis: (int) the axis along which to sort.
     topk: (int or None) if not None, the number of topk values to return. It is
      strongly advised to use topk along with an properly input scaling
      function.
     **kwargs: other soft sorting parameters.
    """
    super().__init__(name='soft_sort', trainable=False)
    self._topk = topk
    self._axis = axis
    self._kwargs = kwargs

  def compute_output_shape(self, input_shape):
    output_shape = input_shape
    if self._topk is not None:
      axis = tf.math.mod(self._axis, tf.shape(input_shape)[0])
      output_shape = input_shape + tf.scatter_nd(
          indices=[[axis]],
          updates=[self._topk - input_shape[axis]],
          shape=tf.shape(input_shape))
    return tf.TensorShape(output_shape)

  def call(self, inputs):
    outputs = ops.softsort(
        inputs, axis=self._axis, topk=self._topk, **self._kwargs)

    if self._topk is not None:
      return outputs
    # For some reason, when doing a full sort, tf has a hard time computing the
    # shape of the output tensor. To specify the shape to tf we use the trick
    # to call tf.reshape.
    return tf.reshape(outputs, tf.shape(inputs))


@gin.configurable
class SoftRanksLayer(tf.keras.layers.Layer):
  """A layer to turn the input values into their soft ranks."""

  def __init__(self, axis = -1, **kwargs):
    super().__init__(name='soft_ranks', trainable=False)
    self._axis = axis
    self._kwargs = kwargs

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)

  def call(self, inputs):
    outputs = ops.softranks(inputs, axis=self._axis, **self._kwargs)
    return tf.reshape(outputs, tf.shape(inputs))


@gin.configurable
class SoftQuantilesLayer(tf.keras.layers.Layer):
  """A layer computing the quantiles of input values.

  It seems like TF has a hard time understanding the output shape of the
  soft quantile operator. Therefore we propose here to explicitly add the output
  shape (it can be left None in Eager mode).
  """

  def __init__(self,
               quantiles,
               output_shape = None,
               axis = -1,
               **kwargs):
    super().__init__(name='soft_quantiles', trainable=False)
    self._quantiles = quantiles
    self._output_shape = output_shape
    self._axis = axis
    self._kwargs = kwargs

  def get_output_shape(self, input_shape):
    if self._output_shape is not None:
      return self._output_shape

    axis = tf.math.mod(self._axis, tf.shape(input_shape)[0])
    return input_shape + tf.scatter_nd(
        indices=[[axis]],
        updates=[len(self._quantiles) - input_shape[axis]],
        shape=tf.shape(input_shape))

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(self.get_output_shape(input_shape))

  def call(self, inputs):
    # For some reason, in graph mode the reshape is necessary.
    return tf.reshape(
        ops.softquantiles(
            inputs, self._quantiles, axis=self._axis, **self._kwargs),
        self.get_output_shape(tf.shape(inputs)))
