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

# Lint as: python3
"""Builds the Keras Shake-Shake Model.

This is the model described in https://arxiv.org/abs/1705.07485.

"""
import tensorflow.compat.v2 as tf


def _shake_shake_skip_connection(layer_input, output_filters, stride,
                                 weight_decay):
  """Adds a residual connection to the layer_input for the Shake-Shake model.

  This Shake-Shake skip connection layer forms the basis of the residual
  (identity) component of a Shake-Shake block.

  Args:
    layer_input: Tensor or Keras layer.
    output_filters: Integer representing the number of output filters.
    stride: Integer representing the stride.

  Returns:
    Returns a Shake-Shake skip connection keras layer.
  """
  curr_filters = int(layer_input.shape[3])
  if curr_filters == output_filters:
    return layer_input
  stride_spec = [1, stride, stride, 1]
  path1 = tf.nn.avg_pool(
      layer_input, [1, 1, 1, 1], stride_spec, "VALID", data_format="NHWC")
  path1 = tf.keras.layers.Conv2D(
      filters=int(output_filters / 2),
      kernel_size=1,
      strides=1,
      padding="SAME",
      data_format="channels_last",
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          path1)
  path2 = tf.keras.layers.ZeroPadding2D(
      padding=[[0, 1], [0, 1]], data_format="channels_last")(
          layer_input)
  path2 = tf.keras.layers.Cropping2D(
      cropping=[[1, 0], [1, 0]], data_format="channels_last")(
          path2)
  concat_axis = 3
  path2 = tf.nn.avg_pool(
      path2, [1, 1, 1, 1], stride_spec, "VALID", data_format="NHWC")
  path2 = tf.keras.layers.Conv2D(
      filters=int(output_filters / 2),
      kernel_size=1,
      strides=1,
      padding="SAME",
      data_format="channels_last",
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          path2)
  final_path = tf.concat(values=[path1, path2], axis=concat_axis)
  final_path = tf.keras.layers.BatchNormalization(
      momentum=0.999, center=True, scale=False, epsilon=0.001)(
          final_path)
  return final_path


def _shake_shake_branch(layer_input, output_filters, stride, rand_forward,
                        rand_backward, weight_decay):
  """Building a 2 branching convnet, part of a Shake-Shake block.

  Component of a Shake-Shake block.

  Args:
    layer_input: Input Keras layer to the Shake-Shake branch.
    output_filters: Output filters.
    stride: The stride size.
    rand_forward: Uniform random tensor of dimensionality [batch_size, 1,1,1]
    rand_backward: Uniform random tensor of dimensionality [batch_size, 1,1,1]

  Returns:
    A Shake-Shake branch keras layer.
  """
  layers = [
      tf.keras.layers.ReLU(),
      # tf.keras.layers.BatchNormalization(
      #     momentum=0.999, center=True, scale=False, epsilon=.001),
      tf.keras.layers.Conv2D(
          filters=output_filters,
          kernel_size=3,
          strides=stride,
          padding="SAME",
          data_format="channels_last",
          kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
      tf.keras.layers.BatchNormalization(
          momentum=0.999, center=True, scale=False, epsilon=0.001),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Conv2D(
          filters=output_filters,
          kernel_size=3,
          strides=1,
          padding="SAME",
          data_format="channels_last",
          kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
      tf.keras.layers.BatchNormalization(
          momentum=0.999, center=True, scale=False, epsilon=0.001)
  ]
  for layer in layers:
    layer_input = layer(layer_input)
  if tf.keras.backend.learning_phase():  # training
    layer_input = layer_input * rand_backward + tf.stop_gradient(
        layer_input * rand_forward - layer_input * rand_backward)
  else:
    layer_input *= 1.0 / 2
  return layer_input


def _shake_shake_block(layer_input,
                       output_filters,
                       stride,
                       weight_decay,
                       tag=""):
  """Builds a full Shake-Shake sub layer made of Shake-Shake branches.

  Args:
    layer_input: Input Keras layer.
    output_filters: Defines the number of output filters of the layer.
    stride: Defines the stride of the shake shake layer block.
    tag: String. Name tag for this shake shake block.

  Returns:
    A Shake-Shake Keras layer block.
  """
  batch_size = tf.shape(layer_input)[0]
  rand_forward = [
      # pylint: disable=g-complex-comprehension
      tf.random.uniform([batch_size, 1, 1, 1],
                        minval=0,
                        maxval=1,
                        dtype=tf.float32,
                        name="{}_1_{}".format(tag, i)) for i in range(2)
  ]
  rand_backward = [
      # pylint: disable=g-complex-comprehension
      tf.random.uniform([batch_size, 1, 1, 1],
                        minval=0,
                        maxval=1,
                        dtype=tf.float32,
                        name="{}_2_{}".format(tag, i)) for i in range(2)
  ]

  total_forward = tf.add_n(rand_forward)
  total_backward = tf.add_n(rand_backward)
  rand_forward = [samp / total_forward for samp in rand_forward]
  rand_backward = [samp / total_backward for samp in rand_backward]
  zipped_rand = zip(rand_forward, rand_backward)
  branches = []
  for _, (r_forward, r_backward) in enumerate(zipped_rand):
    b = _shake_shake_branch(layer_input, output_filters, stride, r_forward,
                            r_backward, weight_decay)
    branches.append(b)
  res = _shake_shake_skip_connection(layer_input, output_filters, stride,
                                     weight_decay)
  return res + tf.add_n(branches)


def _shake_shake_layer(layer_input,
                       output_filters,
                       num_blocks,
                       stride,
                       weight_decay,
                       tag=""):
  """Builds many sub layers into one full layer.

  Args:
    layer_input: Keras layer. Input layer.
    output_filters: Defines the number of output filters of the layer.
    num_blocks: Defines the number of Shake-Shake blocks this layer will have.
    stride: Defines the stride of the Shake-Shake layer's blocks.
    tag: Defines the name of some component tensors of the layer.

  Returns:
    A Shake-Shake layer.
  """
  for block_num in range(num_blocks):
    curr_stride = stride if (block_num == 0) else 1
    layer_input = _shake_shake_block(
        layer_input,
        output_filters,
        curr_stride,
        weight_decay,
        tag="{}_{}".format(tag, block_num))

  return layer_input


def build_shake_shake_model(num_classes,
                            depth,
                            width,
                            weight_decay,
                            image_shape=(32, 32, 3)):
  """Builds the Shake-Shake model.

  Build the Shake-Shake model from https://arxiv.org/abs/1705.07485.

  Args:
    image_shape: Shape of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    depth: Depth of the model.
    width: Width of the model.

  Returns:
    Returns the Shake-Shake model.
  """
  n = int((depth - 2) / 6)

  inputs = tf.keras.Input(shape=image_shape)
  x = inputs

  x = tf.keras.layers.Conv2D(
      filters=16,
      kernel_size=3,
      strides=1,
      padding="SAME",
      data_format="channels_last",
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          x)
  x = tf.keras.layers.BatchNormalization(
      momentum=0.999, center=True, scale=False, epsilon=0.001)(
          x)

  x = _shake_shake_layer(x, 16 * width, n, 1, weight_decay, tag="layer1")
  x = _shake_shake_layer(x, 32 * width, n, 2, weight_decay, tag="layer2")
  x = _shake_shake_layer(x, 64 * width, n, 2, weight_decay, tag="layer3")

  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)

  logits = tf.keras.layers.Dense(num_classes)(x)
  model = tf.keras.Model(inputs=inputs, outputs=logits, name="ShakeShake")
  return model
