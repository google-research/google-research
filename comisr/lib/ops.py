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

"""Ops overwritten."""

from cvx2 import latest as cv
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import tf_slim as slim

keras = tf.keras


# Define our Lrelu
def lrelu(inputs, alpha):
  return keras.layers.LeakyReLU(alpha=alpha).call(inputs)


def preprocess(image):
  with tf.name_scope('preprocess'):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
  with tf.name_scope('deprocess'):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def maxpool(inputs, scope='maxpool'):
  return slim.max_pool2d(inputs, [2, 2], scope=scope)


def conv2_tran(batch_input,
               kernel=3,
               output_channel=64,
               stride=1,
               use_bias=True,
               scope='conv'):
  """Define the convolution transpose building block."""
  with tf.variable_scope(scope):
    if use_bias:
      return slim.conv2d_transpose(
          batch_input,
          output_channel, [kernel, kernel],
          stride,
          'SAME',
          data_format='NHWC',
          activation_fn=None,)
    else:
      return slim.conv2d_transpose(
          batch_input,
          output_channel, [kernel, kernel],
          stride,
          'SAME',
          data_format='NHWC',
          activation_fn=None,
          biases_initializer=None)


def conv2(batch_input,
          kernel=3,
          output_channel=64,
          stride=1,
          use_bias=True,
          scope='conv'):
  """Define the convolution building block."""
  with tf.variable_scope(scope):
    if use_bias:
      return slim.conv2d(
          batch_input,
          output_channel, [kernel, kernel],
          stride,
          'SAME',
          data_format='NHWC',
          activation_fn=None)
    else:
      return slim.conv2d(
          batch_input,
          output_channel, [kernel, kernel],
          stride,
          'SAME',
          data_format='NHWC',
          activation_fn=None,
          biases_initializer=None)


def upscale_x(
    inputs,
    scale=4,
    scope='upscale_x'
):
  """mimic the tensorflow bilinear-upscaling for a fix ratio of x."""
  with tf.variable_scope(scope):
    size = tf.shape(inputs)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]

    p_inputs = tf.concat((inputs, inputs[:, -1:, :, :]), axis=1)  # pad bottom
    p_inputs = tf.concat((p_inputs, p_inputs[:, :, -1:, :]),
                         axis=2)  # pad right

    hi_res_bin = [
        [
            inputs,  # top-left
            p_inputs[:, :-1, 1:, :]  # top-right
        ],
        [
            p_inputs[:, 1:, :-1, :],  # bottom-left
            p_inputs[:, 1:, 1:, :]  # bottom-right
        ]
    ]

    hi_res_array = []
    factor = 1.0 / float(scale)
    for hi in range(scale):
      for wj in range(scale):
        hi_res_array.append(hi_res_bin[0][0] * (1.0 - factor * hi) *
                            (1.0 - factor * wj) + hi_res_bin[0][1] *
                            (1.0 - factor * hi) * (factor * wj) +
                            hi_res_bin[1][0] * (factor * hi) *
                            (1.0 - factor * wj) + hi_res_bin[1][1] *
                            (factor * hi) * (factor * wj))

    hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h,w,16,c)
    hi_res_reshape = tf.reshape(hi_res, (b, h, w, scale, scale, c))
    hi_res_reshape = tf.transpose(hi_res_reshape, (0, 1, 3, 2, 4, 5))
    hi_res_reshape = tf.reshape(hi_res_reshape, (b, h * scale, w * scale, c))

  return hi_res_reshape


def bicubic_x(inputs, scale=4, scope='bicubic_x'):
  """Upscaling using tf.bicubic function."""
  with tf.variable_scope(scope):
    if scale == 4:
      return bicubic_four(inputs)
    size = tf.shape(inputs)
    output_size = [scale * size[1], scale * size[2]]

    bicubic_x_inputs = tf2.image.resize(
        inputs, output_size, method=tf2.image.ResizeMethod.BICUBIC)
  return bicubic_x_inputs


def bicubic_four(inputs, scope='bicubic_four'):
  """Bicubic four upscaling."""

  with tf.variable_scope(scope):
    size = tf.shape(inputs)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]

    p_inputs = tf.concat((inputs[:, :1, :, :], inputs), axis=1)  # pad top
    p_inputs = tf.concat((p_inputs[:, :, :1, :], p_inputs), axis=2)  # pad left
    p_inputs = tf.concat(
        (p_inputs, p_inputs[:, -1:, :, :], p_inputs[:, -1:, :, :]),
        axis=1)  # pad bottom
    p_inputs = tf.concat(
        (p_inputs, p_inputs[:, :, -1:, :], p_inputs[:, :, -1:, :]),
        axis=2)  # pad right

    hi_res_bin = [p_inputs[:, bi:bi + h, :, :] for bi in range(4)]
    r = 0.75
    mat = np.float32([[0, 1, 0, 0], [-r, 0, r, 0],
                      [2 * r, r - 3, 3 - 2 * r, -r], [-r, 2 - r, r - 2, r]])
    weights = [
        np.float32([1.0, t, t * t, t * t * t]).dot(mat)
        for t in [0.0, 0.25, 0.5, 0.75]
    ]

    hi_res_array = []
    for hi in range(4):
      cur_wei = weights[hi]
      cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[
          1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]

      hi_res_array.append(cur_data)

    hi_res_y = tf.stack(hi_res_array, axis=2)  # shape (b,h,4,w,c)
    hi_res_y = tf.reshape(hi_res_y, (b, h * 4, w + 3, c))

    hi_res_bin = [hi_res_y[:, :, bj:bj + w, :] for bj in range(4)]

    hi_res_array = []
    for hj in range(4):
      cur_wei = weights[hj]
      cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[
          1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]

      hi_res_array.append(cur_data)

    hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h*4,w,4,c)
    hi_res = tf.reshape(hi_res, (b, h * 4, w * 4, c))

  return hi_res


def save_img(out_path, img):
  """Save an image to disk."""
  img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
  cv.imwrite(out_path, img[:, :, ::-1])
