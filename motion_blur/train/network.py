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

"""Motion blur "Line Prediction Network" architecture..

Learning to Synthesize Motion Blur
http://timothybrooks.com/tech/motion-blur
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def apply_line_prediction(inputs,
                          features,
                          blur_steps,
                          learn_alpha=True,
                          name=None):
  """Applies "Line Prediction" layer to input images."""
  inputs.shape.assert_is_compatible_with([None, None, None, 6])

  with tf.name_scope(name, 'blur_prediction', values=[inputs, features]):

    with tf.name_scope(None, 'input_frames', values=[inputs]):
      frames = [inputs[:, :, :, :3], inputs[:, :, :, 3:]]

    with tf.name_scope(None, 'frame_size', values=[inputs, features]):
      shape = tf.shape(inputs)
      height = shape[1]
      width = shape[2]

    with tf.name_scope(None, 'identity_warp', values=[]):
      x_idx, y_idx = tf.meshgrid(tf.range(width), tf.range(height))
      identity_warp = tf.to_float(tf.stack([x_idx, y_idx], axis=-1))
      identity_warp = identity_warp[tf.newaxis, :, :, tf.newaxis, :]

      warp_steps = tf.to_float(tf.range(blur_steps - 1) + 1) / (blur_steps - 1)
      warp_steps = warp_steps[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]

      max_warps = tf.to_float(tf.stack([width - 1, height - 1]))
      max_warps = max_warps[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]

    output_frames = []
    for frame in frames:
      with tf.name_scope(None, 'predict_blurs', values=[features]):
        flow = tf.layers.conv2d(features, 2, 1, padding='same')

        if learn_alpha:
          alpha = tf.layers.conv2d(
              features, blur_steps, 1, padding='same', activation=tf.nn.softmax)

      with tf.name_scope(None, 'apply_blurs', values=[]):
        with tf.name_scope(None, 'warp', values=[frame, flow]):
          warps = identity_warp + flow[:, :, :, tf.newaxis, :] * warp_steps
          warps = tf.clip_by_value(warps, 0.0, max_warps)
          warped = tf.contrib.resampler.resampler(frame, warps)
          warped = tf.concat([frame[:, :, :, tf.newaxis, :], warped], axis=3)

        with tf.name_scope(None, 'apply_alpha', values=[frame, flow]):
          if learn_alpha:
            mask = alpha[:, :, :, :, tf.newaxis]
          else:
            mask = 1.0 / blur_steps
          output_frames.append(tf.reduce_sum(warped * mask, axis=3))

    with tf.name_scope(None, 'outputs', values=[output_frames]):
      output = tf.add_n(output_frames) / len(frames)
      return output


def pad(features, multiple, name=None):
  """Pads height and width of tensor to a multiple of `multiple`."""
  with tf.name_scope(name, 'pad', values=[features]):
    height = tf.shape(features)[1]
    width = tf.shape(features)[2]

    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padding = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    features = tf.pad(features, padding, mode='CONSTANT')

    batch_size = tf.shape(features)[0]
    height = tf.shape(features)[1]
    width = tf.shape(features)[2]

    return tf.reshape(features, [batch_size, height, width, 6])


def crop(features, height, width, name=None):
  """Crops a rectangle with `height` and `width` from `features`."""
  with tf.name_scope(name, 'crop', values=[features]):
    y0 = (tf.shape(features)[1] - height) // 2
    x0 = (tf.shape(features)[2] - width) // 2
    return features[:, y0:y0 + height, x0:x0 + width, :]


def apply_conv(features, num_channels, name=None):
  """Applies a 3x3 spatial convolution with leaky relu activation."""
  return tf.layers.conv2d(
      features,
      num_channels,
      3,
      padding='same',
      activation=tf.nn.leaky_relu,
      name=name)


def inference(frame_0,
              frame_1,
              blur_steps=17,
              num_convs=3,
              levels=((32, 64), (64, 64), (128, 128), (256, 256), 256),
              is_training=False):
  """Creates the graph that generates the motion blur from two input frames."""
  del is_training  # Unused.

  inputs = tf.concat([frame_0, frame_1], axis=-1)
  inputs = tf.identity(inputs, 'inputs')

  multiple = 2**(len(levels) - 1)
  features = pad(inputs, multiple)

  skip_connections = []

  with tf.name_scope('encoder', values=[features, skip_connections]):
    for num_channels, _ in levels[:-1]:
      for _ in range(num_convs):
        features = apply_conv(features, num_channels)

      skip_connections.append(features)
      features = tf.layers.max_pooling2d(features, 2, 2, padding='same')

  with tf.name_scope('latent_space', values=[features]):
    num_channels = levels[-1]
    for _ in range(num_convs):
      features = apply_conv(features, num_channels)

  with tf.name_scope('decoder', values=[features, skip_connections]):
    for _, num_channels in levels[-2::-1]:
      with tf.name_scope(None, 'bilinear_upsample', values=[features]):
        shape = tf.shape(features)
        shape = [shape[1] * 2, shape[2] * 2]
        features = tf.image.resize_bilinear(features, shape)

      with tf.name_scope(
          None, 'skip_connection', values=[features, skip_connections]):
        features = tf.concat([features, skip_connections.pop()], axis=-1)

      for _ in range(num_convs):
        features = apply_conv(features, num_channels)

  with tf.name_scope('frame_size', values=[inputs, features]):
    shape = tf.shape(inputs)
    height = shape[1]
    width = shape[2]

  features = crop(features, height, width)
  output = apply_line_prediction(inputs, features, blur_steps)
  return tf.identity(output, 'output')
