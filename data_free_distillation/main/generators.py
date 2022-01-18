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

# Lint as: python3
"""Generator models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import List

import tensorflow.compat.v1 as tf
import tf_slim as slim


def simple_generator(z,
                     image_size,
                     num_interpolate = 2,
                     channels = None,
                     depthwise_separate = None,
                     output_bn = True,
                     is_training = True,
                     reuse=None,
                     scope=None):
  """A simple generator model used in the paper."""
  # The generator structure is originally defined in "XNOR-Net: ImageNet
  # Classification Using Binary Convolutional Neural Networks"
  # https://arxiv.org/pdf/1603.05279.pdf

  if not channels:
    # default: [128, 64]
    channels = [128 // (i + 1) for i in range(num_interpolate)]
  if not depthwise_separate:
    # default: no depthwise separate conv
    depthwise_separate = [False] * num_interpolate

  # noinspection PyTypeChecker
  assert len(channels) == len(depthwise_separate) == num_interpolate

  init_size = image_size // (2**num_interpolate)
  resize = functools.partial(
      tf.image.resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  with tf.variable_scope(scope, 'generator', [z], reuse=reuse):
    # noinspection PyCallingNonCallable
    with slim.arg_scope([slim.batch_norm],
                        decay=0.9,
                        center=True,
                        scale=True,
                        epsilon=0.8,
                        is_training=is_training):
      # noinspection PyCallingNonCallable
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                          activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.batch_norm):
        x = slim.fully_connected(
            z,
            init_size * init_size * channels[0],
            activation_fn=None,
            biases_initializer=None,
            scope='dense')
        x = tf.reshape(x, [-1, init_size, init_size, channels[0]])
        # The code of the DAFL paper uses different epsilon values for batch
        # normalization layers. We keep these settings for reproducibility.
        # See https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/DAFL-train.py#L54  pylint: disable=line-too-long
        # for details.
        x = slim.batch_norm(x, epsilon=1e-5, scope='bn_0')
        x = tf.nn.leaky_relu(x)

        # Interpolate layers
        size = init_size
        for i, (n_channels, ds) in enumerate(zip(channels, depthwise_separate)):
          size *= 2
          x = resize(x, [size, size], name='interpolate_{}'.format(i))
          if not ds:
            x = slim.conv2d(x, n_channels, [3, 3], scope='conv_{}'.format(i))
          else:
            x = slim.separable_conv2d(
                x, None, [3, 3], scope='conv_{}_depthwise'.format(i))
            x = slim.conv2d(
                x, n_channels, [1, 1], scope='conv_{}_pointwise'.format(i))

        # Output layer
        x = slim.conv2d(
            x,
            3, [3, 3],
            activation_fn=tf.nn.tanh,
            normalizer_fn=None,
            scope='conv_{}'.format(num_interpolate))

        if output_bn:
          x = slim.batch_norm(
              x,
              center=False,
              scale=True,
              scope='bn_output',
              is_training=is_training)

      return x


def conditioned_generator(z,
                          one_hot_label,
                          image_size,
                          num_interpolate = 2,
                          channels = None,
                          depthwise_separate = None,
                          output_bn = True,
                          is_training = True,
                          reuse=None,
                          scope=None):
  """Conditioned generator which outputs image conditioned on the input label."""
  with tf.variable_scope(
      scope, 'conditioned_generator', [z, one_hot_label], reuse=reuse):
    z = tf.concat([z, one_hot_label], axis=1)
  return simple_generator(
      z,
      image_size,
      num_interpolate=num_interpolate,
      channels=channels,
      depthwise_separate=depthwise_separate,
      output_bn=output_bn,
      is_training=is_training,
      reuse=reuse,
      scope=scope)


# This is the function we would typically use in generator training and
# distillation script. We wrap the `conditioned_generator` and name it as
# `generator` for compatibility.
def generator(z,
              label,
              image_size,
              n_classes,
              num_interpolate = 2,
              channels = None,
              depthwise_separate = None,
              output_bn = True,
              is_training = True,
              reuse=None,
              scope='generator'):
  """Generator."""
  one_hot_label = tf.one_hot(label, n_classes)
  return conditioned_generator(
      z,
      one_hot_label,
      image_size,
      num_interpolate=num_interpolate,
      channels=channels,
      depthwise_separate=depthwise_separate,
      output_bn=output_bn,
      is_training=is_training,
      reuse=reuse,
      scope=scope)
