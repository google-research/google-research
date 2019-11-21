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

# Lint as: python2, python3
"""Contains the definition for no padding inception FCN.

This is a variant of inception v3 by removing all the paddings. This change
allows the network to be trained and inference run with different patch size
(Fully Convolutional Network, FCN mode) while having the same inference results.
"""
import tensorflow as tf

slim = tf.contrib.slim


def _trim_border_px(inputs, n):
  """Crop n pixels around the border of inputs.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    n: an integer for number of pixels to crop.

  Returns:
    cropped tensor.
  Raises:
    ValueError: if cropping leads to empty output tensor.
  """
  if n > min(inputs.shape[1], inputs.shape[2]) // 2:
    raise ValueError(
        'n (%d) can not be greater than or equal to half of the input shape.' %
        n)
  return inputs[:, n:-n, n:-n, :]


def nopad_inception_v3_base(inputs,
                            min_depth=16,
                            depth_multiplier=1.0,
                            num_final_1x1_conv=0,
                            scope=None):
  """Constructs a no padding Inception v3 network from inputs.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels) for
      all convolution ops. The value must be greater than zero. Typical usage
      will be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    num_final_1x1_conv: Int, number of final 1x1 conv layers.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'NopadInceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1,
                        padding='VALID'):
      # 911 x 911 x 3
      end_point = 'Conv2d_1a_3x3'
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      # 455 x 455 x 32
      end_point = 'Conv2d_2a_3x3'
      net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      # 453 x 453 x 32
      end_point = 'Conv2d_2b_3x3'
      net = slim.conv2d(net, depth(64), [3, 3], scope=end_point)
      end_points[end_point] = net
      # 451 x 451 x 64
      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      # 225 x 225 x 64
      end_point = 'Conv2d_3b_1x1'
      net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      # 225 x 225 x 80.
      end_point = 'Conv2d_4a_3x3'
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      # 223 x 223 x 192.
      end_point = 'MaxPool_5a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      # 111 x 111 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1,
                        padding='VALID'):
      # Mixed_5b: 107 x 107 x 256.
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 2),  # branch_0: 111 x 111 x 64
                branch_1,  # branch_1: 107 x 107 x 64
                branch_2,  # branch_2: 107 x 107 x 96
                _trim_border_px(branch_3, 1)  # branch_3: 109 x 109 x 32
            ],
            3)
      end_points[end_point] = net

      # Mixed_5c: 103 x 103 x 288.
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 2),  # branch_0: 107 x 107 x 64
                branch_1,  # branch_1: 103 x 103 x 64
                branch_2,  # branch_2: 103 x 103 x 96
                _trim_border_px(branch_3, 1)  # branch_3: 105 x 105 x 64
            ],
            3)
      end_points[end_point] = net

      # Mixed_5d: 99 x 99 x 288.
      end_point = 'Mixed_5d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(
              branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 2),  # branch_0: 103 x 103 x 64
                branch_1,  # branch_1: 99 x 99 x 64
                branch_2,  # branch_2: 99 x 99 x 96
                _trim_border_px(branch_3, 1)  # branch_2: 101 x 101 x 64
            ],
            3)

      end_points[end_point] = net

      # Mixed_6a: 49 x 49 x 768.
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net,
              depth(384), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1,
              depth(96), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = tf.concat(
            [
                branch_0,  # branch_0: 49 x 49 x 384
                branch_1,  # branch_1: 49 x 49 x 96
                branch_2,  # branch_2: 49 x 49 x 288
            ],
            3)
      end_points[end_point] = net

      # Mixed_6b: 37 x 37 x 768.
      end_point = 'Mixed_6b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(
              branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 6),  # branch_0: 49 x 49 x 192
                _trim_border_px(branch_1, 3),  # branch_1: 43 x 43 x 192
                branch_2,  # branch_2: 37 x 37 x 192
                _trim_border_px(branch_3, 5)  # branch_3: 47 x 47 x 192
            ],
            3)
      end_points[end_point] = net

      # Mixed_6c: 25 x 25 x 768.
      end_point = 'Mixed_6c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 6),  # branch_0: 37 x 37 x 192
                _trim_border_px(branch_1, 3),  # branch_1: 31 x 31 x 192
                branch_2,  # branch_2: 25 x 25 x 192
                _trim_border_px(branch_3, 5)  # branch_3: 35 x 35 x 192
            ],
            3)
      end_points[end_point] = net

      # mixed_6: 13 x 13 x 768.
      end_point = 'Mixed_6d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 6),  # branch_0: 25 x 25 x 192
                _trim_border_px(branch_1, 3),  # branch_1: 19 x 19 x 192
                branch_2,  # branch_2: 13 x 13 x 192
                _trim_border_px(branch_3, 5)  # branch_3: 23 x 23 x 192
            ],
            3)
      end_points[end_point] = net

      # Mixed_6e: 1 x 1 x 768.
      end_point = 'Mixed_6e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(
              branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            [
                _trim_border_px(branch_0, 6),  # branch_0: 13 x 13 x 192
                _trim_border_px(branch_1, 3),  # branch_1: 7 x 7 x 192
                branch_2,  # branch_2: 1 x 1 x 192
                _trim_border_px(branch_3, 5)  # branch_3: 11 x 11 x 192
            ],
            3)
      end_points[end_point] = net

      for i in range(num_final_1x1_conv):
        slim.conv2d(
            net, depth(256), [1, 1], scope='Final_Conv2d_{}_1x1'.format(i))
        end_points['Final_Conv2d_{}_1x1'.format(i)] = net
      return net, end_points


def nopad_inception_v3_fcn(inputs,
                           num_classes=1000,
                           dropout_keep_prob=0.8,
                           is_training=True,
                           min_depth=16,
                           depth_multiplier=1.0,
                           prediction_fn=slim.softmax,
                           reuse=None,
                           inception_fcn_stride=1,
                           scope='NoPadInceptionV3'):
  """No pad inception FCN model.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training.
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer (before dropout) are
      returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels) for
      all convolution ops. The value must be greater than zero. Typical usage
      will be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    inception_fcn_stride: The stride that's used to match the input stride with
      output label resolution.
    scope: Optional variable_scope.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  with tf.variable_scope(
      scope, 'NopadInceptionV3', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = nopad_inception_v3_base(
          inputs,
          scope=scope,
          min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        logits = slim.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            stride=inception_fcn_stride,
            scope='Conv2d_1c_1x1')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
