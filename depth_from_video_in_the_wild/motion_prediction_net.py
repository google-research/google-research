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

"""A network for predicting egomotion, a 3D translation field and intrinsics."""

import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope


def add_intrinsics_head(bottleneck, image_height, image_width):
  """Adds a head the preficts camera intrinsics.

  Args:
    bottleneck: A tf.Tensor of shape [B, 1, 1, C], typically the bottlenech
      features of a netrowk.
    image_height: A scalar tf.Tensor or an python scalar, the image height in
      pixels.
    image_width: A scalar tf.Tensor or an python scalar, the image width in
      pixels.

  image_height and image_width are used to provide the right scale for the focal
  length and the offest parameters.

  Returns:
    a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
    intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
  """
  with tf.variable_scope('CameraIntrinsics'):
    # Since the focal lengths in pixels tend to be in the order of magnitude of
    # the image width and height, we multiply the network prediction by them.
    focal_lengths = tf.squeeze(
        layers.conv2d(
            bottleneck,
            2, [1, 1],
            stride=1,
            activation_fn=tf.nn.softplus,
            weights_regularizer=None,
            scope='foci'),
        axis=(1, 2)) * tf.to_float(
            tf.convert_to_tensor([[image_width, image_height]]))

    # The pixel offsets tend to be around the center of the image, and they
    # are typically a fraction the image width and height in pixels. We thus
    # multiply the network prediction by the width and height, and the
    # additional 0.5 them by default at the center of the image.
    offsets = (tf.squeeze(
        layers.conv2d(
            bottleneck,
            2, [1, 1],
            stride=1,
            activation_fn=None,
            weights_regularizer=None,
            biases_initializer=None,
            scope='offsets'),
        axis=(1, 2)) + 0.5) * tf.to_float(
            tf.convert_to_tensor([[image_width, image_height]]))

    foci = tf.linalg.diag(focal_lengths)

    intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
    batch_size = tf.shape(bottleneck)[0]
    last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
    intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
    return intrinsic_mat


def motion_field_net(images, weight_reg=0.0):
  """Predict object-motion vectors from a stack of frames or embeddings.

  Args:
    images: Input tensor with shape [B, h, w, 2c], containing two
      depth-concatenated images.
    weight_reg: A float scalar, the amount of weight regularization.

  Returns:
    A tuple of 3 tf.Tensors:
    rotation: [B, 3], global rotation angles (due to camera rotation).
    translation: [B, 1, 1, 3], global translation vectors (due to camera
      translation).
    residual_translation: [B, h, w, 3], residual translation vector field, due
      to motion of objects relatively to the scene. The overall translation
      field is translation + residual_translation.
  """

  with tf.variable_scope('MotionFieldNet'):
    conv1 = layers.conv2d(
        images,
        16, [3, 3],
        stride=2,
        scope='Conv1',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv2 = layers.conv2d(
        conv1,
        32, [3, 3],
        stride=2,
        scope='Conv2',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv3 = layers.conv2d(
        conv2,
        64, [3, 3],
        stride=2,
        scope='Conv3',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv4 = layers.conv2d(
        conv3,
        128, [3, 3],
        stride=2,
        scope='Conv4',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv5 = layers.conv2d(
        conv4,
        256, [3, 3],
        stride=2,
        scope='Conv5',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv6 = layers.conv2d(
        conv5,
        512, [3, 3],
        stride=2,
        scope='Conv6',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)
    conv7 = layers.conv2d(
        conv6,
        1024, [3, 3],
        stride=2,
        scope='Conv7',
        weights_regularizer=layers.l2_regularizer(weight_reg),
        activation_fn=tf.nn.relu)

    bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)

    background_motion = layers.conv2d(
        bottleneck,
        6, [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None,
        scope='background_motion')

    rotation = background_motion[:, 0, 0, :3]
    translation = background_motion[:, :, :, 3:]
    residual_translation = _refine_motion_field(translation, conv7)
    residual_translation = _refine_motion_field(residual_translation, conv6)
    residual_translation = _refine_motion_field(residual_translation, conv5)
    residual_translation = _refine_motion_field(residual_translation, conv4)
    residual_translation = _refine_motion_field(residual_translation, conv3)
    residual_translation = _refine_motion_field(residual_translation, conv2)
    residual_translation = _refine_motion_field(residual_translation, conv1)
    residual_translation = _refine_motion_field(residual_translation, images)

    rot_scale, trans_scale = create_scales(0.001)
    translation *= trans_scale
    residual_translation *= trans_scale
    rotation *= rot_scale

    image_height, image_width = tf.unstack(tf.shape(images)[1:3])
    intrinsic_mat = add_intrinsics_head(bottleneck, image_height, image_width)

    return (rotation, translation, residual_translation, intrinsic_mat)


def create_scales(constraint_minimum):
  """Creates variables representing rotation and translation scaling factors.

  Args:
    constraint_minimum: A scalar, the variables will be constrained to not fall
      below it.

  Returns:
    Two scalar variables, rotation and translation scale.
  """

  def constraint(x):
    return tf.nn.relu(x - constraint_minimum) + constraint_minimum

  with tf.variable_scope('Scales', initializer=0.01, constraint=constraint):
    rot_scale = tf.get_variable('rotation')
    trans_scale = tf.get_variable('translation')

  return rot_scale, trans_scale


def _refine_motion_field(motion_field, layer):
  """Refines a motion field using features from another layer.

  This function builds an element of a UNet-like architecture. `motion_field`
  has a lower spatial resolution than `layer`. First motion_field is resized to
  `layer`'s spatial resolution using bilinear interpolation, then convolutional
  filters are applied on `layer` and the result is added to the upscaled
  `motion_field`.

  This scheme is inspired by FlowNet (https://arxiv.org/abs/1504.06852), and the
  realization that keeping the bottenecks at the same (low) dimension as the
  motion field will pressure the network to gradually transfer details from
  depth channels to space.

  The specifics are slightly different form FlowNet: We use two parallel towers,
  a 3x3 convolution, and two successive 3x3 convolutions, as opposed to one
  3x3 convolution in FLowNet. Also, we add the result to the upscaled
  `motion_field`, forming a residual connection, unlike FlowNet. These changes
  seemed to improve the depth prediction metrics, but exploration was far from
  exhaustive.

  Args:
    motion_field: a tf.Tensor of shape [B, h1, w1, m]. m is the number of
      dimensions in the motion field, for example, 3 in case of a 3D translation
      field.
    layer: tf.Tensor of shape [B, h2, w2, c].

  Returns:
    A tf.Tensor of shape [B, h2, w2, m], obtained by upscaling motion_field to
    h2, w2, and mixing it with layer using a few convolutions.

  """
  _, h, w, _ = tf.unstack(tf.shape(layer))
  upsampled_motion_field = tf.image.resize_bilinear(motion_field, [h, w])
  conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
  conv_output = layers.conv2d(
      conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
  conv_input = layers.conv2d(
      conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
  conv_output2 = layers.conv2d(
      conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
  conv_output = tf.concat([conv_output, conv_output2], axis=-1)

  return upsampled_motion_field + layers.conv2d(
      conv_output,
      motion_field.shape.as_list()[-1], [1, 1],
      stride=1,
      activation_fn=None,
      biases_initializer=None,
      scope=layer.op.name + '/MotionBottleneck')


# In the networks above there is a bug, where some of the MotionBottleneck
# variables are replicated instead of being reused. This issue was raised here:
# https://github.com/google-research/google-research/issues/230#issue-583223537
# The code below fixes the issue. The effect on model performance is generally
# positive but minor. This code is released per users' request. However all the
# published checkpoints are only compatible with the old version of the code
# (above this comment).


def motion_field_net_v2(images, weight_reg=0.0):
  """Predict object-motion vectors from a stack of frames or embeddings.

  Args:
    images: Input tensor with shape [B, h, w, 2c], containing two
      depth-concatenated images.
    weight_reg: A float scalar, the amount of weight regularization.

  Returns:
    A tuple of 3 tf.Tensors:
    rotation: [B, 3], global rotation angles (due to camera rotation).
    translation: [B, 1, 1, 3], global translation vectors (due to camera
      translation).
    residual_translation: [B, h, w, 3], residual translation vector field, due
      to motion of objects relatively to the scene. The overall translation
      field is translation + residual_translation.
  """

  with tf.variable_scope('MotionFieldNet'):
    with arg_scope([layers.conv2d],
                   weights_regularizer=layers.l2_regularizer(weight_reg),
                   activation_fn=tf.nn.relu):

      conv1 = layers.conv2d(images, 16, [3, 3], stride=2, scope='Conv1')
      conv2 = layers.conv2d(conv1, 32, [3, 3], stride=2, scope='Conv2')
      conv3 = layers.conv2d(conv2, 64, [3, 3], stride=2, scope='Conv3')
      conv4 = layers.conv2d(conv3, 128, [3, 3], stride=2, scope='Conv4')
      conv5 = layers.conv2d(conv4, 256, [3, 3], stride=2, scope='Conv5')
      conv6 = layers.conv2d(conv5, 512, [3, 3], stride=2, scope='Conv6')
      conv7 = layers.conv2d(conv6, 1024, [3, 3], stride=2, scope='Conv7')

      bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)

      background_motion = layers.conv2d(
          bottleneck,
          6, [1, 1],
          stride=1,
          activation_fn=None,
          biases_initializer=None,
          scope='background_motion')

      rotation = background_motion[:, 0, 0, :3]
      translation = background_motion[:, :, :, 3:]

      residual_translation = layers.conv2d(
          background_motion,
          3, [1, 1],
          stride=1,
          activation_fn=None,
          scope='unrefined_residual_translation')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv7, scope='Refine7')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv6, scope='Refine6')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv5, scope='Refine5')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv4, scope='Refine4')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv3, scope='Refine3')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv2, scope='Refine2')
      residual_translation = _refine_motion_field_v2(
          residual_translation, conv1, scope='Refine1')
      residual_translation = _refine_motion_field_v2(
          residual_translation, images, scope='RefineImages')

    rot_scale, trans_scale = create_scales(0.001)
    translation *= trans_scale
    residual_translation *= trans_scale
    rotation *= rot_scale

    image_height, image_width = tf.unstack(tf.shape(images)[1:3])
    intrinsic_mat = add_intrinsics_head(bottleneck, image_height, image_width)

    return (rotation, translation, residual_translation, intrinsic_mat)


def _refine_motion_field_v2(motion_field, layer, scope=None):
  """Refines a motion field using features from another layer.

  Same as _refine_motion_field above, but all variables are now properly reused
  (see comment above).

  Args:
    motion_field: a tf.Tensor of shape [B, h1, w1, m]. m is the number of
      dimensions in the motion field, for example, 3 in case of a 3D translation
      field.
    layer: tf.Tensor of shape [B, h2, w2, c].
    scope: the variable scope.

  Returns:
    A tf.Tensor of shape [B, h2, w2, m], obtained by upscaling motion_field to
    h2, w2, and mixing it with layer using a few convolutions.

  """
  with tf.variable_scope(scope):
    _, h, w, _ = tf.unstack(tf.shape(layer))
    upsampled_motion_field = tf.image.resize_bilinear(motion_field, [h, w])
    conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
    conv_output = layers.conv2d(
        conv_input, max(4,
                        layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_input = layers.conv2d(
        conv_input, max(4,
                        layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output2 = layers.conv2d(
        conv_input, max(4,
                        layer.shape.as_list()[-1]), [3, 3], stride=1)
    # pyformat: enable
    conv_output = tf.concat([conv_output, conv_output2], axis=-1)

    return upsampled_motion_field + layers.conv2d(
        conv_output,
        motion_field.shape.as_list()[-1], [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None)
