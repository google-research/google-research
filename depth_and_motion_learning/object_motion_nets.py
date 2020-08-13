# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Networks for motion estimation."""

import tensorflow.compat.v1 as tf

from depth_and_motion_learning import maybe_summary
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

layers = contrib_layers
arg_scope = contrib_framework.arg_scope


def motion_vector_net(images, weight_reg, predict_intrinsics=True):
  """Predict object-motion vectors from a stack of frames or embeddings.

  Args:
    images: Input tensor with shape [B, h, w, 2c], containing two
      depth-concatenated images.
    weight_reg: A float scalar, the amount of weight regularization.
    predict_intrinsics: A boolean, if True the network will predict the
      intrinsic matrix as well.

  Returns:
    A tuple of 3 tf.Tensors, (rotation, translation, intrinsic_mat), of shapes
    [B, 3], [B, 3] and [B, 3, 3] respectively, representing translation vectors,
    rotation angles, and predicted intrinsics matrix respectively. If
    predict_intrinsics is false, the latter is not returned.
  """
  with tf.variable_scope('MotionVectorNet'):
    with arg_scope([layers.conv2d],
                   weights_regularizer=layers.l2_regularizer(weight_reg),
                   activation_fn=tf.nn.relu,
                   stride=2):
      conv1 = layers.conv2d(images, 16, [7, 7], scope='Conv1')
      conv2 = layers.conv2d(conv1, 32, [5, 5], scope='Conv2')
      conv3 = layers.conv2d(conv2, 64, [3, 3], scope='Conv3')
      conv4 = layers.conv2d(conv3, 128, [3, 3], scope='Conv4')
      conv5 = layers.conv2d(conv4, 256, [3, 3], scope='Conv5')
      conv6 = layers.conv2d(conv5, 256, [3, 3], scope='Conv6')
      conv7 = layers.conv2d(conv6, 256, [3, 3], scope='Conv7')

    bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)

    with arg_scope([layers.conv2d],
                   biases_initializer=None,
                   activation_fn=None,
                   stride=1):
      rotation = layers.conv2d(bottleneck, 3, [1, 1], scope='Rotation')
      translation = layers.conv2d(bottleneck, 3, [1, 1], scope='Translation')
    rotation = tf.squeeze(rotation, axis=(1, 2))
    translation = tf.squeeze(translation, axis=(1, 2))
    image_height, image_width = tf.unstack(tf.shape(images)[1:3])
    rot_scale, trans_scale = create_scales(0.001)
    if predict_intrinsics:
      intrinsic_mat = add_intrinsics_head(bottleneck, image_height, image_width)
      return rotation * rot_scale, translation * trans_scale, intrinsic_mat
    # returning different number of items to unpack might cause issues.
    return rotation * rot_scale, translation * trans_scale, None


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

    maybe_summary.scalar('foci', tf.reduce_mean(foci))
    maybe_summary.scalar('offsets', tf.reduce_mean(offsets))

    intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
    batch_size = tf.shape(bottleneck)[0]
    last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
    intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
    return intrinsic_mat


def motion_field_net(images,
                     weight_reg=0.0,
                     align_corners=True,
                     auto_mask=False):
  """Predict object-motion vectors from a stack of frames or embeddings.

  Args:
    images: Input tensor with shape [B, h, w, 2c], containing two
      depth-concatenated images.
    weight_reg: A float scalar, the amount of weight regularization.
    align_corners: align_corners in resize_bilinear. Only used in version 2.
    auto_mask: True to automatically masking out the residual translations
      by thresholding on their mean values.

  Returns:
    A tuple of 3 tf.Tensors:
    rotation: [B, 3], global rotation angles.
    background_translation: [B, 1, 1, 3], global translation vectors.
    residual_translation: [B, h, w, 3], residual translation vector field. The
      overall translation field is background_translation+residual_translation.
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
      background_translation = background_motion[:, :, :, 3:]

      residual_translation = layers.conv2d(
          background_motion,
          3, [1, 1],
          stride=1,
          activation_fn=None,
          scope='unrefined_residual_translation')
      residual_translation = _refine_motion_field(
          residual_translation, conv7, align_corners, scope='Refine7')
      residual_translation = _refine_motion_field(
          residual_translation, conv6, align_corners, scope='Refine6')
      residual_translation = _refine_motion_field(
          residual_translation, conv5, align_corners, scope='Refine5')
      residual_translation = _refine_motion_field(
          residual_translation, conv4, align_corners, scope='Refine4')
      residual_translation = _refine_motion_field(
          residual_translation, conv3, align_corners, scope='Refine3')
      residual_translation = _refine_motion_field(
          residual_translation, conv2, align_corners, scope='Refine2')
      residual_translation = _refine_motion_field(
          residual_translation, conv1, align_corners, scope='Refine1')
      residual_translation = _refine_motion_field(
          residual_translation, images, align_corners, scope='RefineImages')

      rot_scale, trans_scale = create_scales(0.001)
      background_translation *= trans_scale
      residual_translation *= trans_scale
      rotation *= rot_scale

      if auto_mask:
        sq_residual_translation = tf.sqrt(
            tf.reduce_sum(residual_translation**2, axis=3, keepdims=True))
        mean_sq_residual_translation = tf.reduce_mean(
            sq_residual_translation, axis=[0, 1, 2])
        # A mask of shape [B, h, w, 1]
        mask_residual_translation = tf.cast(
            sq_residual_translation > mean_sq_residual_translation,
            residual_translation.dtype.base_dtype)
        residual_translation *= mask_residual_translation

      image_height, image_width = tf.unstack(tf.shape(images)[1:3])
      intrinsic_mat = add_intrinsics_head(bottleneck, image_height, image_width)

      return (rotation, background_translation, residual_translation,
              intrinsic_mat)


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
    maybe_summary.scalar('rotation', rot_scale)
    maybe_summary.scalar('translation', trans_scale)

  return rot_scale, trans_scale


def _refine_motion_field(motion_field, layer, align_corners, scope=None):
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
    align_corners: align_corners in resize_bilinear.
    scope: the variable scope.

  Returns:
    A tf.Tensor of shape [B, h2, w2, m], obtained by upscaling motion_field to
    h2, w2, and mixing it with layer using a few convolutions.

  """
  with tf.variable_scope(scope):
    _, h, w, _ = tf.unstack(tf.shape(layer))
    # Only align_corners=True is supported on TPU
    upsampled_motion_field = tf.image.resize_bilinear(
        motion_field, [h, w], align_corners=align_corners)
    conv_input = tf.concat([upsampled_motion_field, layer], axis=3)
    # pyformat: disable
    conv_output = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_input = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output2 = layers.conv2d(
        conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    # pyformat: enable
    conv_output = tf.concat([conv_output, conv_output2], axis=-1)

    return upsampled_motion_field + layers.conv2d(
        conv_output,
        motion_field.shape.as_list()[-1], [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None)
