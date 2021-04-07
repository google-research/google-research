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

"""DNN definitions for KeyPose keypoint files.

Adapted from 'Discovery of Latent 3D Keypoints via End-to-end
Geometric Reasoning' keypoint network.
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dilated_cnn(inputs,
                num_filters,
                max_dilation,
                dilation_rep,
                filter_size,
                dropout=0.0,
                bn_decay=0.999,
                bn_epsilon=1.0e-8,
                bn_scale=False,
                is_training=True):
  """Constructs a base dilated convolutional network.

  Args:
    inputs: Typically [batch, h, w, [3,6]] Input RGB images. Two 3-channel
      images are concatenated for stereo.
    num_filters: The number of filters for all layers.
    max_dilation: Size of the last dilation of a series.  Must be a power of
      two.fine
    dilation_rep: Number of times to repeat the last dilation.
    filter_size: kernel size for CNNs.
    dropout: >0 if dropout is to be applied.
    bn_decay: batchnorm parameter.
    bn_epsilon: batchnorm parameter.
    bn_scale: True to scale batchnorm.
    is_training: True if this function is called during training.

  Returns:
    Output of this dilated CNN.
  """
  # Progression of powers of 2: [1, 2, 4, 8 ... max_dliation].
  maxlog = int(math.log(max_dilation, 2))
  seq = [2**x for x in range(maxlog + 1)]
  seq += [1] * (dilation_rep - 1)
  net = inputs
  for i, r in enumerate([1] + seq + seq + [1]):
    # Split off head before the last dilation 1 convolutions.
    if i == (len(seq) * 2 - dilation_rep + 2):
      head = net

    fs = filter_size
    net = layers.Conv2D(
        num_filters,
        fs,
        dilation_rate=r,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        use_bias=False,
        trainable=is_training,
        name='dil_conv_%d_%d' % (r, i))(
            net)
    # From https://arxiv.org/pdf/1905.05928.pdf; batchnorm then dropout
    # slim layers has decay/momentum = 0.999, not 0.99.
    # slim layers has scale=False, not scale=True.
    net = layers.BatchNormalization(
        scale=bn_scale,
        epsilon=bn_epsilon,
        trainable=is_training,
        momentum=bn_decay)(
            net, training=is_training)
    net = layers.LeakyReLU(alpha=0.1)(net)
    if dropout and i == len(seq):
      net = layers.SpatialDropout2D(dropout)(net, training=is_training)

  return net, head


# Image normalization.  These values are taken from ImageNet.
def norm_image(im):
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  std_inv = 1.0 / std
  return tf.multiply(tf.subtract(im, mean), std_inv)


# Integral calculation for centroid coordinates.
def meshgrid(w, h):
  """Returns a meshgrid ranging from [-1,1] in x, y axes."""

  rh = np.arange(0.5, h, 1) / (h / 2) - 1
  rw = np.arange(0.5, w, 1) / (w / 2) - 1
  ranx, rany = tf.meshgrid(rw, rh)
  return tf.cast(ranx, tf.float32), tf.cast(rany, tf.float32)


# Keras layer for getting centroid coord by multiplying with a mesh field.
def k_reduce_mult_sum(tensor, mult, axis=None, name=None):

  def reduce_mult_sum(tensor, mult, axis):
    return tf.reduce_sum(tensor * mult, axis=axis)

  if not axis:
    axis = [-1]
  return layers.Lambda(
      reduce_mult_sum, arguments={
          'mult': mult,
          'axis': axis
      }, name=name)(
          tensor)


# Keras layer for reducing sum.
def k_reduce_sum(tensor, axis=None, name=None):

  def reduce_sum(tensor, axis):
    return tf.reduce_sum(tensor, axis=axis)

  if not axis:
    axis = [-1]
  return layers.Lambda(reduce_sum, arguments={'axis': axis}, name=name)(tensor)


def project(tmat, tvec, tvec_transpose=False):
  """Projects homogeneous 3D XYZ coordinates to image uvd coordinates, or vv.

  Args:
    tmat: has shape [[N,] batch_size, 4, 4]
    tvec: has shape [[N,] batch_size, 4, num_kp] or [batch_size, num_kp, 4].
    tvec_transpose: True if tvec is to be transposed before use.

  Returns:
    Has shape [[N,] batch_size, 4, num_kp]
  """
  tp = tf.matmul(tmat, tvec, transpose_b=tvec_transpose)
  # Using <3:4> instead of <3> preserves shape.
  tp = tp / (tp[Ellipsis, 3:4, :] + 1.0e-10)
  return tp


def project_hom(hom, tvec):
  """Transforms homogeneous 2D uvw coordinates to 2D uv1 coordinates.

  Args:
    hom: has shape [[N,] batch_size, 3, 3].
    tvec: has shape [[N,] batch_size, num_kp, 3].

  Returns:
    tensor with shape [[N,] batch_size, num_kp, 3].
  """
  tp = tf.matmul(hom, tvec, transpose_b=True)
  # Using <2:3> instead of <2> preserves shape.
  tp = tp / (tp[Ellipsis, 2:3, :] + 1.0e-10)
  return tf.transpose(tp, [0, 2, 1])


def to_pixel(vec, res):
  """Converts a normalized image coord to pixel coords."""
  return (vec + 1.0) * res * 0.5  # [-1,1]


def to_norm_vec(vec, mparams, weight_disp=1.0):
  """Converts a pixel-based tensor to normalized image coordinates."""
  # [..., num_kp, 3]
  return tf.concat([
      vec[Ellipsis, 0:1] * 2.0 / mparams.modelx - 1.0,
      vec[Ellipsis, 1:2] * 2.0 / mparams.modely - 1.0,
      vec[Ellipsis, 2:3] / mparams.modelx * weight_disp
  ],
                   axis=-1)


def to_norm(mat, mparams, weight_disp=1.0):
  """Converts a pixel-based tensor to normalized image coordinates."""
  # [..., 4, num_kp]
  return tf.concat([
      mat[Ellipsis, 0:1, :] * 2.0 / mparams.modelx - 1.0,
      mat[Ellipsis, 1:2, :] * 2.0 / mparams.modely - 1.0,
      mat[Ellipsis, 2:3, :] * 2.0 / mparams.modelx * weight_disp, mat[Ellipsis, 3:4, :]
  ],
                   axis=-2)


def convert_uvd_raw(uvd, offsets, hom, mparams):
  """Converts output of uvd integrals on cropped, warped img to original image.

  Args:
    uvd: is [batch, num_kp, 3], with order u,v,d.
    offsets: is [batch, 3].
    hom: is [batch, 3, 3].
    mparams: model parameters.

  Returns:
    Various pixel and disparity results.
  """
  num_kp = mparams.num_kp
  uv_pix_raw = tf.stack([
      to_pixel(uvd[:, :, 0], mparams.modelx),
      to_pixel(uvd[:, :, 1], mparams.modely),
      tf.ones_like(uvd[:, :, 0])
  ],
                        axis=-1)  # [batch, num_kp, 3]
  uv_pix = project_hom(hom, uv_pix_raw)  # [batch, num_kp, 3]
  disp = uvd[:, :, 2] * mparams.modelx  # Convert to pixels.

  # uvdw is in pixel coords of the original image.
  uvdw = tf.stack([
      uv_pix[:, :, 0] + tf.stack([offsets[:, 0]] * num_kp, axis=1),
      uv_pix[:, :, 1] + tf.stack([offsets[:, 1]] * num_kp, axis=1),
      disp + tf.stack([offsets[:, 2]] * num_kp, axis=1),
      tf.ones_like(uv_pix[:, :, 0])
  ],
                  axis=-1,
                  name='uvdw_out')
  print('uvdw shape is [batch, num_kps, 4]:', uvdw.shape)

  # uvdw_pos is in pixel coords of original image,
  # and has always-positive disparities.
  uvdw_pos = tf.stack([
      uvdw[:, :, 0], uvdw[:, :, 1],
      tf.nn.relu(uvdw[:, :, 2]) + 1.0e-5,
      tf.ones_like(uvdw[:, :, 0])
  ],
                      axis=-1,
                      name='uvdw_pos')

  return uv_pix_raw, uv_pix, uvdw, uvdw_pos


def keypose_model(mparams, is_training):
  """Constructs a Keras model that predicts 3D keypoints.

  Model input is left and optionally right image, modelx x modely x 3, float32.
  Values are from 0.0 to 1.0

  Args:
    mparams: ConfigParams object use_stereo - True if right image is input.
      num_filters - The number of filters for all layers. num_kp - The number of
      keypoints. ...
    is_training: True if training the model.

  Returns:
    uv: [batch, num_kp, 2] 2D locations of keypoints.
    d: [batch, num_kp] The inverse depth of keypoints.
    prob_viz: A visualization of all predicted keypoints.
    prob_vizs: A list of visualizations of each keypoint.
    kp_viz: A visualization of GT keypoints.
    img_out: The photometrically and geometrically altered image.
    rot: rotation matrix modifying input: [batch, 3, 3].
  """
  print('Mparams in keypose_model:\n', mparams)
  use_stereo = mparams.use_stereo
  num_filters = mparams.num_filters
  max_dilation = mparams.max_dilation
  dilation_rep = mparams.dilation_rep
  dropout = mparams.dropout
  num_kp = mparams.num_kp
  modelx = mparams.modelx
  modely = mparams.modely
  filter_size = mparams.filter_size
  bn_decay = mparams.batchnorm[0]
  bn_epsilon = mparams.batchnorm[1]
  bn_scale = mparams.batchnorm[2]

  # Aux params input to the model.
  offsets = keras.Input(shape=(3,), name='offsets', dtype='float32')
  hom = keras.Input(shape=(3, 3), name='hom', dtype='float32')
  to_world = keras.Input(shape=(4, 4), name='to_world_L', dtype='float32')

  # Images input to the model.
  img_l = keras.Input(shape=(modely, modelx, 3), name='img_L', dtype='float32')
  img_l_norm = layers.Lambda(norm_image)(img_l)
  img_r = keras.Input(shape=(modely, modelx, 3), name='img_R', dtype='float32')
  if use_stereo:
    img_r_norm = layers.Lambda(norm_image)(img_r)
    img_l_norm = layers.concatenate([img_l_norm, img_r_norm])

  net, _ = dilated_cnn(
      img_l_norm,
      num_filters,
      max_dilation,
      dilation_rep,
      filter_size,
      bn_decay=bn_decay,
      bn_epsilon=bn_epsilon,
      bn_scale=bn_scale,
      dropout=dropout,
      is_training=is_training)

  print('Dilation net shape:', net.shape)

  # Regression to keypoint values.
  if mparams.use_regress:
    if dropout:
      net = layers.SpatialDropout2D(dropout)(net, training=is_training)

    net = layers.Conv2D(
        64,
        1,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        use_bias=False,
        padding='valid')(
            net)
    net = layers.BatchNormalization(
        scale=bn_scale, epsilon=bn_epsilon, momentum=bn_decay)(
            net, training=is_training)
    net = layers.LeakyReLU(alpha=0.1)(net)
    net = layers.Conv2D(
        64,
        1,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        use_bias=False,
        padding='valid')(
            net)
    net = layers.BatchNormalization(
        scale=bn_scale, epsilon=bn_epsilon, momentum=bn_decay)(
            net, training=is_training)
    net = layers.LeakyReLU(alpha=0.1)(net)
    net = layers.Conv2D(
        num_kp * 3,
        1,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        padding='valid')(net)  # [batch, h, w, num_kp * 3]
    net = tf.reduce_mean(net, axis=[-3, -2])  # [batch, num_kp * 3]
    print('Regress reduce mean shape:', net.shape)

    uvd = tf.reshape(net, [-1, num_kp, 3])  # [batch, num_kp, 3]
    print('Regress uvd shape:', uvd.shape)
    prob = tf.stack([tf.zeros_like(img_l[Ellipsis, 0])] * num_kp, axis=1)
    disp_map = prob
    # [batch, num_kp, h, w]

  else:
    # The probability distribution map for keypoints.  No activation.
    prob = layers.Conv2D(
        num_kp,
        filter_size,
        dilation_rate=1,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        padding='same')(
            net)

    # Disparity map.
    disp_map = layers.Conv2D(
        num_kp,
        filter_size,
        dilation_rate=1,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        padding='same')(
            net)

    # [batch_size, h, w, num_kp]
    prob = layers.Permute((3, 1, 2))(prob)
    disp_map = layers.Permute((3, 1, 2))(disp_map)

    # [batch_size, num_kp, h, w]
    prob = layers.Reshape((num_kp, modely * modelx))(prob)
    prob = layers.Softmax()(prob)
    prob = layers.Reshape((num_kp, modely, modelx), name='prob')(prob)
    disp = layers.multiply([prob, disp_map])
    disp = k_reduce_sum(disp, axis=[-1, -2], name='disp_out')

    ranx, rany = meshgrid(modelx, modely)
    # Use centroid to find indices.
    sx = k_reduce_mult_sum(prob, ranx, axis=[-1, -2])
    sy = k_reduce_mult_sum(prob, rany, axis=[-1, -2])
    # uv are in normalized coords [-1, 1], uv order.
    uvd = layers.concatenate([sx, sy, disp])
    uvd = layers.Reshape((3, num_kp))(uvd)
    uvd = layers.Permute((2, 1), name='uvd')(uvd)  # [batch, num_kp, 3]

  uv_pix_raw, uv_pix, uvdw, uvdw_pos = convert_uvd_raw(uvd, offsets, hom,
                                                       mparams)

  xyzw = project(to_world, uvdw_pos, True)  # [batch, 4, num_kp]

  model = keras.Model(
      inputs={
          'img_L': img_l,
          'img_R': img_r,
          'offsets': offsets,
          'hom': hom,
          'to_world_L': to_world
      },
      outputs={
          'uvd': uvd,
          'uvdw': uvdw,
          'uvdw_pos': uvdw_pos,
          'uv_pix': uv_pix,
          'uv_pix_raw': uv_pix_raw,
          'xyzw': xyzw,
          'prob': prob,
          'disp': disp_map
      },
      name='keypose')

  model.summary()
  return model
