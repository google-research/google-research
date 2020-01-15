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

"""Library for EM capsule layers.

This has the layer discription for coincidence detection, routing and
capsule layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow.compat.v1 as tf
from capsule_em import layers
from capsule_em import utils


def update_conv_routing(wx, input_activation, activation_biases, sigma_biases,
                        logit_shape, num_out_atoms, input_dim, num_routing,
                        output_dim, final_beta, min_var):
  """Convolutional Routing with EM for Mixture of Gaussians."""
  # Wx: [batch, indim, outdim, outatom, height, width, k, k]
  # logit_shape: [indim, outdim, 1, height, width, k, k]
  # input_activations: [batch, indim, 1, 1, height, width, k, k]
  # activation_biases: [1, 1, outdim, 1, height, width]
  # prior = utils.bias_variable([1] + logit_shape, name='prior')
  post = tf.nn.softmax(
      tf.fill(
          tf.stack([
              tf.shape(input_activation)[0], logit_shape[0], logit_shape[1],
              logit_shape[2], logit_shape[3], logit_shape[4], logit_shape[5],
              logit_shape[6]
          ]), 0.0),
      dim=2)
  out_activation = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], 1, output_dim, 1, logit_shape[3],
          logit_shape[4], 1, 1
      ]), 0.0)
  out_center = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], 1, output_dim, num_out_atoms,
          logit_shape[3], logit_shape[4], 1, 1
      ]), 0.0)
  out_mass = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], 1, output_dim, 1, logit_shape[3],
          logit_shape[4], 1, 1
      ]), 0.0)
  n = logit_shape[3]
  k = logit_shape[5]

  def _body(i, posterior, activation, center, masses):
    """Body of the EM while loop."""
    del activation
    beta = final_beta * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
    # beta = final_beta
    # route: [outdim, height?, width?, batch, indim]
    vote_conf = posterior * input_activation
    # masses: [batch, 1, outdim, 1, height, width, 1, 1]
    masses = tf.reduce_sum(
        tf.reduce_sum(
            tf.reduce_sum(vote_conf, axis=1, keep_dims=True),
            axis=-1,
            keep_dims=True),
        axis=-2,
        keep_dims=True) + 0.0000001
    preactivate_unrolled = vote_conf * wx
    # center: [batch, 1, outdim, outatom, height, width]
    center = .9 * tf.reduce_sum(
        tf.reduce_sum(
            tf.reduce_sum(preactivate_unrolled, axis=1, keep_dims=True),
            axis=-1,
            keep_dims=True),
        axis=-2,
        keep_dims=True) / masses + .1 * center

    noise = (wx - center) * (wx - center)
    variance = min_var + tf.reduce_sum(
        tf.reduce_sum(
            tf.reduce_sum(vote_conf * noise, axis=1, keep_dims=True),
            axis=-1,
            keep_dims=True),
        axis=-2,
        keep_dims=True) / masses
    log_variance = tf.log(variance)
    p_i = -1 * tf.reduce_sum(log_variance, axis=3, keep_dims=True)
    log_2pi = tf.log(2 * math.pi)
    win = masses * (p_i - sigma_biases * num_out_atoms * (log_2pi + 1.0))
    logit = beta * (win - activation_biases * 5000)
    activation_update = tf.minimum(0.0,
                                   logit) - tf.log(1 + tf.exp(-tf.abs(logit)))
    # return activation, center
    log_det_sigma = -1 * p_i
    sigma_update = (num_out_atoms * log_2pi + log_det_sigma) / 2.0
    exp_update = tf.reduce_sum(noise / (2 * variance), axis=3, keep_dims=True)
    prior_update = activation_update - sigma_update - exp_update
    max_prior_update = tf.reduce_max(
        tf.reduce_max(
            tf.reduce_max(
                tf.reduce_max(prior_update, axis=-1, keep_dims=True),
                axis=-2,
                keep_dims=True),
            axis=-3,
            keep_dims=True),
        axis=-4,
        keep_dims=True)
    prior_normal = tf.add(prior_update, -1 * max_prior_update)
    prior_exp = tf.exp(prior_normal)
    t_prior = tf.transpose(prior_exp, [0, 1, 2, 3, 4, 6, 5, 7])
    c_prior = tf.reshape(t_prior, [-1, n * k, n * k, 1])
    pad_prior = tf.pad(
        c_prior,
        [[0, 0], [(k - 1) * (k - 1),
                  (k - 1) * (k - 1)], [(k - 1) * (k - 1),
                                       (k - 1) * (k - 1)], [0, 0]], 'CONSTANT')
    patch_prior = tf.extract_image_patches(
        images=pad_prior,
        ksizes=[1, k, k, 1],
        strides=[1, k, k, 1],
        rates=[1, k - 1, k - 1, 1],
        padding='VALID')
    sum_prior = tf.reduce_sum(patch_prior, axis=-1, keep_dims=True)
    sum_prior_patch = tf.extract_image_patches(
        images=sum_prior,
        ksizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    sum_prior_reshape = tf.reshape(
        sum_prior_patch, [-1, input_dim, output_dim, 1, n, n, k, k]) + 0.0000001
    posterior = prior_exp / sum_prior_reshape
    return (posterior, logit, center, masses)

  # activations = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # centers = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # updates = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # updates.write(0, prior_update)
  for i in range(num_routing):
    post, out_activation, out_center, out_mass = _body(i, post, out_activation,
                                                       out_center, out_mass)
  # for j in range(num_routing):
  #   _, prior_update, out_activation, out_center = _body(
  #       i, prior_update, start_activation, start_center)
  with tf.name_scope('out_activation'):
    utils.activation_summary(tf.sigmoid(out_activation))
  with tf.name_scope('masses'):
    utils.activation_summary(tf.sigmoid(out_mass))
  with tf.name_scope('posterior'):
    utils.activation_summary(post)
  with tf.name_scope('noise'):
    utils.variable_summaries((wx - out_center) * (wx - out_center))
  with tf.name_scope('Wx'):
    utils.variable_summaries(wx)

  # for i in range(num_routing):
  #   utils.activation_summary(activations.read(i))
  # return activations.read(num_routing - 1), centers.read(num_routing - 1)
  return out_activation, out_center


def update_em_routing(wx, input_activation, activation_biases, sigma_biases,
                      logit_shape, num_out_atoms, num_routing, output_dim,
                      leaky, final_beta, min_var):
  """Fully connected routing with EM for Mixture of Gaussians."""
  # Wx: [batch, indim, outdim, outatom, height, width]
  # logit_shape: [indim, outdim, 1, height, width]
  # input_activations: [batch, indim, 1, 1, 1, 1]
  # activation_biases: [1, 1, outdim, 1, height, width]
  # prior = utils.bias_variable([1] + logit_shape, name='prior')
  update = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], logit_shape[0], logit_shape[1],
          logit_shape[2], logit_shape[3], logit_shape[4]
      ]), 0.0)
  out_activation = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], 1, output_dim, 1, logit_shape[3],
          logit_shape[4]
      ]), 0.0)
  out_center = tf.fill(
      tf.stack([
          tf.shape(input_activation)[0], 1, output_dim, num_out_atoms,
          logit_shape[3], logit_shape[4]
      ]), 0.0)

  def _body(i, update, activation, center):
    """Body of the EM while loop."""
    del activation
    beta = final_beta * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
    # beta = final_beta
    # route: [outdim, height?, width?, batch, indim]
    if leaky:
      posterior = layers.leaky_routing(update, output_dim)
    else:
      posterior = tf.nn.softmax(update, dim=2)
    vote_conf = posterior * input_activation
    # masses: [batch, 1, outdim, 1, height, width]
    masses = tf.reduce_sum(vote_conf, axis=1, keep_dims=True) + 0.00001
    preactivate_unrolled = vote_conf * wx
    # center: [batch, 1, outdim, outatom, height, width]
    center = .9 * tf.reduce_sum(
        preactivate_unrolled, axis=1, keep_dims=True) / masses + .1 * center

    noise = (wx - center) * (wx - center)
    variance = min_var + tf.reduce_sum(
        vote_conf * noise, axis=1, keep_dims=True) / masses
    log_variance = tf.log(variance)
    p_i = -1 * tf.reduce_sum(log_variance, axis=3, keep_dims=True)
    log_2pi = tf.log(2 * math.pi)
    win = masses * (p_i - sigma_biases * num_out_atoms * (log_2pi + 1.0))
    logit = beta * (win - activation_biases * 5000)
    activation_update = tf.minimum(0.0,
                                   logit) - tf.log(1 + tf.exp(-tf.abs(logit)))
    # return activation, center
    log_det_sigma = tf.reduce_sum(log_variance, axis=3, keep_dims=True)
    sigma_update = (num_out_atoms * log_2pi + log_det_sigma) / 2.0
    exp_update = tf.reduce_sum(noise / (2 * variance), axis=3, keep_dims=True)
    prior_update = activation_update - sigma_update - exp_update
    return (prior_update, logit, center)

  # activations = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # centers = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # updates = tf.TensorArray(
  #     dtype=tf.float32, size=num_routing, clear_after_read=False)
  # updates.write(0, prior_update)
  for i in range(num_routing):
    update, out_activation, out_center = _body(i, update, out_activation,
                                               out_center)
  # for j in range(num_routing):
  #   _, prior_update, out_activation, out_center = _body(
  #       i, prior_update, start_activation, start_center)
  with tf.name_scope('out_activation'):
    utils.activation_summary(tf.sigmoid(out_activation))
  with tf.name_scope('noise'):
    utils.variable_summaries((wx - out_center) * (wx - out_center))
  with tf.name_scope('Wx'):
    utils.variable_summaries(wx)

  # for i in range(num_routing):
  #   utils.activation_summary(activations.read(i))
  # return activations.read(num_routing - 1), centers.read(num_routing - 1)
  return out_activation, out_center


def connector_capsule_mat(input_tensor,
                          position_grid,
                          input_activation,
                          input_dim,
                          output_dim,
                          layer_name,
                          num_routing=3,
                          num_in_atoms=3,
                          num_out_atoms=3,
                          leaky=False,
                          final_beta=1.0,
                          min_var=0.0005):
  """Final Capsule Layer with Pose Matrices and Shared connections."""
  # One weight tensor for each capsule of the layer bellow: w: [8*128, 8*10]
  with tf.variable_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('input_center_connector'):
      utils.activation_summary(input_tensor)
    weights = utils.weight_variable(
        [input_dim, num_out_atoms, output_dim * num_out_atoms], stddev=0.01)
    # weights = tf.clip_by_norm(weights, 1.0, axes=[1])
    activation_biases = utils.bias_variable([1, 1, output_dim, 1, 1, 1],
                                            init_value=1.0,
                                            name='activation_biases')
    sigma_biases = utils.bias_variable([1, 1, output_dim, 1, 1, 1],
                                       init_value=2.0,
                                       name='sigma_biases')

    with tf.name_scope('Wx_plus_b'):
      # input_tensor: [x, 128, 8, h, w]
      input_shape = tf.shape(input_tensor)
      input_trans = tf.transpose(input_tensor, [1, 0, 3, 4, 2])
      input_share = tf.reshape(input_trans, [input_dim, -1, num_in_atoms])
      # input_expanded: [x, 128, 8, 1]
      wx_share = tf.matmul(input_share, weights)
      # sqr_num_out_atoms = num_out_atoms
      num_out_atoms *= num_out_atoms
      wx_trans = tf.reshape(wx_share, [
          input_dim, input_shape[0], input_shape[3], input_shape[4],
          num_out_atoms, output_dim
      ])
      wx_trans.set_shape((input_dim, None, input_tensor.get_shape()[3].value,
                          input_tensor.get_shape()[4].value, num_out_atoms,
                          output_dim))
      h, w, _ = position_grid.get_shape()
      height = h.value
      width = w.value
      # t_pose = tf.transpose(position_grid, [2, 0, 1])
      # t_pose_exp = tf.scatter_nd([[sqr_num_out_atoms -1],
      #   [2 * sqr_num_out_atoms - 1]], t_pose, [num_out_atoms, height, width])
      # pose_g_exp = tf.transpose(t_pose_exp, [1, 2, 0])
      zero_grid = tf.zeros([height, width, num_out_atoms - 2])
      pose_g_exp = tf.concat([position_grid, zero_grid], axis=2)
      pose_g = tf.expand_dims(
          tf.expand_dims(tf.expand_dims(pose_g_exp, -1), 0), 0)
      wx_posed = wx_trans + pose_g
      wx_posed_t = tf.transpose(wx_posed, [1, 0, 2, 3, 5, 4])

      # Wx_reshaped: [x, 128, 10, 8]
      wx = tf.reshape(
          wx_posed_t,
          [-1, input_dim * height * width, output_dim, num_out_atoms, 1, 1])
    with tf.name_scope('routing'):
      # Routing
      # logits: [x, 128, 10]
      logit_shape = [input_dim * height * width, output_dim, 1, 1, 1]
      for _ in range(4):
        input_activation = tf.expand_dims(input_activation, axis=-1)
      activation, center = update_em_routing(
          wx=wx,
          input_activation=input_activation,
          activation_biases=activation_biases,
          sigma_biases=sigma_biases,
          logit_shape=logit_shape,
          num_out_atoms=num_out_atoms,
          num_routing=num_routing,
          output_dim=output_dim,
          leaky=leaky,
          final_beta=final_beta / 4,
          min_var=min_var,
      )
    out_activation = tf.squeeze(activation, axis=[1, 3, 4, 5])
    out_center = tf.squeeze(center, axis=[1, 4, 5])
    return tf.sigmoid(out_activation), out_center


def conv_capsule_mat(input_tensor,
                     input_activation,
                     input_dim,
                     output_dim,
                     layer_name,
                     num_routing=3,
                     num_in_atoms=3,
                     num_out_atoms=3,
                     stride=2,
                     kernel_size=5,
                     min_var=0.0005,
                     final_beta=1.0):
  """Convolutional Capsule layer with Pose Matrices."""
  print('caps conv stride: {}'.format(stride))
  in_atom_sq = num_in_atoms * num_in_atoms
  with tf.variable_scope(layer_name):
    input_shape = tf.shape(input_tensor)
    _, _, _, in_height, in_width = input_tensor.get_shape()
    # This Variable will hold the state of the weights for the layer
    kernel = utils.weight_variable(
        shape=[
            input_dim, kernel_size, kernel_size, num_in_atoms,
            output_dim * num_out_atoms
        ],
        stddev=0.3)
    # kernel = tf.clip_by_norm(kernel, 3.0, axes=[1, 2, 3])
    activation_biases = utils.bias_variable([1, 1, output_dim, 1, 1, 1, 1, 1],
                                            init_value=0.5,
                                            name='activation_biases')
    sigma_biases = utils.bias_variable([1, 1, output_dim, 1, 1, 1, 1, 1],
                                       init_value=.5,
                                       name='sigma_biases')
    with tf.name_scope('conv'):
      print('convi;')
      # input_tensor: [x,128,8, c1,c2] -> [x*128,8, c1,c2]
      print(input_tensor.get_shape())
      input_tensor_reshaped = tf.reshape(input_tensor, [
          input_shape[0] * input_dim * in_atom_sq, input_shape[3],
          input_shape[4], 1
      ])
      input_tensor_reshaped.set_shape((None, input_tensor.get_shape()[3].value,
                                       input_tensor.get_shape()[4].value, 1))
      input_act_reshaped = tf.reshape(
          input_activation,
          [input_shape[0] * input_dim, input_shape[3], input_shape[4], 1])
      input_act_reshaped.set_shape((None, input_tensor.get_shape()[3].value,
                                    input_tensor.get_shape()[4].value, 1))
      print(input_tensor_reshaped.get_shape())
      # conv: [x*128,out*out_at, c3,c4]
      conv_patches = tf.extract_image_patches(
          images=input_tensor_reshaped,
          ksizes=[1, kernel_size, kernel_size, 1],
          strides=[1, stride, stride, 1],
          rates=[1, 1, 1, 1],
          padding='VALID',
      )
      act_patches = tf.extract_image_patches(
          images=input_act_reshaped,
          ksizes=[1, kernel_size, kernel_size, 1],
          strides=[1, stride, stride, 1],
          rates=[1, 1, 1, 1],
          padding='VALID',
      )
      o_height = (in_height.value - kernel_size) // stride + 1
      o_width = (in_width.value - kernel_size) // stride + 1
      patches = tf.reshape(conv_patches,
                           (input_shape[0], input_dim, in_atom_sq, o_height,
                            o_width, kernel_size, kernel_size))
      patches.set_shape((None, input_dim, in_atom_sq, o_height, o_width,
                         kernel_size, kernel_size))
      patch_trans = tf.transpose(patches, [1, 5, 6, 0, 3, 4, 2])
      patch_split = tf.reshape(
          patch_trans,
          (input_dim, kernel_size, kernel_size,
           input_shape[0] * o_height * o_width * num_in_atoms, num_in_atoms))
      patch_split.set_shape((input_dim, kernel_size, kernel_size, None,
                             num_in_atoms))
      a_patches = tf.reshape(act_patches,
                             (input_shape[0], input_dim, 1, 1, o_height,
                              o_width, kernel_size, kernel_size))
      a_patches.set_shape((None, input_dim, 1, 1, o_height, o_width,
                           kernel_size, kernel_size))
      with tf.name_scope('input_act'):
        utils.activation_summary(
            tf.reduce_sum(
                tf.reduce_sum(tf.reduce_sum(a_patches, axis=1), axis=-1),
                axis=-1))
      with tf.name_scope('Wx'):
        wx = tf.matmul(patch_split, kernel)
        wx = tf.reshape(
            wx, (input_dim, kernel_size, kernel_size, input_shape[0], o_height,
                 o_width, num_in_atoms * num_out_atoms, output_dim))
        wx.set_shape((input_dim, kernel_size, kernel_size, None, o_height,
                      o_width, num_in_atoms * num_out_atoms, output_dim))
        wx = tf.transpose(wx, [3, 0, 7, 6, 4, 5, 1, 2])
        utils.activation_summary(wx)

    with tf.name_scope('routing'):
      # Routing
      # logits: [x, 128, 10, c3, c4]
      logit_shape = [
          input_dim, output_dim, 1, o_height, o_width, kernel_size, kernel_size
      ]
      activation, center = update_conv_routing(
          wx=wx,
          input_activation=a_patches,
          activation_biases=activation_biases,
          sigma_biases=sigma_biases,
          logit_shape=logit_shape,
          num_out_atoms=num_out_atoms * num_out_atoms,
          input_dim=input_dim,
          num_routing=num_routing,
          output_dim=output_dim,
          min_var=min_var,
          final_beta=final_beta,
      )
      # activations: [x, 10, 8, c3, c4]

    out_activation = tf.squeeze(activation, axis=[1, 3, 6, 7])
    out_center = tf.squeeze(center, axis=[1, 6, 7])
    with tf.name_scope('center'):
      utils.activation_summary(out_center)
    return tf.sigmoid(out_activation), out_center


def primary_caps(conv, conv_dim, output_dim, out_atoms):
  """First Capsule layer where activation is calculated via sigmoid+conv."""
  with tf.variable_scope('conv_capsule1'):
    w_kernel = utils.weight_variable(
        shape=[1, 1, conv_dim, (out_atoms) * output_dim], stddev=0.5)
    # w_kernel = tf.clip_by_norm(w_kernel, 1.0, axes=[2])
    with tf.variable_scope('conv_capsule1_act'):
      a_kernel = utils.weight_variable(
          shape=[1, 1, conv_dim, output_dim], stddev=3.0)
    kernel = tf.concat((w_kernel, a_kernel), axis=3)
    conv_caps = tf.nn.conv2d(
        conv, kernel, [1, 1, 1, 1], padding='SAME', data_format='NCHW')
    _, _, c_height, c_width = conv_caps.get_shape()
    conv_shape = tf.shape(conv_caps)
    # conv_reshaped: [x, 128, out, out_at, c3, c4]
    conv_reshaped = tf.reshape(conv_caps, [
        conv_shape[0], output_dim, out_atoms + 1, conv_shape[2], conv_shape[3]
    ])
    conv_reshaped.set_shape((None, output_dim, out_atoms + 1, c_height.value,
                             c_width.value))
    conv_caps_center, conv_caps_logit = tf.split(
        conv_reshaped, [out_atoms, 1], axis=2)
    conv_caps_activation = tf.sigmoid(conv_caps_logit - 1.0)
  return conv_caps_activation, conv_caps_center
