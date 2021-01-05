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

# Lint as: python2, python3
# pylint: disable=invalid-name,g-bad-import-order,missing-docstring
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import functools
import os

from absl import app
from absl import flags
from concurrent import futures
import gin
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from typing import Any, Dict, List, Optional, Tuple

from neutra import utils

tfd = tfp.distributions
tfb = tfp.bijectors

FLAGS = flags.FLAGS

TRAIN_BATCH = 250
TEST_BATCH = 1000
AIS_BATCH = 50



def ReduceL2(tensor, dims):
  return tf.sqrt(tf.reduce_sum(tf.square(tensor), dims))


@utils.MakeTFTemplate
def Conv2DWN(inputs,
             num_filters,
             kernel_size=[3, 3],
             stride=[1, 1],
             pad="SAME",
             activation=None,
             weights_initializer=utils.L2HMCInitializer(),
             biases_initializer=tf.zeros_initializer(),
             scope=None):
  if activation is None:
    activation = lambda x: x
  num_inputs = int(inputs.shape[3])

  with tf.variable_scope(scope, "conv_2d_wn"):
    w = tf.get_variable(
        "w", [kernel_size[0], kernel_size[1], num_inputs, num_filters],
        initializer=weights_initializer)
    if biases_initializer is not None:
      b = tf.get_variable("b", [num_filters], initializer=biases_initializer)
    g = tf.get_variable(
        "g", initializer=tf.log(ReduceL2(w.initialized_value(), [0, 1, 2])))

    g = tf.exp(g)
    w = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(w, [0, 1, 2])
    out = tf.nn.conv2d(inputs, w, [1, stride[0], stride[1], 1], pad)

    if biases_initializer is not None:
      out += tf.reshape(b, [1, 1, 1, num_filters])

    return activation(out)


def GetLinearARMask(num_inputs, num_outputs, zero_diagonal=False):
  assert num_inputs % num_outputs == 0 or num_outputs % num_inputs == 0, "%d vs %d" % (num_inputs, num_outputs)

  mask = np.ones([num_inputs, num_outputs], dtype=np.float32)
  if num_outputs >= num_inputs:
    k = num_outputs // num_inputs
    for i in range(num_inputs):
      mask[i + 1:, i * k:(i + 1) * k] = 0
      if zero_diagonal:
        mask[i:i + 1, i * k:(i + 1) * k] = 0
  else:
    k = num_inputs // num_outputs
    for i in range(num_outputs):
      mask[(i + 1) * k:, i:i + 1] = 0
      if zero_diagonal:
        mask[i * k:(i + 1) * k:, i:i + 1] = 0
  return mask


def GetConvARMask(h, w, num_inputs, num_filters, zero_diagonal=False):
  l = (h - 1) // 2
  m = (w - 1) // 2
  mask = np.ones([h, w, num_inputs, num_filters], dtype=np.float32)
  mask[:l, :, :, :] = 0
  mask[l, :m, :, :] = 0
  mask[l, m, :, :] = GetLinearARMask(num_inputs, num_filters, zero_diagonal)
  return mask


@utils.MakeTFTemplate
def Conv2DAR(inputs, num_filters,
                     kernel_size=[3, 3],
                     zero_diagonal=False,
                     weights_initializer=None,
                     biases_initializer=tf.zeros_initializer(),
                     scope=None):
  num_inputs = int(inputs.get_shape()[3])
  mask = GetConvARMask(kernel_size[0], kernel_size[1], num_inputs, num_filters, zero_diagonal)
  w = tf.get_variable("w", [kernel_size[0], kernel_size[1], num_inputs, num_filters], initializer=weights_initializer)
  b = tf.get_variable("b", [num_filters], initializer=biases_initializer)
  g = tf.get_variable(
      "g", initializer=tf.log(ReduceL2(w.initialized_value() * mask, [0, 1, 2])))

  g = tf.exp(g)
  w = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(w * mask, [0, 1, 2])

  out = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME")
  return out + tf.reshape(b, [1, 1, 1, num_filters])


@utils.MakeTFTemplate
def ConvAR(x,
           h=None,
           real_event_shape=[],
           hidden_layers=[],
           **kwargs):
  #input_shape = (
  #    np.int32(x.shape.as_list())
  #    if x.shape.is_fully_defined() else tf.shape(x))
  #x = tf.reshape(x, [-1] + real_event_shape)
  for i, units in enumerate(hidden_layers):
    x = Conv2DAR("conv2d_ar_%d"%i, num_filters=units, zero_diagonal=False, **kwargs)(inputs=x)
    if i == 0 and h is not None:
      if h.shape[-1] != x.shape[-1]:
        x += Conv2DWN("conv2d_h", num_filters=int(x.shape[-1]), kernel_size=[1, 1], stride=[1, 1])(h)
      else:
        x += h
    x = tf.nn.elu(x)

  shift = Conv2DAR(
      "conv2d_shift",
      num_filters=real_event_shape[-1],
      zero_diagonal=True,
      **kwargs)(
          inputs=x)
  log_scale = Conv2DAR(
      "conv2d_scale",
      num_filters=real_event_shape[-1],
      zero_diagonal=True,
      **kwargs)(
          inputs=x)

  #shift = tf.reshape(shift, input_shape)
  #log_scale = tf.reshape(log_scale, input_shape)

  return shift, log_scale


@utils.MakeTFTemplate
def DenseWN(inputs,
            num_outputs,
            activation=None,
            weights_initializer=utils.L2HMCInitializer(),
            biases_initializer=tf.zeros_initializer(),
            scope=None):
  if activation is None:
    activation = lambda x: x
  num_inputs = int(inputs.get_shape()[1])

  with tf.variable_scope(scope, "dense_wn"):
    w = tf.get_variable(
        "w", [num_inputs, num_outputs], initializer=weights_initializer)
    if biases_initializer is not None:
      b = tf.get_variable("b", [num_outputs], initializer=biases_initializer)
    g = tf.get_variable(
        "g", initializer=tf.log(ReduceL2(w.initialized_value(), [0])))

    g = tf.exp(g)
    w = g * tf.nn.l2_normalize(w, [0])
    out = tf.matmul(inputs, w)

    if biases_initializer is not None:
      out += tf.expand_dims(b, 0)

    return activation(out)


@utils.MakeTFTemplate
def ResConv2D(inputs,
              num_filters,
              kernel_size,
              stride,
              activation=tf.nn.elu,
              output_init_factor=1.0):
  x = Conv2DWN(
      "conv2d_in",
      num_filters=num_filters,
      kernel_size=kernel_size,
      stride=stride,
      activation=activation)(
          inputs=inputs)
  non_linear = Conv2DWN(
      "conv2d_nl",
      num_filters=num_filters,
      kernel_size=kernel_size,
      stride=[1, 1],
      weights_initializer=utils.L2HMCInitializer(factor=output_init_factor))(
          inputs=x)
  skip = Conv2DWN(
      "conv2d_skip",
      num_filters=num_filters,
      kernel_size=kernel_size,
      stride=stride,
      weights_initializer=utils.L2HMCInitializer(factor=output_init_factor))(
          inputs=inputs)
  return non_linear + skip


@utils.MakeTFTemplate
def ResDense(inputs, num_dims, activation=None):
  x = DenseWN("dense_in", num_outputs=num_dims, activation=activation)(inputs)
  non_linear = DenseWN("dense_nl", num_outputs=num_dims)(x)
  skip = DenseWN("dense_skip", num_outputs=num_dims)(x)
  return non_linear + skip


@gin.configurable("conv_hier_encoder")
@utils.MakeTFTemplate
def ConvHierEncoder(images, depth = 2, num_blocks = 2, z_dims = 32, h_dims=160):
  x = Conv2DWN("conv2d_in", num_filters=h_dims, stride=[2, 2], kernel_size=[5, 5])(inputs=images - 0.5)
  means = []
  raw_scales = []
  contexts = []
  for i in range(depth):
    for j in range(num_blocks):
      downsample = i > 0 and j == 0
      if downsample:
        stride = [2, 2]
      else:
        stride = [1, 1]

      h = tf.nn.elu(x)
      h = Conv2DWN("conv2d_in_%d_%d"%(i, j), num_filters=2*z_dims + 2 * h_dims, stride=stride, kernel_size=[3, 3])(inputs=h)
      mean, raw_scale, context, h = tf.split(h, [z_dims, z_dims, h_dims, h_dims], -1)
      means.append(mean)
      raw_scales.append(raw_scale)
      contexts.append(context)
      h = tf.nn.elu(h)
      h = Conv2DWN("conv2d_h_%d_%d"%(i, j), num_filters=h_dims, stride=[1, 1], kernel_size=[3, 3])(inputs=h)
      if downsample:
        x = tf.image.resize_nearest_neighbor(x, [int(x.shape[1]) // 2, int(x.shape[2]) // 2])
      x += 0.1 * h

  return means, raw_scales, contexts

@gin.configurable("conv_hier_prior_post")
@utils.MakeTFTemplate
def ConvHierPriorPost(images=None,
                      encoder=None,
                      z=None,
                      batch=None,
                      depth = 2,
                      num_blocks = 2,
                      z_dims = 32,
                      h_dims = 160,
                      image_width = 32):
  is_q = encoder is not None
  if is_q:
    means, raw_scales, up_contexts = encoder(images)

  if batch is None:
    if images is not None:
      batch = tf.shape(images)[0]
    else:
      batch = tf.shape(z[0])[0]
  h = tf.get_variable("h_top", [h_dims], initializer=tf.zeros_initializer())
  h = tf.reshape(h, [1, 1, 1, -1])
  top_width = image_width // 2 ** num_blocks
  h = tf.tile(h, [batch, top_width, top_width, 1])
  x = h

  ret_z = []
  ret_log_pz = []
  for i in reversed(list(range(depth))):
    for j in reversed(list(range(num_blocks))):
      downsample = i > 0 and j == 0

      h = tf.nn.elu(x)
      h_p = Conv2DWN(
          "conv2d_p_%d_%d" % (i, j),
          num_filters=2 * h_dims + 2 * z_dims,
          stride=[1, 1],
          kernel_size=[3, 3])(
              inputs=h)
      p_mean, p_raw_scale, down_context, h_det = tf.split(
          h_p, [z_dims, z_dims, h_dims, h_dims], -1)
      p_z = tfd.Independent(
          tfd.Normal(loc=p_mean, scale=tf.nn.softplus(p_raw_scale)),
          reinterpreted_batch_ndims=3)

      if is_q:
        h_q = Conv2DWN(
            "conv2d_q_%d_%d" % (i, j),
            num_filters=2 * z_dims,
            stride=[1, 1],
            kernel_size=[3, 3])(
                inputs=h)
        q_mean, q_raw_scale = tf.split(h_q, [z_dims, z_dims], -1)
        context = down_context + up_contexts.pop()
        q_mean += means.pop()
        q_raw_scale += raw_scales.pop()

        num_flat_dims = np.prod(q_mean.shape.as_list()[1:])

        _maf_template = ConvAR(
            "iaf_%d_%d" % (i, j),
            real_event_shape=q_mean.shape.as_list()[1:],
            hidden_layers=[h_dims, h_dims],
            h=context,
            weights_initializer=utils.L2HMCInitializer(factor=0.01))

        def maf_template(x, t=_maf_template):
          # TODO: I don't understand why the shape gets lost.
          #x.set_shape([None, num_flat_dims])
          x.set_shape([None] + q_mean.shape.as_list()[1:])
          return t(x)

        bijectors = []
        #bijectors.append(tfb.Reshape(tf.shape(q_mean)[1:], [num_flat_dims]))
        bijectors.append(
            tfb.Invert(
                tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=maf_template)))
        #bijectors.append(tfb.Reshape([num_flat_dims], tf.shape(q_mean)[1:]))
        # Do the shift/scale explicitly, so that we can use bijector to map the
        # distribution to the standard normal, which is helpful for HMC.
        bijectors.append(tfb.AffineScalar(shift=q_mean, scale=tf.nn.softplus(q_raw_scale)))

        bijector = tfb.Chain(bijectors)

        mvn = tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(q_mean), scale=tf.ones_like(q_raw_scale)),
            reinterpreted_batch_ndims=3)
        q_z = tfd.TransformedDistribution(mvn, bijector)

      if is_q:
        dist = q_z
      else:
        dist = p_z

      if z is None:
        z_val = dist.sample()
      else:
        z_val = z[0]
        z = z[1:]

      ret_z.append(z_val)
      ret_log_pz.append(dist.log_prob(z_val))

      h = tf.concat([z_val, h_det], -1)

      if downsample:
        new_shape = [2 * int(x.shape[1]), 2 * int(x.shape[2])]
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        h = tf.image.resize_nearest_neighbor(h, new_shape)
      h = Conv2DWN("deconv2d_%d_%d" % (i, j), num_filters=h_dims, stride=[1, 1], kernel_size=[3, 3])(inputs=h)
      x = x + 0.1 * h

  x = tf.image.resize_nearest_neighbor(x, [2 * int(x.shape[1]), 2 * int(x.shape[2])])
  x = Conv2DWN("conv2d_out", num_filters=3, stride=[1, 1], kernel_size=[5, 5])(inputs=x)

  return ret_z, ret_log_pz, x

@gin.configurable("conv_encoder")
@utils.MakeTFTemplate
def ConvEncoder(images, num_outputs, hidden_dims = 450,
                filter_scale = 1, fully_convolutional = False):
  x = images
  x = ResConv2D("res_1", num_filters=filter_scale * 16, kernel_size=[3, 3], stride=[2, 2])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_2", num_filters=filter_scale * 16, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_3", num_filters=filter_scale * 16, kernel_size=[3, 3], stride=[2, 2])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_4", num_filters=filter_scale * 32, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_5", num_filters=filter_scale * 32, kernel_size=[3, 3], stride=[2, 2])(x)
  x = tf.nn.elu(x)
  if fully_convolutional:
    return ResConv2D("res_out", num_filters=num_outputs, kernel_size=[3, 3], stride=[1, 1])(x)
  else:
    x = tf.reshape(x, [-1, filter_scale * 32 * 4 * 4])
    x = ResDense("dense_h", num_dims=hidden_dims, activation=tf.nn.elu)(x)
    return DenseWN(
        "dense_out",
        num_outputs=num_outputs,
        weights_initializer=utils.L2HMCInitializer())(
            x)


@gin.configurable("conv_decoder")
@utils.MakeTFTemplate
def ConvDecoder(encoding,
                output_shape,
                filter_scale = 1,
                hidden_dims = 450,
                fully_convolutional = False):
  if isinstance(encoding, (list, tuple)):
    encoding = encoding[0]
  if fully_convolutional:
    tf.logging.info("Encoding shape: %s", encoding.shape)
    x = ResConv2D("res_in", num_filters=filter_scale * 32, kernel_size=[3, 3], stride=[1, 1])(encoding)
  else:
    x = ResDense("dense_in", num_dims=hidden_dims, activation=tf.nn.elu)(encoding)
    x = ResDense("dense_h", num_dims=filter_scale * 32 * 4 * 4, activation=tf.nn.elu)(x)
    x = tf.reshape(x, [-1, 4, 4, filter_scale * 32])
  x = tf.image.resize_nearest_neighbor(x, [8, 8])
  x = ResConv2D("res_5", num_filters=32 * filter_scale, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_4", num_filters=32 * filter_scale, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  if output_shape[1] == 28:
    # 8x8 -> 7x7
    x = x[:, 1:, 1:, :]
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0] // 2, output_shape[1] // 2])
  x = ResConv2D("res_3", num_filters=16 * filter_scale, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_2", num_filters=16 * filter_scale, kernel_size=[3, 3], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0], output_shape[1]])
  x = ResConv2D(
      "res_1",
      num_filters=output_shape[-1],
      kernel_size=[3, 3],
      stride=[1, 1],
      output_init_factor=0.01)(
          x)
  return tf.reshape(x, [-1] + output_shape)


@gin.configurable("conv_encoder2")
@utils.MakeTFTemplate
def ConvEncoder2(images, num_outputs, filter_scale = 1):
  x = images
  x = Conv2DWN("conv_1", num_filters=filter_scale * 16, kernel_size=[5, 5], stride=[2, 2], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_2", num_filters=filter_scale * 16, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_3", num_filters=filter_scale * 16, kernel_size=[5, 5], stride=[2, 2], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_4", num_filters=filter_scale * 32, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_5", num_filters=filter_scale * 32, kernel_size=[5, 5], stride=[2, 2], activation=tf.nn.elu)(x)
  return ResConv2D("conv_out", num_filters=num_outputs, kernel_size=[3, 3], stride=[1, 1])(x)


@gin.configurable("conv_decoder2")
@utils.MakeTFTemplate
def ConvDecoder2(encoding,
                output_shape,
                filter_scale = 1):
  if isinstance(encoding, (list, tuple)):
    encoding = encoding[0]
  x = Conv2DWN("conv_in", num_filters=filter_scale * 32, kernel_size=[3, 3], stride=[1, 1])(encoding)
  x = tf.image.resize_nearest_neighbor(x, [8, 8])
  x = Conv2DWN("conv_5", num_filters=32 * filter_scale, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_4", num_filters=32 * filter_scale, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  if output_shape[1] == 28:
    # 8x8 -> 7x7
    x = x[:, 1:, 1:, :]
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0] // 2, output_shape[1] // 2])
  x = Conv2DWN("conv_3", num_filters=16 * filter_scale, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  x = Conv2DWN("conv_2", num_filters=16 * filter_scale, kernel_size=[5, 5], stride=[1, 1], activation=tf.nn.elu)(x)
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0], output_shape[1]])
  x = Conv2DWN(
      "conv_1",
      num_filters=output_shape[-1],
      kernel_size=[5, 5],
      stride=[1, 1],
      weights_initializer=utils.L2HMCInitializer(0.01))(
          x)
  return tf.reshape(x, [-1] + output_shape)


@gin.configurable("conv_encoder3")
@utils.MakeTFTemplate
def ConvEncoder3(images, num_outputs, hidden_dims = 450,
                 filter_scale = 1):
  # This comes from VLAE paper.
  x = images
  x = ResConv2D("res_1", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_2", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = Conv2DWN("conv_3", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[2, 2])(x)
  x = tf.nn.elu(x)

  x = ResConv2D("res_4", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_5", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = Conv2DWN("conv_6", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[2, 2])(x)
  x = tf.nn.elu(x)

  x = ResConv2D("res_7", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_8", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_9", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  return Conv2DWN("conv_10", num_filters=num_outputs, kernel_size=[1, 1], stride=[1, 1])(x)


@gin.configurable("conv_decoder3")
@utils.MakeTFTemplate
def ConvDecoder3(encoding,
                output_shape,
                filter_scale = 1):
  if isinstance(encoding, (list, tuple)):
    encoding = encoding[0]
  x = encoding
  x = Conv2DWN("conv_1", num_filters=filter_scale * 96, kernel_size=[1, 1], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_2", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_3", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_4", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0] // 2, output_shape[1] // 2])
  x = Conv2DWN("conv_5", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)

  x = ResConv2D("res_6", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_7", num_filters=filter_scale * 96, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0], output_shape[1]])
  x = Conv2DWN("conv_8", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)

  x = ResConv2D("res_9", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = ResConv2D("res_10", num_filters=filter_scale * 48, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = Conv2DWN(
      "conv_out",
      num_filters=output_shape[-1],
      kernel_size=[5, 5],
      stride=[1, 1],
      weights_initializer=utils.L2HMCInitializer(0.01))(
          x)
  return tf.reshape(x, [-1] + output_shape)


@gin.configurable("conv_encoder4")
@utils.MakeTFTemplate
def ConvEncoder4(images, num_outputs,
                 filter_scale = 1,
                 fully_convolutional = False):
  x = images
  x = Conv2DWN("conv_1", num_filters=filter_scale * 64, kernel_size=[5, 5], stride=[2, 2])(x)
  x = tf.nn.elu(x)
  x = Conv2DWN("conv_2", num_filters=filter_scale * 64, kernel_size=[5, 5], stride=[2, 2])(x)
  x = tf.nn.elu(x)
  if fully_convolutional:
    return Conv2DWN("conv_out", num_filters=num_outputs, kernel_size=[1, 1], stride=[1, 1])(x)
  else:
    return DenseWN("dense_out", num_outputs=num_outputs)(tf.layers.flatten(x))


@gin.configurable("conv_decoder4")
@utils.MakeTFTemplate
def ConvDecoder4(encoding,
                output_shape,
                filter_scale = 1,
                fully_convolutional = False):
  if isinstance(encoding, (list, tuple)):
    encoding = encoding[0]
  x = encoding
  if not fully_convolutional:
    x = tf.reshape(DenseWN("dense_in", num_outputs=8*8*16)(x), [-1, 8, 8, 16])
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0] // 2, output_shape[1] // 2])
  x = Conv2DWN("conv_1", num_filters=filter_scale * 64, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = tf.image.resize_nearest_neighbor(x, [output_shape[0], output_shape[1]])
  x = Conv2DWN("conv_2", num_filters=filter_scale * 64, kernel_size=[5, 5], stride=[1, 1])(x)
  x = tf.nn.elu(x)
  x = Conv2DWN(
      "conv_out",
      num_filters=output_shape[-1],
      kernel_size=[1, 1],
      stride=[1, 1],
      weights_initializer=utils.L2HMCInitializer(0.01))(
          x)
  return tf.reshape(x, [-1] + output_shape)


@gin.configurable("dense_encoder")
@utils.MakeTFTemplate
def DenseEncoder(images,
                 num_outputs,
                 hidden_layer_sizes = [1024, 1024],
                 activation=tf.nn.elu):
  x = tf.layers.flatten(images)
  # Center the data, assuming it goes from [0, 1] initially.
  # x = 2.0 * x - 1.0
  for size in hidden_layer_sizes:
    x = tf.layers.dense(
        x, size, activation=activation, kernel_initializer=utils.L2HMCInitializer())
  return tf.layers.dense(x, num_outputs, kernel_initializer=utils.L2HMCInitializer())


@gin.configurable("dense_decoder")
@utils.MakeTFTemplate
def DenseDecoder(encoding,
                 output_shape,
                 hidden_layer_sizes = [1024, 1024],
                 activation=tf.nn.elu):
  if isinstance(encoding, (list, tuple)):
    encoding = encoding[0]
  x = tf.layers.flatten(encoding)
  for size in hidden_layer_sizes:
    x = tf.layers.dense(
        x, size, activation=activation, kernel_initializer=utils.L2HMCInitializer())
  num_outputs = np.prod(output_shape)
  return tf.reshape(
      tf.layers.dense(
          x, num_outputs, kernel_initializer=utils.L2HMCInitializer(factor=0.01)),
      [-1] + output_shape)


def IndependentBernouli3D(logits):
  return tfd.Independent(
      tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=3)


def IndependentDiscreteLogistic3D(locations,
                                  scales):
  dist = tfd.TransformedDistribution(
      distribution=tfd.Logistic(loc=locations, scale=scales),
      bijector=tfb.AffineScalar(scale=255.0))

  dist = tfd.QuantizedDistribution(distribution=dist, low=0., high=255.0)

  dist = tfd.Independent(dist, reinterpreted_batch_ndims=3)

  class ScaleHack(object):
    def __init__(self, dist):
      self._dist = dist

    def sample(self, *args, **kwargs):
      return self._dist.sample(*args, **kwargs) / 255.0

    def log_prob(self, x, *args, **kwargs):
      return self._dist.log_prob(tf.clip_by_value(x * 255.0, 0.0, 255.0), *args, **kwargs)

  return ScaleHack(dist)


def IndependentDiscreteLogistic3D2(locations,
                                   scales):
  class IndependentDiscreteLogistic(object):
    def __init__(self, loc, scale):
      self._loc = loc
      self._scale = scale

    def sample(self, *args, **kwargs):
      dist = tfd.Logistic(loc=self._loc, scale=self._scale)
      return tf.clip_by_value(dist.sample(*args, **kwargs), 0.0, 1.0)

    def log_prob(self, x, *args, **kwargs):
      sample = x
      mean = self._loc
      scales = self._scale
      binsize=1.0 / 256.0
      sample = (tf.floor(sample / binsize) * binsize - mean) / scales
      return tf.reduce_sum(
          tf.log(
              tf.sigmoid(sample + binsize / scales) - tf.sigmoid(sample) + 1e-7),
          [-1, -2, -3])

  return IndependentDiscreteLogistic(locations, scales)


@gin.configurable("dense_recognition")
@utils.MakeTFTemplate
def DenseRecognition(images, encoder, z=None, sigma_activation="exp"
                         ):
  """Models Q(z | encoder(x))"""
  encoding = encoder(images)

  num_dims = int(encoding.shape[-1]) // 2
  encoding_parts = tf.unstack(
      tf.reshape(encoding, [-1, num_dims, 2]), num=2, axis=-1)

  mu = encoding_parts[0]
  if sigma_activation == "exp":
    sigma = tf.exp(0.5 * encoding_parts[1])
  elif sigma_activation == "softplus":
    sigma = tf.nn.softplus(encoding_parts[1])

  bijector = tfb.Affine(shift=mu, scale_diag=sigma)

  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]
  tf.logging.info("bijector z shape: %s", z[0].shape)
  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@gin.configurable("dense_recognition_affine")
@utils.MakeTFTemplate
def DenseRecognitionAffine(images, encoder, z=None,
                         z_dims=None):
  """Models Q(z | encoder(x))"""
  encoding = encoder(images)

  mu = encoding[:, :z_dims]
  tril_raw = tfp.math.fill_triangular(encoding[:, z_dims:])
  sigma = tf.nn.softplus(tf.matrix_diag_part(tril_raw))
  tril = tf.linalg.set_diag(tril_raw, sigma)
  bijector = tfb.Affine(shift=mu, scale_tril=tril)

  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]
  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@gin.configurable("dense_recognition_affine_lr")
@utils.MakeTFTemplate
def DenseRecognitionAffineLR(images, encoder, z=None,
                            z_dims=None, rank=1):
  """Models Q(z | encoder(x))"""
  encoding = encoder(images)

  mu = encoding[:, :z_dims]
  sigma = encoding[:, z_dims:2*z_dims]
  perturb = encoding[:, 2*z_dims:]
  perturb = tf.reshape(perturb, [-1, z_dims, rank])

  sigma = tf.nn.softplus(sigma)
  bijector = tfb.Affine(shift=mu, scale_diag=sigma,
                        scale_perturb_factor=perturb)

  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]
  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@gin.configurable("dense_recognition_rnvp")
@utils.MakeTFTemplate
def DenseRecognitionRNVP(
    images,
    encoder,
    z=None,
    num_bijectors=3,
    condition_bijector=False,
    layer_sizes=[128, 128],
    sigma_activation="exp"):
  """Models Q(z | encoder(x)), z = f(w, encoder)"""
  encoding = encoder(images)

  if condition_bijector:
    num_parts = 3
  else:
    num_parts = 2
  num_dims = int(encoding.shape[-1]) // num_parts

  encoding_parts = tf.unstack(
      tf.reshape(encoding, [-1, num_dims, num_parts]), num=num_parts, axis=-1)
  if condition_bijector:
    h = encoding_parts[2]
  else:
    h = None

  swap = tfb.Permute(permutation=np.arange(num_dims - 1, -1, -1))

  bijectors = []
  for i in range(num_bijectors):
    _rnvp_template = utils.DenseShiftLogScale(
        "rnvp_%d" % i,
        h=h,
        hidden_layers=layer_sizes,
        activation=tf.nn.softplus,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01))

    def rnvp_template(x, output_units, t=_rnvp_template):
      # TODO: I don't understand why the shape gets lost.
      x.set_shape([None, num_dims - output_units])
      return t(x, output_units)

    bijectors.append(
        tfb.Invert(
            tfb.RealNVP(
                num_masked=num_dims // 2,
                shift_and_log_scale_fn=rnvp_template)))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]

  # Do the shift/scale explicitly, so that we can use bijector to map the
  # distribution to the standard normal, which is helpful for HMC.
  mu = encoding_parts[0]
  if sigma_activation == "exp":
    sigma = tf.exp(0.5 * encoding_parts[1])
  elif sigma_activation == "softplus":
    sigma = tf.nn.softplus(encoding_parts[1])

  bijectors.append(tfb.Affine(shift=mu, scale_diag=sigma))
  bijector = tfb.Chain(bijectors)

  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]

  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@gin.configurable("dense_recognition_iaf")
@utils.MakeTFTemplate
def DenseRecognitionIAF(
    images,
    encoder,
    z=None,
    num_iaf_layers=2,
    iaf_layer_sizes=[128, 128],
    condition_iaf=False,
    sigma_activation="exp"):
  """Models Q(z | encoder(x)), z = f(w, encoder)"""
  encoding = encoder(images)

  if condition_iaf:
    num_parts = 3
  else:
    num_parts = 2
  num_dims = int(encoding.shape[-1]) // num_parts

  encoding_parts = tf.unstack(
      tf.reshape(encoding, [-1, num_dims, num_parts]), num=num_parts, axis=-1)
  if condition_iaf:
    h = encoding_parts[2]
  else:
    h = None

  swap = tfb.Permute(permutation=np.arange(num_dims - 1, -1, -1))

  bijectors = []
  for i in range(num_iaf_layers):
    #_maf_template = tfb.masked_autoregressive_default_template(
    #    hidden_layers=iaf_layer_sizes,
    #    activation=tf.nn.softplus,
    #    kernel_initializer=utils.L2HMCInitializer(factor=0.01))
    _maf_template = utils.DenseAR(
        "maf_%d" % i,
        hidden_layers=iaf_layer_sizes,
        h=h,
        activation=tf.nn.softplus,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01))

    def maf_template(x, t=_maf_template):
      # TODO: I don't understand why the shape gets lost.
      x.set_shape([None, num_dims])
      return t(x)

    bijectors.append(
        tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=maf_template)))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]

  # Do the shift/scale explicitly, so that we can use bijector to map the
  # distribution to the standard normal, which is helpful for HMC.
  mu = encoding_parts[0]
  if sigma_activation == "exp":
    sigma = tf.exp(0.5 * encoding_parts[1])
  elif sigma_activation == "softplus":
    sigma = tf.nn.softplus(encoding_parts[1])

  bijectors.append(tfb.Affine(shift=mu, scale_diag=sigma))
  bijector = tfb.Chain(bijectors)

  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros_like(mu), scale_diag=tf.ones_like(sigma))
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]

  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


class FlipImageBijector(tfb.Bijector):

  def __init__(self, validate_args=False, name=None):
    """Creates the `Permute` bijector.

    Args:
      permutation: An `int`-like vector-shaped `Tensor` representing the
        permutation to apply to the rightmost dimension of the transformed
        `Tensor`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if `not permutation.dtype.is_integer`.
      ValueError: if `permutation` does not contain exactly one of each of
        `{0, 1, ..., d}`.
    """
    parameters = dict(locals())
    super(FlipImageBijector, self).__init__(
        forward_min_event_ndims=3,
        is_constant_jacobian=True,
        validate_args=validate_args,
        parameters=parameters,
        name=name or "flip_image")

  def _forward(self, x):
    return tf.image.flip_left_right(tf.image.flip_up_down(x))

  def _inverse(self, y):
    return tf.image.flip_up_down(tf.image.flip_left_right(y))

  def _inverse_log_det_jacobian(self, y):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    return tf.constant(0., dtype=y.dtype.base_dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., dtype=x.dtype.base_dtype)


@gin.configurable("conv_iaf")
@utils.MakeTFTemplate
def ConvIAF(
    images,
    encoder,
    z=None,
    num_iaf_layers=2,
    iaf_layer_sizes=[128, 128],
    condition_iaf=False,
    sigma_activation="softplus"):
  """Models Q(z | encoder(x)), z = f(w, encoder)"""
  encoding = encoder(images)
  if encoding.shape.ndims != 4:
    raise ValueError("ConvIAF requires a convolutional encoder. %s", encoding.shape)

  if condition_iaf:
    num_parts = 3
  else:
    num_parts = 2
  num_dims = int(encoding.shape[-1]) // num_parts

  encoding_parts = tf.unstack(
      tf.reshape(encoding, [-1] + encoding.shape.as_list()[1:-1] + [num_dims, num_parts]), num=num_parts, axis=-1)
  if condition_iaf:
    h = encoding_parts[2]
  else:
    h = None

  bijectors = []
  for i in range(num_iaf_layers):
    _maf_template = ConvAR(
        "iaf_%d" % i,
        real_event_shape=encoding_parts[0].shape.as_list()[1:],
        hidden_layers=iaf_layer_sizes,
        h=h,
        weights_initializer=utils.L2HMCInitializer(factor=0.01))

    def maf_template(x, t=_maf_template):
      # TODO: I don't understand why the shape gets lost.
      x.set_shape([None] + encoding_parts[0].shape.as_list()[1:])
      return t(x)

    bijectors.append(
        tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=maf_template)))
    bijectors.append(FlipImageBijector())
  # Drop the last swap.
  bijectors = bijectors[:-1]

  # Do the shift/scale explicitly, so that we can use bijector to map the
  # distribution to the standard normal, which is helpful for HMC.
  mu = encoding_parts[0]
  if sigma_activation == "exp":
    sigma = tf.exp(0.5 * encoding_parts[1])
  elif sigma_activation == "softplus":
    sigma = tf.nn.softplus(encoding_parts[1])

  bijectors.append(tfb.AffineScalar(shift=mu, scale=sigma))
  bijector = tfb.Chain(bijectors)

  mvn = tfd.Independent(
      tfd.Normal(loc=tf.zeros_like(mu), scale=tf.ones_like(sigma)),
         reinterpreted_batch_ndims=3)
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]
  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@gin.configurable("conv_shift_scale")
@utils.MakeTFTemplate
def ConvShiftScale(
    images,
    encoder,
    z=None,
    sigma_activation="softplus"):
  """Models Q(z | encoder(x)), z = f(w, encoder)"""
  encoding = encoder(images)
  if encoding.shape.ndims != 4:
    raise ValueError("ConvIAF requires a convolutional encoder. %s", encoding.shape)

  num_parts = 2
  num_dims = int(encoding.shape[-1]) // num_parts

  encoding_parts = tf.unstack(
      tf.reshape(encoding, [-1] + encoding.shape.as_list()[1:-1] + [num_dims, num_parts]), num=num_parts, axis=-1)

  # Do the shift/scale explicitly, so that we can use bijector to map the
  # distribution to the standard normal, which is helpful for HMC.
  mu = encoding_parts[0]
  if sigma_activation == "exp":
    sigma = tf.exp(0.5 * encoding_parts[1])
  elif sigma_activation == "softplus":
    sigma = tf.nn.softplus(encoding_parts[1])

  bijector = tfb.AffineScalar(shift=mu, scale=sigma)

  mvn = tfd.Independent(
      tfd.Normal(loc=tf.zeros_like(mu), scale=tf.ones_like(sigma)),
         reinterpreted_batch_ndims=3)
  dist = tfd.TransformedDistribution(mvn, bijector)

  if z is None:
    z = [dist.sample()]
  return z, [dist.log_prob(z[0])], [bijector]  # pytype: disable=bad-return-type


@utils.MakeTFTemplate
def SimplePrior(z=None, batch=None,
                num_dims=None):
  """Models P(z)"""
  mvn = tfd.MultivariateNormalDiag(
      loc=tf.zeros(num_dims), scale_diag=tf.ones(num_dims))

  if z is None:
    z = [mvn.sample(batch)]

  return z, [mvn.log_prob(z[0])]  # pytype: disable=bad-return-type


@utils.MakeTFTemplate
def Simple3DPrior(z=None, batch=None,
                  shape=None):
  """Models P(z)"""
  mvn = tfd.Independent(tfd.Normal(loc=tf.zeros(shape), scale=tf.ones(shape)), reinterpreted_batch_ndims=3)

  if z is None:
    z = [mvn.sample(batch)]

  return z, [mvn.log_prob(z[0])]  # pytype: disable=bad-return-type

@utils.MakeTFTemplate
def DenseMNISTNoise(x=None, z=None, decoder=None, return_means=True):
  """Models P(x | decoder(z))"""
  decoding = decoder(z)
  bernoulli = IndependentBernouli3D(decoding)

  if x is None:
    if return_means:
      x = bernoulli.mean()
    else:
      x = tf.to_float(bernoulli.sample())

  return x, bernoulli.log_prob(x)


@gin.configurable("cifar10_noise")
@utils.MakeTFTemplate
def DenseCIFAR10TNoise(x=None, z=None, decoder=None, return_means=True, uniform_scale=False, logistic_impl="mine"):
  """Models P(x | decoder(z))"""
  decoding = decoder(z)
  if uniform_scale:
    scale = tf.get_variable("scale", initializer=1.0)
    scales = tf.reshape(scale, [1, 1, 1])
  else:
    scales = tf.get_variable(
        "scales", [32, 32, 3], initializer=tf.ones_initializer())

  if logistic_impl == "mine":
    disc_logistic = IndependentDiscreteLogistic3D(decoding, tf.nn.softplus(scales))
  elif logistic_impl == "kingma":
    disc_logistic = IndependentDiscreteLogistic3D2(decoding, tf.nn.softplus(scales))

  if x is None:
    x = tf.to_float(disc_logistic.sample())

  return x, disc_logistic.log_prob(x)


@gin.configurable("learning_rate")
def LearningRate(train_size, global_step, schedule = "hoffman", warmup_steps=0):
  if schedule == "hoffman":
    base = tf.train.piecewise_constant(
        global_step, [train_size * 500 // TRAIN_BATCH], [1e-3, 1e-4])
  elif schedule == "new":
    base = tf.train.piecewise_constant(
        global_step,
        [train_size * 500 // TRAIN_BATCH, train_size * 800 // TRAIN_BATCH],
        [1e-3, 1e-4, 1e-5])
  elif schedule == "new_gentle":
    base = tf.train.piecewise_constant(
        global_step,
        [train_size * 500 // TRAIN_BATCH, train_size * 800 // TRAIN_BATCH],
        [0.5e-3, 1e-4, 1e-5])
  elif schedule == "fast":
    base = tf.train.piecewise_constant(
        global_step,
        [train_size * 800 // TRAIN_BATCH],
        [1e-2, 1e-5])
  else:
    raise ValueError("Invalid schedule: " + schedule)
  if warmup_steps == 0:
    return base
  else:
    return tf.minimum(base * tf.to_float(global_step) / warmup_steps, base)


VAEOutputs = collections.namedtuple(
    "VAEOutputs", "log_p_x_z, elbo, sample_means, recon_means, klqp, total_klqp, post_z, prior_z")


AISOutputs = collections.namedtuple(
    "AISOutputs",
    "log_p, p_accept, z_fin, recon"
)

def MakeVAE(images, recognition, prior, noise, beta, num_samples,
        min_kl):
  z, log_q_z = recognition(images)
  _, log_p_z = prior(z)
  _, log_p_x_z = noise(images, z)

  post_z = z

  log_q_z = [tf.reduce_mean(layer_log_q_z) for layer_log_q_z in log_q_z]
  log_p_z = [tf.reduce_mean(layer_log_p_z) for layer_log_p_z in log_p_z]
  log_p_x_z = tf.reduce_mean(log_p_x_z)

  klqp = [layer_log_q_z - layer_log_p_z for layer_log_q_z, layer_log_p_z in zip(log_q_z, log_p_z)]
  klqp = [tf.maximum(min_kl, layer_klqp) for layer_klqp in klqp]
  total_klqp = tf.add_n(klqp)
  elbo = log_p_x_z - beta * total_klqp

  recon_means, _ = noise(None, z)

  z, _ = prior(batch=num_samples)
  sample_means, _ = noise(None, z)

  return VAEOutputs(
      log_p_x_z=log_p_x_z,
      elbo=elbo,
      sample_means=sample_means,
      recon_means=recon_means,
      klqp=klqp,
      total_klqp=total_klqp,
      post_z=post_z,
      prior_z=z)


DLGMOutputs = collections.namedtuple(
    "DLGMOutputs",
    "elbo, sample_means, mcmc_log_p, recon_means, p_accept, post_z, post_z_chain, q_z, xentpq"
)


@gin.configurable("dlgm")
class DLGM(object):

  def __init__(self,
               z_dims=64,
               beta=1.0,
               beta_steps=0,
               step_size=0.2,
               num_leapfrog_steps=5,
               num_hmc_steps=2,
               use_neutra=True,
               condition_bijector=False,
               bijector_type="iaf",
               encoder_type="dense",
               q_loss_type="klqp",
               min_kl=0.0,
               symm_factor=0.5,
               save_chain_state=False,
               chain_warmup_epochs=5,
               use_q_z_for_gen=False,
               no_gen_train_steps=0,
               dataset=None,
               use_bijector_for_ais=False,
               prior_type="simple",
               adapt_step_size=False,
               step_size_gain=1e-3,
               use_q_z_for_ais=False,
               affine_rank=1,
               step_size_warmup=0):
    self.train_size = dataset.train_size
    self._use_q_z_for_ais = use_q_z_for_ais
    if dataset.name == "mnist":
      output_shape = [28, 28, 1]
    elif dataset.name == "cifar10":
      output_shape = [32, 32, 3]
    self._z_dims = z_dims
    self._use_bijector_for_ais = use_bijector_for_ais
    if beta_steps > 0:
      frac = tf.to_float(
          tf.train.get_or_create_global_step()) / tf.to_float(beta_steps)
      frac = tf.minimum(frac, 1.0)
      self._beta = frac * beta
    else:
      self._beta = tf.constant(beta)
    self._min_kl = tf.to_float(min_kl)
    self._use_neutra = use_neutra
    self._num_leapfrog_steps = num_leapfrog_steps
    self._num_hmc_steps = num_hmc_steps
    self._q_loss_type = q_loss_type
    self._symm_factor = symm_factor
    self._save_chain_state = save_chain_state
    self._chain_warmup_epochs = chain_warmup_epochs
    self._use_q_z_for_gen = use_q_z_for_gen
    self._no_gen_train_steps = no_gen_train_steps
    self._step_size_gain = step_size_gain
    self._adapt_step_size = adapt_step_size
    self._step_size_warmup = step_size_warmup
    self._init_step_size = step_size
    if self._adapt_step_size:
      self._step_size = tf.get_variable("step_size", initializer=step_size)
    else:
      self._step_size = tf.constant(step_size)

    if self._save_chain_state:
      self._chain_state = tf.get_variable(
          "train_chain_state", [self.train_size, z_dims], trainable=False)

    if bijector_type == "affine":
      # TriL + shift
      num_outputs = (z_dims * (z_dims + 1)) // 2 + z_dims
    elif bijector_type == "affine_lr":
      num_outputs = z_dims * 2 + z_dims * affine_rank
    elif condition_bijector and bijector_type not in ["conv_shift_scale", "shift_scale"]:
      num_outputs = 3 * z_dims
    else:
      num_outputs = 2 * z_dims

    if encoder_type == "hier_conv":
      #assert dataset.name == "cifar10"
      #self._encoder = ConvHierEncoder("encoder")
      #self._prior_posterior = ConvHierPriorPost("prior_post")
      #self._decoder = lambda z: self._prior_posterior(z=z)[2]
      #self._prior = lambda z=None, batch=None: self._prior_posterior(z=z, batch=batch)[:2]
      #self._recog = lambda images, z=None: self._prior_posterior(images=images, z=z, encoder=self._encoder)[:2]
      pass
    else:
      if encoder_type == "dense":
        self._encoder = DenseEncoder(
            "encoder", num_outputs=num_outputs, activation=tf.nn.softplus)
        self._decoder = DenseDecoder(
            "decoder", output_shape=output_shape, activation=tf.nn.softplus)
      elif encoder_type == "conv":
        self._encoder = ConvEncoder("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder("decoder", output_shape=output_shape)
        conv_z_shape = [4, 4, self._z_dims]
      elif encoder_type == "conv2":
        self._encoder = ConvEncoder2("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder2("decoder", output_shape=output_shape)
        conv_z_shape = [4, 4, self._z_dims]
      elif encoder_type == "conv3":
        self._encoder = ConvEncoder3("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder3("decoder", output_shape=output_shape)
        conv_z_shape = [8, 8, self._z_dims]
      elif encoder_type == "conv4":
        self._encoder = ConvEncoder4("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder4("decoder", output_shape=output_shape)
        conv_z_shape = [8, 8, self._z_dims]

      if prior_type == "simple":
        self._prior = SimplePrior("prior", num_dims=self._z_dims)
      elif prior_type == "simple_3d":
        self._prior = Simple3DPrior("prior", shape=conv_z_shape)
      if bijector_type == "iaf":
        recog = DenseRecognitionIAF(
            "recog", encoder=self._encoder, condition_iaf=condition_bijector)
      elif bijector_type == "rnvp":
        recog = DenseRecognitionRNVP(
            "recog",
            encoder=self._encoder,
            condition_bijector=condition_bijector)
      elif bijector_type == "shift_scale":
        recog = DenseRecognition(
            "recog",
            encoder=self._encoder)
      elif bijector_type == "conv_shift_scale":
        recog = ConvShiftScale("recog", encoder=self._encoder)
      elif bijector_type == "affine":
        recog = DenseRecognitionAffine("recog", encoder=self._encoder, z_dims=z_dims)
      elif bijector_type == "affine_lr":
        recog = DenseRecognitionAffineLR("recog", encoder=self._encoder, z_dims=z_dims, rank=affine_rank)
      elif bijector_type == "conv_iaf":
        recog = ConvIAF("recog", encoder=self._encoder, condition_iaf=condition_bijector)
      self._recog = recog

    if dataset.name == "mnist":
      self._noise = DenseMNISTNoise("noise", decoder=self._decoder)
    else:
      self._noise = DenseCIFAR10TNoise("noise", decoder=self._decoder)

  def AdjustedStepSize(self):
    if self._step_size_warmup > 0:
      global_step = tf.train.get_or_create_global_step()
      max_step = self._init_step_size * tf.to_float(
          global_step) / self._step_size_warmup
      return tf.where(global_step > self._step_size_warmup, self._step_size,
                      tf.minimum(max_step, self._step_size))
    else:
      return self._step_size

  def RecogVars(self):
    return self._encoder.variables + self._recog.variables

  def GenVars(self):
    return (
        self._prior.variables + self._decoder.variables + self._noise.variables)

  def MakeDLGM(self,
               images,
               other_z_init=None,
               use_other_z_init=None,
               num_samples=64):
    z, log_q_z, bijector = self._recog(images)
    _, log_p_z = self._prior(z)
    _, log_p_x_z = self._noise(images, z)

    post_z = z

    q_z = z

    if use_other_z_init is not None:
      z_init = [tf.cond(use_other_z_init, lambda: tf.identity(other_layer_z),
                        lambda: tf.identity(layer_z)) for other_layer_z, layer_z in zip(z, other_z_init)]
    z_init = z

    log_q_z = [tf.reduce_mean(layer_log_q_z) for layer_log_q_z in log_q_z]
    log_p_z = [tf.reduce_mean(layer_log_p_z) for layer_log_p_z in log_p_z]
    log_p_x_z = tf.reduce_mean(log_p_x_z)

    klqp = [layer_log_q_z - layer_log_p_z for layer_log_q_z, layer_log_p_z in zip(log_q_z, log_p_z)]
    klqp = [tf.maximum(self._min_kl, layer_klqp) for layer_klqp in klqp]
    total_klqp = tf.add_n(klqp)
    elbo = log_p_x_z - self._beta * total_klqp

    def TargetLogProbFn(*z):
      for post_z_e, z_e in zip(post_z, z):
        tf.logging.info("Shape here: %s %s", post_z_e.shape, z_e.shape)
        z_e.set_shape(post_z_e.shape)
      _, log_p_z = self._prior(z)
      _, log_p_x_z = self._noise(images, z)
      return tf.add_n(log_p_z) + log_p_x_z

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=TargetLogProbFn,
        step_size=self.AdjustedStepSize(),
        num_leapfrog_steps=self._num_leapfrog_steps)
    if self._use_neutra:
      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=kernel, bijector=bijector)

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=self._num_hmc_steps, current_state=z, kernel=kernel)
    z = [tf.stop_gradient(s[-1, Ellipsis]) for s in states]

    post_z = z

    _, log_q_z, _ = self._recog(images, z=z)

    xentpq = -tf.add_n([tf.reduce_mean(layer_log_q_z) for layer_log_q_z in log_q_z])

    if self._use_q_z_for_gen:
      z = q_z

    recon_means, _ = self._noise(None, z)
    _, log_p_z = self._prior(z)
    _, log_p_x_z = self._noise(images, z)

    mcmc_log_p = tf.reduce_mean(tf.add_n(log_p_z) + log_p_x_z)

    if self._use_neutra:
      log_accept_ratio = kernel_results.inner_results.log_accept_ratio
    else:
      log_accept_ratio = kernel_results.log_accept_ratio

    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

    z, _ = self._prior(batch=num_samples)
    sample_means, _ = self._noise(None, z)

    return DLGMOutputs(
        elbo=elbo,
        sample_means=sample_means,
        mcmc_log_p=mcmc_log_p,
        recon_means=recon_means,
        p_accept=p_accept,
        post_z=post_z,
        post_z_chain=states,
        q_z=z_init,
        xentpq=xentpq)

  def GetPosterior(self, images):
    outputs = self.MakeDLGM(images)
    return outputs.post_z

  def TrainOp(self, data_idx, images):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = LearningRate(self.train_size, global_step)

    if self._save_chain_state:
      other_z_init = tf.gather(self._chain_state, data_idx)
      use_other_z_init = (
          global_step > self._chain_warmup_epochs * self.train_size // TRAIN_BATCH)
    else:
      other_z_init = None
      use_other_z_init = None

    outputs = self.MakeDLGM(
        images, other_z_init=other_z_init, use_other_z_init=use_other_z_init)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #gen_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    utils.LogAndSummarizeMetrics({
        "learning_rate": learning_rate,
        "elbo": outputs.elbo,
        "mcmc_log_p": outputs.mcmc_log_p,
        "mcmc_p_accept": outputs.p_accept,
        "step_size": self.AdjustedStepSize(),
    }, False)

    tf.summary.image(
        "sample_means", utils.StitchImages(outputs.sample_means))

    if self._save_chain_state:
      with tf.control_dependencies([outputs.post_z]):
        chain_state_update_op = tf.scatter_update(self._chain_state, data_idx,
                                                  outputs.post_z)
    else:
      chain_state_update_op = tf.no_op()

    if self._adapt_step_size:
      new_step_size = self._step_size + self._step_size_gain * (outputs.p_accept - 0.651)
      new_step_size = tf.clip_by_value(new_step_size, 1e-3, 0.5)
      step_size_op = self._step_size.assign(
          tf.where(global_step > self._step_size_warmup, new_step_size,
                   self._step_size))
    else:
      step_size_op = tf.no_op()

    with tf.name_scope("recog_train"):
      if self._q_loss_type == "klqp":
        loss = -outputs.elbo
      elif self._q_loss_type == "symm":
        loss = (
            self._symm_factor * -outputs.elbo +
            (1.0 - self._symm_factor) * outputs.xentpq)
      elif self._q_loss_type == "klpq":
        loss = outputs.xentpq
      if self._save_chain_state:
        # Not super efficient...
        loss = tf.cond(use_other_z_init, lambda: tf.identity(loss),
                       lambda: tf.identity(-outputs.elbo))
      recog_train_op = utils.CreateTrainOp(
          total_loss=loss,
          optimizer=opt,
          global_step=global_step,
          variables_to_train=self.RecogVars(),
          transform_grads_fn=utils.ProcessGradients)
    with tf.name_scope("gen_train"):
      gen_loss = tf.cond(global_step < self._no_gen_train_steps,
                         lambda: -outputs.elbo, lambda: -outputs.mcmc_log_p)

      gen_train_op = utils.CreateTrainOp(
          total_loss=gen_loss,
          optimizer=opt,
          global_step=None,
          variables_to_train=self.GenVars(),
          transform_grads_fn=utils.ProcessGradients)

    return tf.group(recog_train_op, gen_train_op, chain_state_update_op, step_size_op)

  def EvalOp(self, data_idx, images):
    outputs = self.MakeDLGM(images)

    tf.summary.image("data", utils.StitchImages(images[:64]))
    tf.summary.image(
        "recon_means", utils.StitchImages(outputs.recon_means[:64]))

    return utils.LogAndSummarizeMetrics({
        "elbo": outputs.elbo,
        "xentpq": outputs.xentpq,
        "mcmc_log_p": outputs.mcmc_log_p,
        "mcmc_p_accept": outputs.p_accept,
    })

  def AIS(self, images, num_chains):

    def ProposalLogProbFn(*z):
      if self._use_q_z_for_ais:
        _, log_p_z, _ = self._recog(images, z=z)
      else:
        _, log_p_z = self._prior(z)
      return tf.add_n(log_p_z)

    def TargetLogProbFn(*z):
      _, log_p_z = self._prior(z)
      _, log_p_x_z = self._noise(images, z)
      return tf.add_n(log_p_z) + log_p_x_z

    images = tf.tile(images, [num_chains, 1, 1, 1])
    if self._use_q_z_for_ais:
      z_init, _, _ = self._recog(images)
    else:
      z_init, _ = self._prior(batch=tf.shape(images)[0])

    if self._use_bijector_for_ais:
      _, _, bijector = self._recog(images)
    else:
      bijector = None
    ais_outputs = utils.AIS(ProposalLogProbFn, TargetLogProbFn, z_init, bijector=bijector)

    recons, _ = self._noise(None, ais_outputs.z_fin)

    tf.summary.image("data", utils.StitchImages(images[:64]))
    tf.summary.image("recon_means", utils.StitchImages(recons[:64]))
    tf.summary.scalar("p_accept", tf.reduce_mean(ais_outputs.p_accept))

    return AISOutputs(
        log_p=tf.reduce_logsumexp(
            tf.reshape(ais_outputs.log_p, [num_chains, -1]) - tf.log(
                tf.to_float(num_chains)), 0),
        p_accept=ais_outputs.p_accept,
        recon=recons,
        z_fin=ais_outputs.z_fin)


@gin.configurable("vae")
class VAE(object):

  def __init__(self,
               z_dims=64,
               condition_bijector=False,
               bijector_type="iaf",
               encoder_type="dense",
               beta=1.0,
               beta_steps=0,
               min_kl=0,
               use_q_z_for_ais=False,
               dataset=None,
               prior_type="simple",
               affine_rank=1):
    self.train_size = dataset.train_size
    if dataset.name == "mnist":
      output_shape = [28, 28, 1]
    elif dataset.name == "cifar10":
      output_shape = [32, 32, 3]

    self._z_dims = z_dims
    self._beta = beta
    self._use_q_z_for_ais = use_q_z_for_ais
    if beta_steps > 0:
      frac = tf.to_float(
          tf.train.get_or_create_global_step()) / tf.to_float(beta_steps)
      frac = tf.minimum(frac, 1.0)
      self._beta = frac * beta
    else:
      self._beta = tf.constant(beta)
    self._min_kl = tf.to_float(min_kl)

    if bijector_type == "affine":
      # TriL + shift
      num_outputs = (z_dims * (z_dims + 1)) // 2 + z_dims
    elif bijector_type == "affine_lr":
      num_outputs = z_dims * 2 + z_dims * affine_rank
    elif condition_bijector and bijector_type not in ["conv_shift_scale", "shift_scale"]:
      num_outputs = 3 * z_dims
    else:
      num_outputs = 2 * z_dims

    if encoder_type == "hier_conv":
      assert dataset.name == "cifar10"
      self._encoder = ConvHierEncoder("encoder")
      self._prior_posterior = ConvHierPriorPost("prior_post")
      self._decoder = lambda z: self._prior_posterior(z=z)[2]
      self._prior = lambda z=None, batch=None: self._prior_posterior(z=z, batch=batch)[:2]
      self._recog = lambda images, z=None: self._prior_posterior(images=images, z=z, encoder=self._encoder)[:2]
    else:
      if encoder_type == "dense":
        self._encoder = DenseEncoder(
            "encoder", num_outputs=num_outputs, activation=tf.nn.softplus)
        self._decoder = DenseDecoder(
            "decoder", output_shape=output_shape, activation=tf.nn.softplus)
      elif encoder_type == "conv":
        self._encoder = ConvEncoder("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder("decoder", output_shape=output_shape)
        conv_z_shape = [4, 4, self._z_dims]
      elif encoder_type == "conv2":
        self._encoder = ConvEncoder2("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder2("decoder", output_shape=output_shape)
        conv_z_shape = [4, 4, self._z_dims]
      elif encoder_type == "conv3":
        self._encoder = ConvEncoder3("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder3("decoder", output_shape=output_shape)
        conv_z_shape = [8, 8, self._z_dims]
      elif encoder_type == "conv4":
        self._encoder = ConvEncoder4("encoder", num_outputs=num_outputs)
        self._decoder = ConvDecoder4("decoder", output_shape=output_shape)
        conv_z_shape = [8, 8, self._z_dims]

      if prior_type == "simple":
        self._prior = SimplePrior("prior", num_dims=self._z_dims)
      elif prior_type == "simple_3d":
        self._prior = Simple3DPrior("prior", shape=conv_z_shape)
      if bijector_type == "iaf":
        recog = DenseRecognitionIAF(
            "recog", encoder=self._encoder, condition_iaf=condition_bijector)
      elif bijector_type == "rnvp":
        recog = DenseRecognitionRNVP(
            "recog",
            encoder=self._encoder,
            condition_bijector=condition_bijector)
      elif bijector_type == "shift_scale":
        recog = DenseRecognition("recog", encoder=self._encoder)
      elif bijector_type == "conv_shift_scale":
        recog = ConvShiftScale("recog", encoder=self._encoder)
      elif bijector_type == "affine":
        recog = DenseRecognitionAffine("recog", encoder=self._encoder, z_dims=z_dims)
      elif bijector_type == "conv_iaf":
        recog = ConvIAF("recog", encoder=self._encoder, condition_iaf=condition_bijector)
      elif bijector_type == "affine_lr":
        recog = DenseRecognitionAffineLR("recog", encoder=self._encoder, z_dims=z_dims, rank=affine_rank)

      # Drop the bijector return.
      self._recog = lambda *args, **kwargs: recog(*args, **kwargs)[:2]

    if dataset.name == "mnist":
      self._noise = DenseMNISTNoise("noise", decoder=self._decoder)
    else:
      self._noise = DenseCIFAR10TNoise("noise", decoder=self._decoder)

  def MakeVAE(self, images, beta_override=None, num_samples=64):
    if beta_override is not None:
      beta = beta_override
    else:
      beta = self._beta
    return MakeVAE(images, self._recog, self._prior, self._noise, beta,
                   num_samples, self._min_kl)

  def TrainOp(self, data_idx, images):
    outputs = self.MakeVAE(images)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = LearningRate(self.train_size, global_step)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    metrics = {
        "learning_rate": learning_rate,
        "log_p_x_z": outputs.log_p_x_z,
        "elbo": outputs.elbo,
        "klqp": outputs.total_klqp,
        "beta": self._beta,
    }
    for i, layer_klqp in enumerate(outputs.klqp):
      metrics["klqp_%d"%i] = layer_klqp
    utils.LogAndSummarizeMetrics(metrics, False)

    tf.summary.image(
        "sample_means", utils.StitchImages(outputs.sample_means))

    return utils.CreateTrainOp(
        total_loss=-outputs.elbo,
        optimizer=opt,
        global_step=global_step,
        variables_to_train=tf.trainable_variables(),
        transform_grads_fn=utils.ProcessGradients)

  def GetPosterior(self, images):
    outputs = self.MakeVAE(images)
    return outputs.post_z

  def EvalOp(self, data_idx, images):
    outputs = self.MakeVAE(images, 1.0)

    tf.summary.image("data", utils.StitchImages(images[:64]))
    tf.summary.image(
        "recon_means", utils.StitchImages(outputs.recon_means[:64]))
    metrics = {
        "elbo": outputs.elbo,
        "klqp": outputs.total_klqp,
    }
    for i, layer_klqp in enumerate(outputs.klqp):
      metrics["klqp_%d"%i] = layer_klqp

    return utils.LogAndSummarizeMetrics(metrics)

  def AIS(self, images, num_chains):
    outputs = self.MakeVAE(images)

    def ProposalLogProbFn(*z):
      if self._use_q_z_for_ais:
        _, log_p_z = self._recog(images, z=z)
      else:
        _, log_p_z = self._prior(z)
      return tf.add_n(log_p_z)

    def TargetLogProbFn(*z):
      _, log_p_z = self._prior(z)
      _, log_p_x_z = self._noise(images, z)
      return tf.add_n(log_p_z) + log_p_x_z

    images = tf.tile(images, [num_chains, 1, 1, 1])
    if self._use_q_z_for_ais:
      z_init, _ = self._recog(images)
    else:
      z_init, _ = self._prior(batch=tf.shape(images)[0])

    ais_outputs = utils.AIS(ProposalLogProbFn, TargetLogProbFn, z_init)

    recons, _ = self._noise(None, ais_outputs.z_fin)

    tf.summary.image("data", utils.StitchImages(images[:64]))
    tf.summary.image("recon_means", utils.StitchImages(recons[:64]))
    tf.summary.scalar("p_accept", tf.reduce_mean(ais_outputs.p_accept))

    return AISOutputs(
        log_p=tf.reduce_logsumexp(
            tf.reshape(ais_outputs.log_p, [num_chains, -1]) - tf.log(
                tf.to_float(num_chains)), 0),
        p_accept=ais_outputs.p_accept,
        recon=recons,
        z_fin=ais_outputs.z_fin)


@gin.configurable("train")
def Train(model, dataset, train_dir, master, epochs=600, polyak_averaging=0.0, warmstart_ckpt=""):
  data_idx, images = dataset.TrainBatch(TRAIN_BATCH, epochs)

  train_op = model.TrainOp(data_idx, images)
  if polyak_averaging > 0.0:
    tf.logging.info("Using polyak averaging")
    ema = tf.train.ExponentialMovingAverage(decay=polyak_averaging)
    with tf.control_dependencies([train_op]):
      train_op = ema.apply()

  utils.LogAndSaveHParams()

  tf.Session.reset(master)

  if warmstart_ckpt:
    tf.init_from_checkpoint(warmstart_ckpt, {"/": "/"})

  hooks = [
      tf.train.StopAtStepHook(last_step=dataset.train_size * epochs //
                              TRAIN_BATCH),
      tf.train.LoggingTensorHook(utils.GetLoggingOutputs(), every_n_secs=60)
  ]
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=True,
      checkpoint_dir=train_dir,
      hooks=hooks,
      save_checkpoint_secs=120,
      save_summaries_steps=60,
      max_wait_secs=7200) as sess:

    while not sess.should_stop():
      sess.run(train_op)


def Eval(model, dataset, train_dir, eval_dir, master,
         use_polyak_averaging=False, max_number_of_evaluations=None):
  data_idx, images = dataset.TestBatch(TEST_BATCH)

  eval_op = model.EvalOp(data_idx, images)
  utils.LogAndSaveHParams()
  tf.train.get_or_create_global_step()

  if use_polyak_averaging:
    tf.logging.info("Using polyak averaging")
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    saver = tf.train.Saver(ema.variables_to_restore())
  else:
    saver = tf.train.Saver()

  scaffold = tf.train.Scaffold(saver=saver)

  tf.Session.reset(master)

  utils.evaluate_repeatedly(
      train_dir,
      eval_dir,
      eval_ops=eval_op,
      stop_after_n_evals=dataset.test_size // TEST_BATCH,
      # This is widely optimistic.
      eval_interval_secs=120,
      max_number_of_evaluations=max_number_of_evaluations,
      master=master,
      scaffold=scaffold)


def AISEvalShard(shard, master, num_workers, num_chains, dataset, use_polyak_averaging, writer, train_dir, model_fn, batch):
  tf.logging.info("Thread started")
  model = model_fn()
  tf.logging.info("Built model")

  shard_idx = tf.placeholder(tf.int64, [])
  tf.logging.info("built data")
  data_iterator = dataset.AISIterator(batch, shard_idx, num_workers)
  images, _ = data_iterator.get_next()
  tf.logging.info("Built mA")
  ais_outputs = model.AIS(images, num_chains)
  log_p = ais_outputs.log_p
  p_accept = ais_outputs.p_accept
  tf.logging.info("Built mB")
  if shard == 1:
    utils.LogAndSaveHParams()

  summary_op = tf.summary.merge_all()
  global_step = tf.train.get_or_create_global_step()
  if use_polyak_averaging:
    tf.logging.info("Using polyak averaging")
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    saver = tf.train.Saver(ema.variables_to_restore())
  else:
    saver = tf.train.Saver()

  tf.logging.info("Built mC")

  global_step_val = []
  tf.logging.info("Starting shard %d, %s", shard, master)
  #with tf.MonitoredSession(
  #    tf.train.ChiefSessionCreator(
  #        master=master,
  #        checkpoint_dir=train_dir)) as sess:
  while True:
    try:
      tf.Session.reset(master)
      with tf.Session(master) as sess:
        all_log_p = np.zeros([0])
        saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        sess.run(data_iterator.initializer, {shard_idx: shard})
        try:
          step_num = 0
          while True:
            fetch = {
                "log_p": log_p,
                "global_step": global_step,
                "p_accept": p_accept
            }
            if shard == 0:
              fetch["summary"] = summary_op
            tf.logging.info("Shard %d step %d started.", shard, step_num)
            fetch = sess.run(fetch)
            tf.logging.info("Shard %d step %d done.", shard, step_num)
            tf.logging.info("Shard %d log_p %.2f, p_accept: %.2f", shard,
                            np.mean(fetch["log_p"]),
                            np.mean(fetch["p_accept"]))
            all_log_p = np.hstack([all_log_p, fetch["log_p"]])
            if shard == 0 and step_num == 0:
              global_step_val.append(fetch["global_step"])
              writer.add_summary(fetch["summary"], global_step_val[0])
            step_num += 1
        except tf.errors.OutOfRangeError:
          tf.logging.info("Shard %d done.", shard)
          pass
      return all_log_p
    except tf.errors.AbortedError:
      pass


def AISEval(model_fn, dataset, train_dir, eval_dir, worker_master_pattern,
            num_workers, num_chains, use_polyak_averaging=False):
  tf.reset_default_graph()
  log_p_ph = tf.placeholder(tf.float32, [None])
  log_p_summary = tf.summary.scalar("log_p", tf.reduce_mean(log_p_ph))

  writer = tf.summary.FileWriter(eval_dir)

  with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = []
    for shard in range(num_workers):
      tf.logging.info("Submitting shard %d", shard)
      master = worker_master_pattern.format(shard)
      results.append(
          executor.submit(AISEvalShard, shard, master, num_workers, num_chains,
                          dataset, use_polyak_averaging, writer, train_dir,
                          model_fn, AIS_BATCH))

    all_log_p = np.zeros([0])
    for result in results:
      log_p = result.result()
      all_log_p = np.hstack([all_log_p, log_p])

  log_p = np.mean(all_log_p)
  tf.logging.info("Log P: %.2f", log_p)
  with tf.Session() as sess:
    writer.add_summary(
        sess.run(log_p_summary, {log_p_ph: all_log_p}), 0)
  writer.flush()
  return log_p


MODEL_TO_CLASS = {"vae": VAE, "dlgm": DLGM}


def main(argv):
  del argv  # Unused.

  utils.BindHParams(FLAGS.hparams)

  if FLAGS.data_type == "mnist":
    dataset = utils.MNISTDataset(FLAGS.mnist_data_dir, FLAGS.test_is_valid)
  elif FLAGS.data_type == "fashion_mnist":
    dataset = utils.MNISTDataset(FLAGS.fashion_mnist_data_dir, FLAGS.test_is_valid)
  elif FLAGS.data_type == "cifar10":
    dataset = utils.CIFAR10Dataset(FLAGS.cifar10_data_dir, FLAGS.test_is_valid)
  elif FLAGS.data_type == "fake":
    dataset = utils.FakeMNISTDataset()

  if FLAGS.mode == "train":
    model = MODEL_TO_CLASS[FLAGS.model](dataset=dataset)
    Train(model, dataset, FLAGS.train_dir, FLAGS.master,
          polyak_averaging=FLAGS.polyak_averaging)
  elif FLAGS.mode == "eval":
    model = MODEL_TO_CLASS[FLAGS.model](dataset=dataset)
    Eval(model, dataset, FLAGS.train_dir, FLAGS.eval_dir,
         FLAGS.master,
         use_polyak_averaging=FLAGS.polyak_averaging > 0.0)
  elif FLAGS.mode == "ais_eval":
    replica_log_p = []
    if FLAGS.ais_replicas:
      replicas = FLAGS.ais_replicas
    else:
      replicas = list(range(FLAGS.ais_num_replicas))
    for i in replicas:
      train_dir = FLAGS.train_dir.format(i)
      eval_dir = FLAGS.eval_dir.format(i)
      model_fn = lambda: MODEL_TO_CLASS[FLAGS.model](dataset=dataset)
      log_p = AISEval(model_fn, dataset, train_dir, eval_dir,
                      FLAGS.ais_worker_pattern, FLAGS.ais_num_workers,
                      FLAGS.ais_num_chains,
                      use_polyak_averaging=FLAGS.polyak_averaging > 0.0)
      replica_log_p.append(log_p)

    log_p = np.mean(replica_log_p)
    std_log_p = np.std(replica_log_p)
    tf.logging.info("Log P: %.2f +- %.2f", log_p,
                    std_log_p / np.sqrt(len(replicas)))
    tf.logging.info("All log_p: %s", replica_log_p)
  elif FLAGS.mode == "ais_eval2":
    if FLAGS.ais_replicas:
      replicas = FLAGS.ais_replicas
    else:
      replicas = list(range(FLAGS.ais_num_replicas))
    for i in replicas:
      tf.reset_default_graph()
      train_dir = FLAGS.train_dir.format(i)
      eval_dir = FLAGS.eval_dir.format(i)
      model_fn = lambda: MODEL_TO_CLASS[FLAGS.model](dataset=dataset)

      sentinel_filename = os.path.join(eval_dir, "ais_shard_%d_done" % FLAGS.ais_shard)
      if tf.gfile.Exists(sentinel_filename):
        continue

      batch = FLAGS.ais_batch_size
      assert (dataset.test_size // FLAGS.ais_num_workers) % batch == 0
      writer = tf.summary.FileWriter(eval_dir)
      log_p = AISEvalShard(FLAGS.ais_shard, "", FLAGS.ais_num_workers, FLAGS.ais_num_chains,
                           dataset, FLAGS.polyak_averaging > 0.0, writer, train_dir, model_fn, batch)
      tf.gfile.MakeDirs(eval_dir)
      with tf.gfile.Open(os.path.join(eval_dir, "ais_shard_%d" % FLAGS.ais_shard), "w") as f:
        np.savetxt(f, log_p)
      with tf.gfile.Open(sentinel_filename, "w") as f:
        f.write("done")


if __name__ == "__main__":
  flags.DEFINE_string("mnist_data_dir", "", "")
  flags.DEFINE_string("fashion_mnist_data_dir", "", "")
  flags.DEFINE_string("cifar10_data_dir", "", "")
  flags.DEFINE_string("data_type", "mnist", "")
  flags.DEFINE_enum("mode", "train", ["train", "eval", "ais_eval", "ais_eval2"], "")
  flags.DEFINE_enum("model", "vae", list(MODEL_TO_CLASS.keys()), "")
  flags.DEFINE_string("train_dir", "/tmp/vae/train", "")
  flags.DEFINE_string("eval_dir", "/tmp/vae/eval", "")
  flags.DEFINE_string("master", "", "")
  flags.DEFINE_string("ais_worker_pattern", "", "")
  flags.DEFINE_integer("ais_shard", 0, "")
  flags.DEFINE_integer("ais_num_workers", 1, "")
  flags.DEFINE_integer("ais_num_chains", 1, "")
  flags.DEFINE_integer("ais_num_replicas", 1, "")
  flags.DEFINE_list("ais_replicas", "", "Manual listing of replicas")
  flags.DEFINE_integer("ais_batch_size", 25, "")
  flags.DEFINE_float("polyak_averaging", 0.0, "")
  flags.DEFINE_boolean("test_is_valid", False, "")
  flags.DEFINE(utils.YAMLDictParser(), "hparams", "", "")

  app.run(main)
