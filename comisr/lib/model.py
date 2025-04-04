# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Model functions to reconstruct models."""

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from lib import ops


# Definition of the fnet, more details can be found in TecoGAN paper
def fnet(fnet_input, reuse=False):
  """Flow net."""
  def down_block(inputs, output_channel=64, stride=1, scope='down_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = ops.lrelu(net, 0.2)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = ops.lrelu(net, 0.2)
      net = ops.maxpool(net)

    return net

  def up_block(inputs, output_channel=64, stride=1, scope='up_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = ops.lrelu(net, 0.2)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = ops.lrelu(net, 0.2)
      new_shape = tf.shape(net)[1:-1] * 2
      net = tf2.image.resize(net, new_shape)

    return net

  with tf.variable_scope('autoencode_unit', reuse=reuse):
    net = down_block(fnet_input, 32, scope='encoder_1')
    net = down_block(net, 64, scope='encoder_2')
    net = down_block(net, 128, scope='encoder_3')

    net = up_block(net, 256, scope='decoder_1')
    net = up_block(net, 128, scope='decoder_2')
    net1 = up_block(net, 64, scope='decoder_3')

    with tf.variable_scope('output_stage'):
      net = ops.conv2(net1, 3, 32, 1, scope='conv1')
      net = ops.lrelu(net, 0.2)
      net2 = ops.conv2(net, 3, 2, 1, scope='conv2')
      net = tf.tanh(net2) * 24.0
      # the 24.0 is the max Velocity, details can be found in TecoGAN paper
  return net


def generator_f_encoder(gen_inputs, num_resblock=10, reuse=False):
  """Generator function encoder."""
  # The Bx residual blocks
  def residual_block(inputs, output_channel=64, stride=1, scope='res_block'):
    with tf.variable_scope(scope):
      net = ops.conv2(
          inputs, 3, output_channel, stride, use_bias=True, scope='conv_1')
      net = tf.nn.relu(net)
      net = ops.conv2(
          net, 3, output_channel, stride, use_bias=True, scope='conv_2')
      net = net + inputs

    return net

  with tf.variable_scope('generator_unit', reuse=reuse):
    # The input layer
    with tf.variable_scope('input_stage'):
      net = ops.conv2(gen_inputs, 3, 64, 1, scope='conv')
      stage1_output = tf.nn.relu(net)

    net = stage1_output

    # The residual block parts
    for i in range(1, num_resblock + 1,
                   1):  # should be 16 for TecoGAN, and 10 for TecoGANmini
      name_scope = 'resblock_%d' % (i)
      net = residual_block(net, 64, 1, name_scope)

  return net


def generator_f_decoder(net,
                        gen_inputs,
                        gen_output_channels,
                        vsr_scale,
                        reuse=False):
  """Generator function decoder."""
  with tf.variable_scope('generator_unit', reuse=reuse):
    with tf.variable_scope('conv_tran2highres'):
      if vsr_scale == 2:
        net = ops.conv2_tran(
            net, kernel=3, output_channel=64, stride=2, scope='conv_tran1')
        net = tf.nn.relu(net)
      if vsr_scale == 4:
        net = ops.conv2_tran(net, 3, 64, 2, scope='conv_tran1')
        net = tf.nn.relu(net)
        net = ops.conv2_tran(net, 3, 64, 2, scope='conv_tran2')
        net = tf.nn.relu(net)

    with tf.variable_scope('output_stage'):
      net = ops.conv2(net, 3, gen_output_channels, 1, scope='conv')
      low_res_in = gen_inputs[:, :, :, 0:3]  # ignore warped pre high res
      bicubic_hi = ops.bicubic_x(low_res_in, scale=vsr_scale)  # can put on GPU
      net = net + bicubic_hi
      net = ops.preprocess(net)
    return net


# Definition of the generator.
def generator_f(gen_inputs,
                gen_output_channels,
                num_resblock=10,
                vsr_scale=4,
                reuse=False):
  net = generator_f_encoder(gen_inputs, num_resblock, reuse)
  net = generator_f_decoder(net, gen_inputs, gen_output_channels, vsr_scale,
                            reuse)

  return net
