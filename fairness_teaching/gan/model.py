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

import os
import numpy as np
import tensorflow as tf
# pylint: skip-file

def get_weight(shape, stddev, reg, name):
  wd = 0.0
  # init = tf.random_normal_initializer(stddev=stddev)
  init = tf.contrib.layers.xavier_initializer()
  if reg:
    regu = tf.contrib.layers.l2_regularizer(wd)
    filt = tf.get_variable(name, shape, initializer=init, regularizer=regu)
  else:
    filt = tf.get_variable(name, shape, initializer=init)
  return filt

def get_bias(shape, init_bias, reg, name):
  wd = 0.0
  init = tf.constant_initializer(init_bias)
  if reg:
    regu = tf.contrib.layers.l2_regularizer(wd)
    bias = tf.get_variable(name, shape, initializer=init, regularizer=regu)
  else:
    bias = tf.get_variable(name, shape, initializer=init)
  return bias

def batch_norm(x, phase_train, moments_dim):
  """
  Batch normalization on convolutional maps.
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  Args:
    x:           Tensor, 4D BHWD input maps
    phase_train: boolean tf.Varialbe, true indicates training phase
    scope:       string, variable scope
  Return:
    normed:      batch-normalized maps
  """
  with tf.variable_scope('bn'):
    n_out = x.get_shape().as_list()[-1]
    gamma = get_bias(n_out, 1.0, True, 'gamma')
    beta = get_bias(n_out, 0.0, True, 'beta')
    batch_mean, batch_var = tf.nn.moments(x, moments_dim, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.999)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
              mean_var_with_update,
              lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed

def max_pool(inputs, name, k_shape=[1, 2, 2, 1],s_shape=[1, 2, 2, 1]):
  with tf.variable_scope(name) as scope:
    outputs = tf.nn.max_pool(inputs, ksize=k_shape, strides=s_shape, padding='SAME', name=name)
  return outputs

def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
  dim_size *= stride_size
  if padding == 'VALID' and dim_size is not None:
    dim_size += max(kernel_size - stride_size, 0)
  return dim_size

def fc(inputs, n_output, is_training, name, bias=0.0, relu=True, reg=True, bn=True):
  with tf.variable_scope(name) as scope:
    n_input = inputs.get_shape().as_list()[-1]
    shape = [n_input, n_output]
    # print("shape of filter %s: %s" % (name, str(shape)))
    filt = get_weight(shape, stddev=tf.sqrt(2.0/tf.to_float(n_input+n_output)), reg=True, name='weight')
    bias = get_bias([n_output],init_bias=bias, reg=True, name='bias')
    outputs = tf.matmul(inputs, filt)
    outputs = tf.nn.bias_add(outputs, bias)
    if bn:
      outputs = batch_norm(outputs, is_training, [0,])
    if relu:
      outputs = tf.nn.leaky_relu(outputs)
  return outputs

def conv_2d(inputs, ksize, n_output, is_training, name, stride=1, pad='SAME', relu=True, reg=True, bn=True):
  with tf.variable_scope(name) as scope:
    n_input = inputs.get_shape().as_list()[3]
    shape = [ksize, ksize, n_input, n_output]
    # print("shape of filter %s: %s\n" % (name, str(shape)))
    filt = get_weight(shape, stddev=tf.sqrt(2.0/tf.to_float(n_input+n_output)), reg=reg, name='weight')
    outputs = tf.nn.conv2d(inputs, filt, [1, stride, stride, 1], padding=pad)
    if bn:
      outputs = batch_norm(outputs, is_training, [0,1,2])
    if relu:
      outputs = tf.nn.leaky_relu(outputs)
  return outputs

def conv_2d_trans(inputs, ksize, n_output, is_training, name, stride=1, pad='SAME', relu=True, reg=True, bn=True):
  with tf.variable_scope(name) as scope:
    batch_size = tf.shape(inputs)[0]
    input_size = inputs.get_shape().as_list()[1]
    n_input = inputs.get_shape().as_list()[3]
    shape = [ksize, ksize, n_output, n_input]
    output_shape = tf.stack([batch_size, input_size*stride, input_size*stride, n_output])
    # print("shape of deconv_filter %s: %s\n" % (name, str(shape)))
    filt = get_weight(shape, stddev=tf.sqrt(2.0/tf.to_float(n_input+n_output)), reg=reg, name='weight')
    outputs = tf.nn.conv2d_transpose(inputs, filt, output_shape, [1, stride, stride, 1], padding=pad)
    if bn:
      outputs = batch_norm(outputs, is_training, [0,1,2])
    if relu:
      outputs = tf.nn.relu(outputs)
  return outputs

class Adv_cls():
  def build(self, inputs, n_class, is_training):
    with tf.variable_scope('Adv', reuse=tf.AUTO_REUSE):
      net = inputs[-1]

      for i in range(3): #4x4
        net = conv_2d(net, 3, 512, is_training, 'conv1_'+str(i))
      # net = max_pool(net, 'pool3')

      for i in range(3): #4x4
        net = conv_2d(net, 3, 256, is_training, 'conv2_'+str(i))
      # net = max_pool(net, 'pool3')

      net = conv_2d(net, 4, 256, is_training, 'fc1', pad='VALID')

      net = conv_2d(net, 1, 128, is_training, 'fc2', pad='VALID')

      net = tf.squeeze(conv_2d(net, 1, n_class, is_training, 'fc3', pad='VALID', relu=False, bn=False))

      self.vars = tf.trainable_variables('Adv')
      self.reg_loss = tf.losses.get_regularization_losses('Adv')

      return net

class Genc():
  def build(self, inputs, is_training):
    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
      net = inputs
      nets = []
      for i in range(5):
        net = conv_2d(net, 4, int(64 * 2**i), is_training, 'enc_'+str(i), stride=2)
        nets.append(net)

      self.vars = tf.trainable_variables('Genc')
      self.reg_loss = tf.losses.get_regularization_losses('Genc')
      return nets

class Gdec():
  def build(self, inputs, labels, is_training):
    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
      labels = tf.reshape(tf.to_float(labels),[-1,1,1,1]) # B,1,1,N
      net = inputs[-1]
      tile_labels = tf.tile(labels,[1,net.shape[1],net.shape[2],1])
      net = tf.concat([net, tile_labels],axis=-1)

      for i in range(4):
        if i==1:
          net = tf.concat([net, inputs[-2]],axis=-1)
          tile_labels = tf.tile(labels,[1,net.shape[1],net.shape[2],1])
          net = tf.concat([net, tile_labels],axis=-1)
        net = conv_2d_trans(net, 4, int(1024 / 2**i), is_training, 'dec_'+str(i), stride=2)

      net = tf.nn.tanh(conv_2d_trans(net, 4, 3, is_training, 'dec_f', stride=2, relu=False, bn=False))

      self.vars = tf.trainable_variables('Gdec')
      self.reg_loss = tf.losses.get_regularization_losses('Gdec')
      return net

class D():
  def build(self, inputs, is_training):
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
      net = inputs
      batch_size = net.get_shape().as_list()[0]
      for i in range(5):
        net = conv_2d(net, 4, int(64 * 2**i), is_training, 'D_'+str(i), stride=2)
      net = tf.reshape(net, [batch_size, -1])

      gan_net = fc(net, 1024, is_training, 'gan1')
      gan_net = fc(gan_net, 1, is_training, 'gan2', relu=False, bn=False)

      cls_net = fc(net, 1024, is_training, 'cls1')
      cls_net = fc(cls_net, 1, is_training, 'cls2', relu=False, bn=False)

      self.vars = tf.trainable_variables('D')
      self.reg_loss = tf.losses.get_regularization_losses('D')
      return gan_net, cls_net




