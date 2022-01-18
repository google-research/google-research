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

import os
import numpy as np
import tensorflow as tf
# pylint: skip-file

def get_weight(shape, stddev, reg, name):
  wd = 5e-4
  init = tf.random_normal_initializer(stddev=stddev)
  if reg:
    regu = tf.contrib.layers.l2_regularizer(wd)
    filt = tf.get_variable(name, shape, initializer=init, regularizer=regu)
  else:
    filt = tf.get_variable(name, shape, initializer=init)
  return filt

def get_bias(shape, init_bias, reg, name):
  wd = 5e-4
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
    # ema = tf.train.ExponentialMovingAverage(decay=0.999, zero_debias=True)
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

def fc(inputs, n_output, is_training, name, relu=True, reg=True, bn=True):
  with tf.variable_scope(name) as scope:
    n_input = inputs.get_shape().as_list()[-1]
    shape = [n_input, n_output]
    # print("shape of filter %s: %s" % (name, str(shape)))
    filt = get_weight(shape, stddev=0.1, reg=True, name='weight')
    bias = get_bias([n_output],init_bias=0.0, reg=True, name='bias')
    outputs = tf.matmul(inputs, filt)
    outputs = tf.nn.bias_add(outputs, bias)
    if bn:
      outputs = batch_norm(outputs, is_training, [0,])
    if relu:
      outputs = tf.nn.relu(outputs)
      # outputs = tf.nn.tanh(outputs)
  return outputs

def conv_2d(inputs, ksize, n_output, is_training, name, stride=1, pad='SAME', relu=True, reg=True, bn=True):
  with tf.variable_scope(name) as scope:
    n_input = inputs.get_shape().as_list()[3]
    shape = [ksize, ksize, n_input, n_output]
    # print("shape of filter %s: %s" % (name, str(shape)))
    filt = get_weight(shape, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)), reg=True, name='weight')
    outputs = tf.nn.conv2d(inputs, filt, [1, stride, stride, 1], padding=pad)
    if bn:
      outputs = batch_norm(outputs, is_training, [0,1,2])
    if relu:
      outputs = tf.nn.relu(outputs)
  return outputs

class VGG():
  def build(self, inputs, n_class, is_training):
    # with tf.variable_scope('VGG', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('VGG'):
      net = inputs

      for i in range(2): #128x128
        net = conv_2d(net, 3, 64, is_training, 'conv1_'+str(i))
      net = max_pool(net, 'pool1')

      for i in range(2): #64x64
        net = conv_2d(net, 3, 128, is_training, 'conv2_'+str(i))
      net = max_pool(net, 'pool2')

      for i in range(3): #32x32
        net = conv_2d(net, 3, 256, is_training, 'conv3_'+str(i))
      net = max_pool(net, 'pool3')

      for i in range(3): #16x16
        net = conv_2d(net, 3, 512, is_training, 'conv4_'+str(i))
      net = max_pool(net, 'pool4')

      for i in range(3): #8x8
        net = conv_2d(net, 3, 512, is_training, 'conv5_'+str(i))
      net = max_pool(net, 'pool5')

      net = conv_2d(net, 4, 256, is_training, 'fc1', pad='VALID')

      net = conv_2d(net, 1, 128, is_training, 'fc2', pad='VALID')

      net = tf.squeeze(conv_2d(net, 1, n_class, is_training, 'fc3', pad='VALID', relu=False, bn=False))

      self.vars = tf.trainable_variables('VGG')
      self.reg_loss = tf.losses.get_regularization_losses('VGG')

      return net

class Actor():
  def build(self, inputs, n_action, is_training):
    # with tf.variable_scope('VGG', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('Actor'):
      net = inputs

      net = fc(net, 15, is_training, 'fc1')

      net = fc(net, n_action, is_training, 'fc2', relu=False, bn=False)

      self.vars = tf.trainable_variables('Actor')
      self.reg_loss = tf.losses.get_regularization_losses('Actor')

      return net
