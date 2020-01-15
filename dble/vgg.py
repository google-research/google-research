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

"""Contains definitions for VGG Networks.
"""
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

configurations = {
    'VGG11': [
        64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ],
    'VGG13': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ],
    'VGG16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'VGG19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ]
}


class VggNet(object):
  """Definition of VGG Networks."""

  def __init__(self, vggname, neck, keep_prob, wd, feature_dim, num_classes=10):
    """Creates a model for classifying an image using VGG networks.

    Args:
      vggname: A string representing the vgg type, such as 'VGG11'.
      neck: A bool value that decides using the MLP neck or not.
      keep_prob: The rate of keeping one neuron in Dropout.
      wd: The co-efficient of weight decay.
      feature_dim: the dimension of the representation space.
      num_classes: The number of classes for classification.
    """
    super(VggNet, self).__init__()
    self.vggname = vggname
    self.num_classes = num_classes

    self.regularizer = contrib_layers.l2_regularizer(scale=wd)
    self.initializer = contrib_layers.xavier_initializer()
    self.variance_initializer = contrib_layers.variance_scaling_initializer(
        factor=0.1,
        mode='FAN_IN',
        uniform=False,
        seed=None,
        dtype=tf.dtypes.float32)

    self.pool_num = 0
    self.conv_num = 0

    self.drop_rate = 1 - keep_prob
    self.neck = neck
    self.feature_dim = feature_dim

  def encoder(self, inputs, training):
    """Forwards a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor. If self.neck is true, the logits Tensor is with shape
      [<batch_size>, self.num_classes]. If self.neck is not true, the logits
      Tensor is with shape [<batch_size>, 512].
    """
    out = self.make_layer(inputs, configurations[self.vggname], training)
    out = tf.layers.flatten(out, name='flatten')
    if self.neck:
      out = tf.layers.dropout(out, rate=self.drop_rate, training=training)
      out = tf.layers.dense(
          out,
          units=10,
          kernel_initializer=self.initializer,
          kernel_regularizer=self.regularizer,
          name='fc_3')
    self.conv_num = 0
    return out

  def confidence_model(self, mu, training):
    """Given a batch of mu, output a batch of variance."""
    out = tf.layers.dropout(mu, rate=self.drop_rate, training=training)
    out = tf.layers.dense(
        out,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc_variance')
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out, rate=self.drop_rate, training=training)
    out = tf.layers.dense(
        out,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc_variance2')
    return out

  def conv2d(self, inputs, out_channel, training):
    """The convolution module, including conv2d, BatchNorm and Activation."""
    inputs = tf.layers.conv2d(
        inputs,
        filters=out_channel,
        kernel_size=3,
        padding='same',
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='conv_' + str(self.conv_num))
    inputs = tf.layers.batch_normalization(
        inputs, training=training, name='bn_' + str(self.conv_num))
    self.conv_num += 1
    return tf.nn.relu(inputs)

  def make_layer(self, inputs, netparam, training):
    """The forward pass before the MLP module given inputs."""
    for param in netparam:
      if param == 'M':
        inputs = tf.layers.max_pooling2d(
            inputs,
            pool_size=2,
            strides=2,
            padding='same',
            name='pool_' + str(self.pool_num))
        self.pool_num += 1
      elif param == 'D':
        inputs = tf.layers.dropout(
            inputs, rate=self.drop_rate, training=training)
      else:
        inputs = self.conv2d(inputs, param, training)
    inputs = tf.layers.average_pooling2d(inputs, pool_size=1, strides=1)
    return inputs


def vgg11(keep_prob, wd, neck, feature_dim):
  net = VggNet(vggname='VGG11', keep_prob=keep_prob, wd=wd, neck=neck,
               feature_dim=feature_dim)
  return net


def vgg13(keep_prob, wd, neck, feature_dim):
  net = VggNet(vggname='VGG13', keep_prob=keep_prob, wd=wd, neck=neck,
               feature_dim=feature_dim)
  return net


def vgg16(keep_prob, wd, neck, feature_dim):
  net = VggNet(vggname='VGG16', keep_prob=keep_prob, wd=wd, neck=neck,
               feature_dim=feature_dim)
  return net


def vgg19(keep_prob, wd, neck, feature_dim):
  net = VggNet(vggname='VGG19', keep_prob=keep_prob, wd=wd, neck=neck,
               feature_dim=feature_dim)
  return net
