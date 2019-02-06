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

"""Network for Cifar10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

# TO BE CHANGED, baseroute folder

PATH_MODEL = 'baseroute/models/cifar100/model'
NUM_CLASSES = 100
NUM_TRAIN = 50000
NUM_EVAL = 10000


def variable_on_cpu(name, shape):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, dtype=tf.float32)
  return var


class NetworkCifar(object):
  """Network for CIFAR10 dataset, computing the features."""

  def __init__(self):
    """Initializes the class."""
    self.build_graph()

  def build_graph(self):
    """Builds the neural network graph."""

    # define graph
    self.g = tf.Graph()
    with self.g.as_default():
      self.sess = tf.Session()
      self.x = tf.placeholder(shape=[None, 24, 24, 3], dtype=tf.float32)
      self.y = tf.placeholder(shape=[None, NUM_CLASSES], dtype=tf.float32)

      # conv1
      with tf.variable_scope('conv1') as scope:
        kernel = variable_on_cpu('weights', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(self.x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

      # pool1
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
      # norm1
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

      # conv2
      with tf.variable_scope('conv2') as scope:
        kernel = variable_on_cpu('weights', shape=[5, 5, 64, 64])
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

      # norm2
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
      # pool2
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      # local3
      with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # images.get_shape().as_list()[0] = batchsize
        # reshape = tf.keras.layers.Flatten()(pool2)
        reshape = tf.reshape(pool2, [tf.shape(self.x)[0], 6*6*64])
        dim = reshape.get_shape()[1].value
        weights = variable_on_cpu('weights', shape=[dim, 384])
        biases = variable_on_cpu('biases', [384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)

      # local4
      with tf.variable_scope('local4') as scope:
        weights = variable_on_cpu('weights', shape=[384, 192])
        biases = variable_on_cpu('biases', [192])
        self.local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                                 name=scope.name)

      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits
      # accepts the unscaled logits
      # and performs the softmax internally for efficiency.
      with tf.variable_scope('softmax_linear') as scope:
        weights = variable_on_cpu('weights', [192, NUM_CLASSES])
        biases = variable_on_cpu('biases', [NUM_CLASSES])
        self.softmax_linear = tf.add(tf.matmul(self.local4, weights),
                                     biases, name=scope.name)

      print('loading the network ...')
      saver_network = tf.train.Saver()
      # Restores from checkpoint
      saver_network.restore(
          self.sess, os.path.join(PATH_MODEL, 'model.ckpt-100000'))
      print('Graph successfully loaded.')

  def features(self, x):
    """Computes the features of the last but one layer."""
    features = []
    nb_chunks = 100
    q = x.shape[0] // nb_chunks
    r = x.shape[0] % nb_chunks
    for i in np.arange(nb_chunks):
      batch_x = x[i*q:(i+1)*q, :]
      feed_dict = {self.x: batch_x}
      features_v = self.sess.run(self.local4, feed_dict=feed_dict)
      features.append(features_v)
    if r > 0:
      batch_x = x[(100*q):, :]
      feed_dict = {self.x: batch_x}
      features_v = self.sess.run(self.local4, feed_dict=feed_dict)
      features.append(features_v)
    features = np.vstack(tuple(features))
    return features
