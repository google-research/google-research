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

"""Model function based on the ResNet architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data
import numpy as np
from options import FLAGS
import tensorflow as tf


# Building blocks of the network
def classify(activations, reuse):
  """Classify block."""
  with tf.variable_scope("Classify", reuse=reuse):
    logits = tf.layers.dense(
        activations, units=input_data.NUM_CLASSES, use_bias=False)
    predictions = tf.nn.softmax(logits)
    return logits, predictions


def relational_attention(encoded_queries,
                         candidate_keys,
                         candidate_values,
                         normalization="softmax"):
  """Block for dot-product based relational attention."""

  activations = tf.matmul(candidate_keys, encoded_queries, transpose_b=True)
  activations /= np.sqrt(FLAGS.attention_dim)
  activations = tf.transpose(activations, [1, 0])
  if normalization == "softmax":
    weight_coefs = tf.nn.softmax(activations)
  elif normalization == "sparsemax":
    weight_coefs = tf.contrib.sparsemax.sparsemax(activations)
  else:
    weight_coefs = activations
  weighted_encoded = tf.matmul(weight_coefs, candidate_values)
  return weighted_encoded, weight_coefs


def resnet_layer(tensor,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation="relu",
                 normalization=True,
                 reuse=False,
                 training=False,
                 name_scope="resnet"):
  """ResNet layer."""
  with tf.variable_scope(name_scope, reuse=reuse):
    x = tf.layers.conv2d(
        tensor,
        num_filters,
        kernel_size,
        strides,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.initializers.he_normal(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))

    if normalization:
      x = tf.layers.batch_normalization(
          x, momentum=0.9, epsilon=1e-5, fused=True, training=training)

    if activation is not None:
      x = tf.nn.relu(x)

    return x


def cnn_encoder(images, reuse, is_training=True, n_filters=16, depth=32):
  """Encoder block based on the ResNet architecture."""

  with tf.variable_scope("CNN_encoder", reuse=reuse):
    num_res_blocks = int((depth - 2) / 6)
    x = resnet_layer(
        images, reuse=reuse, training=is_training, name_scope="resnetblock1")
    for stack in range(3):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:
          strides = 2
        y = resnet_layer(
            tensor=x,
            num_filters=n_filters,
            strides=strides,
            reuse=reuse,
            training=is_training,
            name_scope="resnetblock2" + str(stack) + str(res_block))
        y = resnet_layer(
            tensor=y,
            num_filters=n_filters,
            activation=None,
            reuse=reuse,
            training=is_training,
            name_scope="resnetblock3" + str(stack) + str(res_block))
        if stack > 0 and res_block == 0:
          x = resnet_layer(
              tensor=x,
              num_filters=n_filters,
              kernel_size=1,
              strides=strides,
              activation=None,
              normalization=False,
              reuse=reuse,
              training=is_training,
              name_scope="resnetblock4" + str(stack) + str(res_block))
        x += y
        x = tf.nn.relu(x)
      n_filters *= 2

    x = tf.layers.average_pooling2d(x,
                                    (FLAGS.img_size // 4, FLAGS.img_size // 4),
                                    (1, 1))
    dense = tf.layers.flatten(x)
    dense = tf.layers.dense(
        inputs=dense, units=FLAGS.final_units, activation=tf.nn.relu)
    dense = tf.contrib.layers.layer_norm(dense)

    encoded_keys = tf.layers.dense(
        inputs=dense, units=FLAGS.attention_dim, activation=tf.nn.relu)
    encoded_queries = tf.layers.dense(
        inputs=dense, units=FLAGS.attention_dim, activation=tf.nn.relu)
    encoded_values = tf.layers.dense(
        inputs=dense, units=FLAGS.val_dim, activation=tf.nn.relu)
    encoded_values = tf.contrib.layers.layer_norm(encoded_values)

    return encoded_keys, encoded_queries, encoded_values
