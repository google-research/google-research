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

"""Model architectures."""

from absl import flags

import tensorflow as tf

flags.DEFINE_string('source_dataset', 'mnist', 'Name of the source dataset.')
flags.DEFINE_string('target_dataset', 'svhn_cropped_small', 'Name of the target dataset.')
flags.DEFINE_integer('src_num_classes', 10, 'The number of classes in the source dataset.')
flags.DEFINE_integer('src_hw', 28, 'The height and width of source inputs.')
flags.DEFINE_integer('target_hw', 32, 'The height and width of source inputs.')

FLAGS = flags.FLAGS


def conv_model(features, mode, dataset_name=None, reuse=None):
  """Architecutre of convt."""

  def build_network(features, is_training):
    """Model function for CNN."""
      # Input reshape
    if dataset_name == 'mnist' or FLAGS.target_dataset == 'mnist':
        input_layer = tf.reshape(
            features,[FLAGS.train_batch_size, FLAGS.src_hw, FLAGS.src_hw, 1])
        input_layer = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]])
    else:
        input_layer = tf.reshape(
            features, [FLAGS.train_batch_size, FLAGS.target_hw, FLAGS.target_hw, 3])
        input_layer = tf.image.rgb_to_grayscale(input_layer)

    input_layer = tf.reshape(
        input_layer, [FLAGS.train_batch_size, FLAGS.target_hw, FLAGS.target_hw, 1])
    input_layer = tf.image.convert_image_dtype(input_layer, dtype=tf.float32)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1',
        reuse=reuse)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2',
        reuse=reuse,
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    discard_rate = 0.7
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu,
                            name='dense_256', reuse=reuse)
    dropout = dense
    if is_training:
        dropout = tf.layers.dropout(
            inputs=dense, rate=discard_rate)

    return dropout

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  avg_pool = build_network(features, is_training=is_training)
  return avg_pool
