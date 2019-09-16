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

from models import resnet_model
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_float('batch_norm_decay', 0.9, '')


def input_processing(features,
                     mode,
                     params,
                     batch_size,
                     do_input_processing=True,
                     multi_crops=False):
  """input preprocessing."""
  if not do_input_processing:
    return features

  if params['data_format'] == 'channels_first':
    assert not params['transpose_input']
    features = tf.transpose(features, [0, 3, 1, 2])

  if params['transpose_input'] and mode != tf.estimator.ModeKeys.PREDICT:
    image_size = 224
    if multi_crops:
      features = tf.reshape(features,
                            [image_size, image_size, 3, 10, batch_size])
      features = tf.transpose(features, [4, 3, 0, 1, 2])
      features = tf.reshape(features,
                            [batch_size * 10, image_size, image_size, 3])
    else:
      features = tf.reshape(features, [image_size, image_size, 3, batch_size])
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  return features


def resnet_v1_model(features, mode, params):
  """Architecutre of resnet."""
  features = input_processing(
      features, mode, params, batch_size=params['batch_size'])

  dropblock_keep_probs = [None] * 4
  if params['dropblock_groups']:
    # Scheduled keep_prob for DropBlock.
    train_steps = tf.cast(FLAGS.train_steps, tf.float32)
    current_step = tf.cast(tf.train.get_global_step(), tf.float32)
    current_ratio = current_step / train_steps
    dropblock_keep_prob = (1 - current_ratio *
                           (1 - params['dropblock_keep_prob']))

    # Computes DropBlock keep_prob for different block groups of ResNet.
    dropblock_groups = [int(x) for x in params['dropblock_groups'].split(',')]
    for block_group in dropblock_groups:
      if block_group < 1 or block_group > 4:
        raise ValueError(
            'dropblock_groups should be a comma separated list of integers '
            'between 1 and 4 (dropblcok_groups: {}).'.format(
                params['dropblock_groups']))
      dropblock_keep_probs[block_group - 1] = 1 - (
          (1 - dropblock_keep_prob) / 4.0**(4 - block_group))

  def build_network(features, is_training, reuse=None):
    network = resnet_model.resnet_v1(
        resnet_depth=params['resnet_depth'],
        num_classes=-1,
        dropblock_size=params['dropblock_size'],
        dropblock_keep_probs=dropblock_keep_probs,
        data_format=params['data_format'])
    avg_pool = network(inputs=features, is_training=is_training, reuse=reuse)
    return avg_pool

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  avg_pool = build_network(features, is_training=is_training)
  return avg_pool
