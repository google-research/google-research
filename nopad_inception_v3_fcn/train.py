# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Training script for nopad_inception_v3_fcn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.examples.tutorials.mnist import input_data

from nopad_inception_v3_fcn import model
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'number_of_steps', 20, 'The number of steps for training.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 5, 'How often, in seconds, to save summaries.')
tf.app.flags.DEFINE_string(
    'logdir', '/tmp/nopad_inception_v3', 'The directory for logging.')

FLAGS = tf.app.flags.FLAGS

_NUM_CLASSES = 10
_MNIST_IMAGE_SIZE = 28
_IMG_SIZE = 911


def main(_):

  mnist = input_data.read_data_sets(
      os.path.join(FLAGS.logdir, 'data'), one_hot=True, seed=0).train

  with tf.Graph().as_default():
    images, labels = mnist.next_batch(FLAGS.batch_size)

    images = tf.reshape(images, [-1, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1])
    images = tf.image.resize_images(images, [_IMG_SIZE, _IMG_SIZE])

    labels = tf.reshape(labels, [-1, 1, 1, _NUM_CLASSES])

    logits, _ = model.nopad_inception_v3_fcn(
        images, num_classes=_NUM_CLASSES)

    slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = slim.losses.get_total_loss()

    tf.summary.scalar('losses/Total_Loss', total_loss)

    optimizer = tf.train.RMSPropOptimizer(0.01)

    train_op = slim.learning.create_train_op(total_loss, optimizer)

    slim.learning.train(
        train_op,
        logdir=FLAGS.logdir,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs)


if __name__ == '__main__':
  tf.app.run()
