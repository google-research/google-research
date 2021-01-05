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

"""A binary to train pruned CIFAR-10.

Accuracy:
  When using low-rank decomposition and a rank of 400 (meaning 50% compression)
  cifar10_train.py achieves ~84% accuracy after 120K steps - as judged by
  cifar10_eval.py using test data.

Results:
Compression (Low-Rank) | Accuracy after 150K steps
---------------------- | -------------------------
0%                     | 86%
50%                    | 84.3%
66%                    | 83.9%

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys
import time

import tensorflow.compat.v1 as tf

from graph_compression.compression_lib.examples.cifar10 import cifar10_compression as cifar10
from tensorflow.contrib import framework as contrib_framework

FLAGS = None


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = contrib_framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Create a compression object using the compression hyperparameters
    compression_obj = cifar10.create_compressor(
        FLAGS.compression_hparams, global_step=global_step)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, compression_obj)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, compression_obj)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = 128
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[
            tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
            tf.train.NanTensorHook(loss),
            _LoggerHook()
        ],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(unused_argv=None):
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/cifar10_train',
      help="""Directory where to write event logs and checkpoint.""")
  parser.add_argument(
      '--compression_hparams',
      type=str,
      default='',
      help="""Comma separated list of compression-related hyperparameters""")
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000000,
      help="""Number of batches to run.""")
  parser.add_argument(
      '--log_device_placement',
      type=bool,
      default=False,
      help="""Whether to log device placement.""")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
