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

"""Utility functions and other shared chemgraph code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile


def read_hparams(filename, defaults):
  """Reads HParams from JSON.

  Args:
    filename: String filename.
    defaults: HParams containing default values.

  Returns:
    HParams.

  Raises:
    gfile.Error: If the file cannot be read.
    ValueError: If the JSON record cannot be parsed.
  """
  with gfile.Open(filename) as f:
    logging.info('Reading HParams from %s', filename)
    return defaults.parse_json(f.read())


def write_hparams(hparams, filename):
  """Writes HParams to disk as JSON.

  Args:
    hparams: HParams.
    filename: String output filename.
  """
  with gfile.Open(filename, 'w') as f:
    f.write(hparams.to_json(indent=2, sort_keys=True, separators=(',', ': ')))


def learning_rate_decay(initial_learning_rate, decay_steps, decay_rate):
  """Initializes exponential learning rate decay.

  Args:
    initial_learning_rate: Float scalar tensor containing the initial learning
      rate.
    decay_steps: Integer scalar tensor containing the number of steps between
      updates.
    decay_rate: Float scalar tensor containing the decay rate.

  Returns:
    Float scalar tensor containing the learning rate. The learning rate will
    automatically be exponentially decayed as global_step increases.
  """
  with tf.variable_scope('learning_rate_decay'):
    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=tf.train.get_global_step(),
        decay_steps=decay_steps,
        decay_rate=decay_rate)
  tf.summary.scalar('learning_rate', learning_rate)
  return learning_rate
