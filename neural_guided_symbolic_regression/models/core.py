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

"""Utility functions for tensorflow model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile


# Sentinel values for empty lists in HParams. See hparams_list_value() below.
HPARAMS_EMPTY_LIST_INT = [-123456789]
HPARAMS_EMPTY_LIST_FLOAT = [-123456789.0]
HPARAMS_EMPTY_LIST_STRING = ['_hparams_empty_list_string']


def hparams_list_value(value):
  """Get a list-valued parameter.

  HParams cannot store empty lists as values, so we use this function to
  convert sentinel values to empty lists.

  Args:
    value: List; the value of any hyperparameter.

  Returns:
    If value matches one of the empty list sentinel values, an empty list.
    Otherwise the value is returned unchanged.

  Raises:
    TypeError: If value is not a list.
  """
  if not isinstance(value, list):
    raise TypeError('value must be a list')
  if (np.array_equal(value, HPARAMS_EMPTY_LIST_INT) or
      np.array_equal(value, HPARAMS_EMPTY_LIST_FLOAT) or
      np.array_equal(value, HPARAMS_EMPTY_LIST_STRING)):
    return []
  else:
    return value[:]  # Protect against in-place modification of lists.


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


def wait_for_hparams(filename, defaults, sleep_secs=60, max_attempts=10):
  """Waits for HParams to appear on disk.

  Args:
    filename: String hparams filename.
    defaults: HParams containing default values.
    sleep_secs: Integer number of seconds to sleep between attempts.
    max_attempts: Integer maximum number of attempts to read HParams.

  Returns:
    HParams.

  Raises:
    ValueError: If max_attempts is reached, or if one of the input arguments is
    invalid.
  """
  if sleep_secs < 0:
    raise ValueError('sleep_secs must be a positive integer or zero')
  if max_attempts <= 0:
    raise ValueError('max_attempts must be a positive integer')
  num_attempts = 0
  while num_attempts < max_attempts:
    try:
      hparams = read_hparams(filename, defaults)
      logging.info('Model HParams:\n%s', '\n'.join([
          '\t%s: %s' % (key, value)
          for key, value in sorted(six.iteritems(hparams.values()))
      ]))
      return hparams
    except ValueError as error:
      num_attempts += 1
      logging.info(
          'Could not find or parse hparams at %s, will sleep and retry:\n%s',
          filename, error)
      time.sleep(sleep_secs)
  raise ValueError('reached maximum number of attempts')


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
