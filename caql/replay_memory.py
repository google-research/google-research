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

"""Thread-safe and checkpoint-able Replay Memory."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import os.path
import pickle
import random
import re
import threading

import tensorflow.compat.v1 as tf


class ReplayMemory(object):
  """Replay memory."""

  def __init__(self, name, capacity):
    self._name = name
    self._buffer = collections.deque(maxlen=capacity)
    self._lock = threading.Lock()

  def _get_checkpoint_filename(self, number):
    return 'replay_memory-{}.pkl-{}'.format(self._name, number)

  def _get_latest_checkpoint_number(self, checkpoint_dir_path):
    checkpoint_numbers = []
    if checkpoint_dir_path and tf.gfile.Exists(checkpoint_dir_path):
      for filename in tf.gfile.ListDirectory(checkpoint_dir_path):
        m = re.match(r'replay_memory-{}\.pkl-(\d+)$'.format(self._name),
                     filename)
        if m:
          checkpoint_numbers.append(int(m.group(1)))
    if checkpoint_numbers:
      checkpoint_numbers.sort(reverse=True)
      return checkpoint_numbers[0]
    return -1

  @property
  def size(self):
    return len(self._buffer)

  @property
  def capacity(self):
    return self._buffer.maxlen

  def clear(self):
    with self._lock:
      self._buffer.clear()

  def extend(self, experience):
    with self._lock:
      self._buffer.extend(experience)

  def batch_extend(self, batch_experience, include_init_state=False):
    with self._lock:
      for (init_state, state, action, reward, next_state, done,
           info) in zip(*batch_experience):
        if include_init_state:
          self._buffer.append(
              [init_state, state, action, reward, next_state, done, info])
        else:
          self._buffer.append([state, action, reward, next_state, done, info])

  def get_buffer(self):
    return self._buffer

  def sample_with_replacement(self, size):
    with self._lock:
      return random.choices(self._buffer, k=size)

  def sample(self, size):
    with self._lock:
      return random.sample(self._buffer, size)

  def save(self, checkpoint_dir_path, delete_old=False):
    if not tf.gfile.Exists(checkpoint_dir_path):
      tf.gfile.MakeDirs(checkpoint_dir_path)

    with self._lock:
      latest_checkpoint_number = self._get_latest_checkpoint_number(
          checkpoint_dir_path)
      file_path = os.path.join(
          checkpoint_dir_path,
          self._get_checkpoint_filename(latest_checkpoint_number + 1))
      with tf.gfile.Open(file_path, 'wb') as f:
        pickle.dump(self._buffer, f)

      if delete_old:
        file_path = os.path.join(
            checkpoint_dir_path,
            self._get_checkpoint_filename(latest_checkpoint_number))
        if tf.gfile.Exists(file_path):
          tf.gfile.Remove(file_path)

  def restore(self, checkpoint_dir_path):
    with self._lock:
      checkpoint_number = self._get_latest_checkpoint_number(
          checkpoint_dir_path)
      if checkpoint_number < 0:
        return
      file_path = os.path.join(checkpoint_dir_path,
                               self._get_checkpoint_filename(checkpoint_number))
      tf.logging.info('Restoring replay memory using checkpoint file: %s',
                      file_path)
      with tf.gfile.Open(file_path, 'rb') as f:
        self._buffer = pickle.load(f)
