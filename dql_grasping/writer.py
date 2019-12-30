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

"""Write episode transitions to Recordio-backed replay buffer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class TFRecordReplayWriter(object):
  """Saves transitions to a TFRecord-backed replay buffer."""

  def __init__(self):
    self.writer = None

  def open(self, path):
    if self.writer is not None:
      raise ValueError('Writer is already open!')

    path_dirname = os.path.dirname(path)
    if not tf.gfile.IsDirectory(path_dirname):
      tf.gfile.MakeDirs(path_dirname)

    self.writer = tf.python_io.TFRecordWriter(path + '.tfrecord')

  def close(self):
    if self.writer is None:
      raise ValueError('Writer is not open!')
    self.writer.close()
    self.writer = None

  def write(self, transitions):
    """Writes entire episode to a TFRecord file.

    Args:
      transitions: List of tf.Examples.

    Raises:
      ValueError: If writer has not been opened.
    """
    if self.writer is None:
      raise ValueError('Writer is not open!')
    for transition in transitions:
      self.writer.write(transition.SerializeToString())
