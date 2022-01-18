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

# Lint as: python3
"""Object for logging training stats.

Logger is for Tensorflow 1.0 and Logger2 is for Tensorflow 2.0.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Logger:
  """Logging stats to std out and save tf summary.

  Attributes:
    summary_writer: a tf1.0 summary writer
  """

  def __init__(self, save_dir):
    self.summary_writer = tf.summary.FileWriter(save_dir)

  def log(self, epoch, cycle, stats):
    """Logs stats for one cycle.

    Args:
      epoch: current epoch number
      cycle: current cycle number
      stats: stats to be saved
    """
    print('#####################################')
    print('epoch: {}'.format(epoch))
    print('cycle: {}'.format(cycle))
    _ = [print('{}:  {}'.format(k, v)) for k, v in stats.items()]
    summary_list = [
        tf.Summary.Value(tag=k, simple_value=stats[k]) for k in stats
    ]
    summary = tf.Summary(value=summary_list)
    self.summary_writer.add_summary(summary, stats['global_step'])


class Logger2:
  """Logging stats to std out and save tf summary.

  Attributes:
    summary_writer: a tf2.0 file writer
  """

  def __init__(self, save_dir):
    self.summary_writer = tf.compat.v2.summary.create_file_writer(save_dir)

  def log(self, epoch, cycle, stats):
    """Logs stats for one cycle.

    Args:
      epoch: current epoch number
      cycle: current cycle number
      stats: stats to be saved
    """
    print('#####################################')
    print('epoch: {}'.format(epoch))
    print('cycle: {}'.format(cycle))
    _ = [print('{}:  {}'.format(k, v)) for k, v in stats.items()]
    with self.summary_writer.as_default():
      for k in stats:
        tf.compat.v2.summary.scalar(k, stats[k], step=stats['global_step'])
    self.summary_writer.flush()
