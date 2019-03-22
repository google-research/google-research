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

"""Common utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import tensorflow as tf

gfile = tf.gfile


class Logger(object):
  """Prints to both STDOUT and a file."""

  def __init__(self, filepath):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filepath, 'a+')

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def get_lr(curr_step, params):
  """Compute learning rate at step depends on `params`."""
  lr = tf.constant(params.learning_rate, dtype=tf.float32)
  if 'num_warmup_steps' in params and params.num_warmup_steps > 0:
    num_warmup_steps = tf.cast(params.num_warmup_steps, dtype=tf.float32)
    step = tf.cast(curr_step, dtype=tf.float32)
    warmup_lr = params.learning_rate * step / num_warmup_steps
    lr = tf.cond(tf.less(step, num_warmup_steps), lambda: warmup_lr, lambda: lr)
  return lr


def strip_var_name(var_name):
  """Strips variable name of sub-strings blocking variable name matching."""
  # Strip trailing number, e.g. convert
  # 'lstm/W_0:0' to 'lstm/W_0'.
  var_name = re.sub(r':\d+$', '', var_name)
  # Strip partitioning info, e.g. convert
  # 'W_0/part_3/Adagrad' to 'W_0/Adagrad'.
  var_name = re.sub(r'/part_\d+', '', var_name)
  return var_name
