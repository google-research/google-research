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

"""Utilities for training the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import re
import time
import tensorflow.compat.v1 as tf

from bam.helpers import utils


class ETAHook(tf.train.SessionRunHook):
  """Prints out time elapsed and ETA during training."""

  def __init__(self, config, to_log, n_steps):
    self._config = config
    self._to_log = to_log
    self._n_steps = n_steps
    self._iter_count = 0
    self._should_trigger = False

  def before_run(self, run_context):
    self._should_trigger = (self._iter_count %
                            self._config.iterations_per_loop == 0)
    if self._should_trigger:
      return tf.train.SessionRunArgs(self._to_log)
    return None

  def after_run(self, run_context, run_values):
    if self._iter_count == 0:
      self._start_time = time.time()
    elif self._should_trigger:
      self.log(run_values)
    self._iter_count += (self._config.iterations_per_loop
                         if self._config.use_tpu else 1)

  def end(self, session):
    self.log()

  def log(self, run_values=None):
    msg = '{:}/{:} = {:.1f}%'.format(self._iter_count, self._n_steps,
                                     100.0 * self._iter_count / self._n_steps)
    time_elapsed = time.time() - self._start_time
    time_per_step = time_elapsed / self._iter_count
    msg += ', GPS: {:.1f}'.format(1 / time_per_step)
    msg += ', ELAP: ' + secs_to_str(time_elapsed)
    msg += ', ETA: ' +  secs_to_str(
        (self._n_steps - self._iter_count) * time_per_step)
    if run_values is not None:
      for tag, value in run_values.results.items():
        msg += ' - ' + str(tag) + (': {:.4f}'.format(value))
    utils.log(msg)


def secs_to_str(secs):
  s = str(datetime.timedelta(seconds=int(round(secs))))
  s = re.sub('^0:', '', s)
  s = re.sub('^0', '', s)
  s = re.sub('^0:', '', s)
  s = re.sub('^0', '', s)
  return s
