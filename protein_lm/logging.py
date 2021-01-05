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

# Lint as: python3
"""Logging utilities."""
from absl import logging
import tensorflow.compat.v1 as tf

# Alias to TF summary module that supports eager mode and is TF2 compatible.
tf_summary = tf.compat.v2.summary


class ScalarSummary(object):
  """Enables logging a scalar metric to Tensorboard.

  Example:
    num_rounds = tf.Variable(0, dtype=tf.int64, trainable=False)
    summary = ScalarSummary()

    Anywhere in your code:
      summary('summary_name', summary_value)

    After each round:
      num_rounds.assign_add(1)
  """

  def __init__(self, step=None, scope=None, enable_tf=False, verbose=1):
    """Creates an instance of this class.

    Args:
      step: An optional `tf.Variable` for tracking the logging step. If `None`,
        will use the global Tensorflow step variable.
      scope: An optional string that is prepended to metric names passed to
        `__call__`.
      enable_tf: Whether to create a TF summary.
      verbose: Whether to also summaries to the console.
    """
    self._step = tf.train.get_or_create_global_step() if step is None else step
    self._scope = scope
    self._verbose = verbose
    self._enable_tf = enable_tf

  def __call__(self, name, value, step=None):
    """Creates or updates summary `name` with `value`.

    Args:
      name: The name of the summary.
      value: The value of the summary.
      step: An optional step variable. If `None`, will use the step variable
        passed to the constructor.
    """
    step = self._step if step is None else step
    if self._scope:
      name = self._scope + name
    if self._enable_tf:
      tf_summary.scalar(name, value, step=step)

    if self._verbose:
      logging.info('Summary step=%d: %s=%.3f', int(step), name, value)
