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

"""Logging utilities."""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union, List

from absl import logging
from absl import flags
from clu import metric_writers
from acme.utils import loggers
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from acme.utils.loggers import tf_summary
from acme.utils.loggers import auto_close



PrimaryKeyList = Sequence[Tuple[str, Union[int, str]]]
FLAGS = flags.FLAGS


class StepFilter(loggers.Logger):
  """Logger which subsamples logs to another logger.

  It considers a counter defined by 'steps_label' and makes sure that in each
  window of length 'delta' we write at most one set of values.
  """


  def __init__(self,
               to: loggers.Logger,
               steps_label: str = 'steps',
               delta: int = 1000):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      steps_label: label for the step counter.
      delta: logs data at most every #delta steps
    """
    self._to = to
    self._steps_label = steps_label
    self._delta = delta
    self._last_logged_step = None

  def write(self, values: loggers.LoggingData):
    current_step = values.get(self._steps_label, 0)
    if (self._last_logged_step is None or
        current_step >= self._last_logged_step + self._delta):
      self._to.write(values)
      self._last_logged_step = current_step

  def close(self):
    self._to.close()


def create_default_logger(
    label: str,
    tf_summary_logdir: str,
    save_data: bool = True,
    step_filter_delta: int = 1,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = 'steps',
    extra_primary_keys: Optional[PrimaryKeyList] = None,) -> base.Logger:
  """Creates a logger that has TerminalLogger and TF Summary."""
  loggers = []

  terminal_logger = terminal.TerminalLogger(label, print_fn=print)
  loggers.append(terminal_logger)


  # tensorboard logger
  tf_logger = tf_summary.TFSummaryLogger(
      logdir=tf_summary_logdir,
      label=label,
      steps_key=steps_key,
      # steps_key=None,
  )
  loggers.append(tf_logger)

  # aggregate and add modifiers
  logger = aggregators.Dispatcher(loggers, serialize_fn=serialize_fn)
  logger = filters.NoneFilter(logger)
  if step_filter_delta > 1:
    logger = StepFilter(logger, steps_label=steps_key, delta=step_filter_delta)
  logger = filters.TimeFilter(logger, time_delta=time_delta)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  # logger = filters.TimeFilter(logger, time_delta=time_delta)
  # logger = logger_filters.FlattenDictLogger(logger, label=label)

  return logger


