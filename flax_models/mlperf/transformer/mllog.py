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

# Copyright 2019 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================
"""Convenience function for logging compliance tags to stdout.

This should be replaced by the upstream mllog when that's ready.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import inspect
import json
import logging
import re
import sys
import time

try:
  import numpy as np  # pylint: disable=g-import-not-at-top
  NUMPY_FOUND = True
except ImportError:
  NUMPY_FOUND = False

LOG_TEMPLATE = ':::MLLOG {log_json}'


def get_caller(stack_index=2, root_dir=None):
  """Get caller's file and line number information.

  Args:
    stack_index: a stack_index of 2 will provide the caller of the
      function calling this function. Notice that stack_index of 2
      or more will fail if called from global scope.
    root_dir: the root dir prefixed to the file name. The root_dir
      will be trimmed from the file name in the output.
  Returns:
    Call site info in a dictionary with these fields:
      "file": (string) file path
      "lineno": (int) line number
  """
  frame = inspect.currentframe()

  for _ in range(stack_index):
    frame = frame.f_back
  caller = inspect.getframeinfo(frame)

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub('^' + root_dir + '/', '', filename)
  return {'file': filename, 'lineno': caller.lineno}


def _now_as_str():
  """Returns the current time as a human readable string."""
  return datetime.datetime.now().strftime('%H:%M:%S.%f')


def _current_time_ms():
  """Returns current milliseconds since epoch."""
  return int(time.time() * 1e3)


def _encode_log(namespace, time_ms, event_type, key, value, metadata):
  """Encodes an MLEvent as a string log line.

  Args:
    namespace: provides structure, e.g. "GPU0".
    time_ms: milliseconds since unix epoch.
    event_type: one of: 'INTERVAL_START', 'INTERVAL_END', 'POINT_IN_TIME'
    key: the name of the thing being logged.
    value: a json value.
    metadata: a json value.
  Returns:
    A string log like, i.e. ":::MLLog { ..."
  """
  # preserve the order of key-values
  ordered_key_val_pairs = [
      ('namespace', namespace),
      ('time_ms', time_ms),
      ('event_type', event_type),
      ('key', key),
      ('value', value),
      ('metadata', metadata)
  ]
  d = collections.OrderedDict(ordered_key_val_pairs)
  if NUMPY_FOUND:
    encoded = json.dumps(d, cls=_NumpyJSONEncoder)
  else:
    encoded = json.dumps(d)
  return LOG_TEMPLATE.format(log_json=encoded)


def _encode_log_quiet(key, value, metadata):
  if value is None:
    return f'{key}; {metadata}'
  else:
    return f'{key}: {value}; {metadata}'


class _NumpyJSONEncoder(json.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


class MLLogger(object):
  """MLPerf logging helper."""

  def __init__(self,
               logger=None,
               default_namespace='',
               default_stack_offset=1,
               default_clear_line=False,
               root_dir=None,
               full=True):
    """Create a new MLLogger.

    Args:
      logger: a logging.Logger instance. If not specified, a default logger
        will be used which prints to stdout. Customize the logger to change
        the logging behavior (e.g. logging to a file, etc.)
      default_namespace: the default namespace to use if one isn't provided.
      default_stack_offset: the default depth to go into the stack to find the
        call site. Default value is 1.
      default_clear_line: the default behavior of line clearing (i.e. print
        an extra new line to clear any pre-existing text in the log line).
      root_dir: directory prefix which will be trimmed when reporting calling
        file for logging.
      full: whether to emit full MLPerf logging or more minimal logging.
    """
    if logger is None:
      self.logger = self._get_default_logger()
    elif not isinstance(logger, logging.Logger):
      raise ValueError('logger must be a `logging.Logger` instance.')
    else:
      self.logger = logger
    self.full = full

    self.default_namespace = default_namespace
    self.default_stack_offset = default_stack_offset
    self.default_clear_line = default_clear_line
    self.root_dir = root_dir

  def _get_default_logger(self):
    """Create a default logger which prints INFO level messages to stdout."""
    logger = logging.getLogger('mllog_default')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger

  def _do_log(self, level, message, clear_line=False):
    if clear_line:
      message = '\n' + message
    self.logger.log(level, message)

  def _log_helper(self, event_type, key, value=None, metadata=None,
                  namespace=None, time_ms=None, stack_offset=None,
                  clear_line=None):
    """Log an event."""
    if namespace is None:
      namespace = self.default_namespace
    if time_ms is None:
      time_ms = _current_time_ms()
    if stack_offset is None:
      stack_offset = self.default_stack_offset
    if clear_line is None:
      clear_line = self.default_clear_line

    log_metadata = {}
    if self.full:
      log_metadata.update(get_caller(2 + stack_offset, root_dir=self.root_dir))
    if metadata:
      if not isinstance(metadata, dict):
        self._do_log(logging.WARNING, 'Metadata is not dictionary, ignored.',
                     clear_line=True)
      else:
        overlap_keys = set(log_metadata.keys()).intersection(metadata.keys())
        if overlap_keys:
          self._do_log(
              logging.WARNING,
              'Metadata fields overridden: {}'.format(', '.join(overlap_keys)),
              clear_line=True)
        log_metadata.update(metadata)

    try:
      if self.full:
        log_line = _encode_log(
            namespace, time_ms, event_type, key, value, log_metadata)
      else:
        log_line = _encode_log_quiet(key, value, metadata)
      self._do_log(logging.INFO, log_line, clear_line)
    except Exception:  # pylint:disable=broad-except
      self._do_log(
          logging.ERROR,
          'Failed to encode: {}'.format(
              str([namespace, time_ms, event_type, key, value, log_metadata])),
          clear_line=True)

  def start(self, key, value=None, metadata=None, namespace=None, time_ms=None,
            stack_offset=None, clear_line=None):
    """Start a time interval in the log.

    All intervals which are started must be ended. This interval must be
    ended before a new interval with the same key and namespace can be started.
    Args:
      key: the key for the event, e.g. "mlperf.training"
      value: the value to log at the start of the interval.
      metadata: a dictionary containing metadata corresponding to the log event.
      namespace: override the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper('INTERVAL_START', key, value,
                     metadata=metadata, namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)

  def end(self, key, value=None, metadata=None, namespace=None, time_ms=None,
          stack_offset=None, clear_line=None):
    """End a time interval in the log.

    Ends an interval which was already started with the same key and in the
    same namespace.
    Args:
      key: the same log key which was passed to start().
      value: the value to log at the end of the interval.
      metadata: a dictionary containing metadata corresponding to the log event.
      namespace: optional override of the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper('INTERVAL_END', key, value,
                     metadata=metadata, namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)

  def event(self, key, value=None, metadata=None, namespace=None, time_ms=None,
            stack_offset=None, clear_line=None):
    """Log a point in time event.

    The event does not have an associated duration like an interval has.
    Args:
      key: the "name" of the event.
      value: the event data itself.
      metadata: a dictionary containing metadata corresponding to the log event.
      namespace: optional override of the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper('POINT_IN_TIME', key, value,
                     metadata=metadata, namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)
