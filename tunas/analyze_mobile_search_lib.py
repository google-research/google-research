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

# Lint as: python2, python3
"""Utilities for reading logs from mobile architecture search experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Any, Dict, Text, List, TypeVar

import six
from six.moves import map
from six.moves import range

import tensorflow.compat.v1 as tf

# Matches a pattern like 'rllogits/0_3' or 'rlfilterslogits/2_5'.
#
# In older legacy code, we log the RL controller's logits using the format
# 'rllogits/[index]_[choice]', where [index] and [choice] are non-negative
# integers. For example, if we search over 9 possible ops then the value of
# [index] would range from 0 to 8 (inclusive). For Op 5, if there are 4 possible
# choices then the logs will contain scalars 'rllogits/5_0' ... 'rllogits/5_3'.
# The situation is similar for filters, except that scalars start with
# 'rlfilterslogits/' instead of 'rllogits'.
_EVENT_PATTERN_TAG_V1 = re.compile(
    six.u(r'^(rllogits|rlfilterslogits)/(\d+)_(\d+)$'))

# Matches a pattern like 'rltaglogits/op_indices_0/3'.
#
# In newer code, we log the RL controller's logits by tag using the format
# 'rltaglogits/[tagname]_[index]/[choice]'. For example, if we search over a
# structure with 9 OneOf nodes that have the tag 'op_indices' then we'll
# end up generating scalars whose names start with 'rllogits/op_indices_0/'
# through 'rllogits/op_indices_8/'. For OneOf 5, if there are 4 possible choices
# then the logs will contain scalars 'rltaglogits/op_indices_5/0' through
# 'rltaglogits/op_indices_5/3'.
_EVENT_PATTERN_TAG_V2 = re.compile(
    six.u(r'^rltaglogits/(\w+)_(\d+)/(\d+)$'))

# Matches a pattern like 'rlpathlogits/backbone/blocks/0/filters/3'
#
# In addition to the tag logits described above, the newer code also logs the RL
# controller's logits by path using the format 'rltaglogits/[path]/[choice]'.
# Each `path` is formatted as, e.g., `backbone/blocks/2/filters`, corresponding
# to the tuple path of a OneOf object in the model spec.
_EVENT_PATTERN_PATH = re.compile(six.u(r'^rlpathlogits/(.*)/(\d+)$'))


_T = TypeVar('_T')


def _dict_to_list(elements):
  expected_keys = list(range(len(elements)))
  if sorted(elements) != expected_keys:
    raise ValueError('dictionary keys must be consecutive and start from 0: {}'
                     .format(sorted(elements)))
  return [elements[k] for k in expected_keys]


def _get_scalar_value(value):
  """Extract the value from a summary_pb2.Value proto."""
  if value.HasField('simple_value'):
    return float(value.simple_value)
  elif value.HasField('tensor'):
    return float(tf.make_ndarray(value.tensor).item())
  else:
    raise ValueError('Can\'t extract scalar value from field: {}'.format(value))


def read_tag_logits(dirname):
  """Extract tag logits from tfevents files in the specified directory.

  Args:
    dirname: String, path of the directory to read events files from.

  Returns:
    A data structure
        (global_step:int -> tag:string -> index:int -> logits:float list).
    In other words, we return a dictionary of dictionary of lists of floats,
    where result[global_step][tag] is a list of logits for a OneOf object in
    the search space.
  """
  events = dict()  # type: Dict[int, Dict[Text, Dict[int, Dict[int, float]]]]
  # Type of `events` is (step -> key -> index -> choice -> logit)
  #   - step is the current global step
  #   - key is a human-readable string (like 'rllogits')
  #   - index is the index of the current OneOf (0, 1, 2, ...)
  #   - choice is the index of the current OneOf choice (0, 1, 2, ...)
  #   - logit is a float, the logit of the current choice

  filenames = os.path.join(dirname, 'events.out.tfevents.*.v2')
  for filename in tf.io.gfile.glob(filenames):
    try:
      for event in tf.train.summary_iterator(filename):
        for value in event.summary.value:
          match = _EVENT_PATTERN_TAG_V2.match(value.tag)
          match = match or _EVENT_PATTERN_TAG_V1.match(value.tag)
          if match:
            key = match.group(1)
            index = int(match.group(2))
            choice = int(match.group(3))
            logit = float(_get_scalar_value(value))

            if event.step not in events:
              events[event.step] = dict()
            if key not in events[event.step]:
              events[event.step][key] = dict()
            if index not in events[event.step][key]:
              events[event.step][key][index] = dict()
            events[event.step][key][index][choice] = logit
    except tf.errors.DataLossError:
      # For some reason, data corruption is pretty common when reading tfevents
      # files. Ignore the rest of the current file, and go on to the next file.
      pass

  result = dict()  # type: Dict[int, Dict[Text, List[List[float]]]]
  for step, step_events in events.items():
    step_logits = dict()
    try:
      for key, current_logits in step_events.items():
        current_logits = _dict_to_list(current_logits)
        step_logits[key] = list(map(_dict_to_list, current_logits))
      result[step] = step_logits
    except ValueError:
      # Looks like some data is missing from the event files. This is pretty
      # common if we try to read the logs of a search job while it's running.
      # Just go on to the next step, and don't try to populate result[step]
      # for the current training step.
      pass
  return result


def read_path_logits(dirname):
  """Extract path logits from tfevents files in the specified directory.

  Args:
    dirname: String, path of the directory to read events files from.

  Returns:
    A data structure (global_step:int -> path:string -> logits:float list). In
    other words, we return a dictionary of dictionary of float-valued logits,
    where result[global_step][path] is the logits for a OneOf object in the
    search space.
  """
  events = dict()  # type: Dict[int, Dict[Text, Dict[int, float]]]
  filenames = os.path.join(dirname, 'events.out.tfevents.*.v2')
  for filename in tf.io.gfile.glob(filenames):
    try:
      for event in tf.train.summary_iterator(filename):
        for value in event.summary.value:
          match = _EVENT_PATTERN_PATH.match(value.tag)
          if match:
            key = match.group(1)
            choice = int(match.group(2))
            logit = float(_get_scalar_value(value))

            if event.step not in events:
              events[event.step] = dict()
            if key not in events[event.step]:
              events[event.step][key] = dict()
            events[event.step][key][choice] = logit
    except tf.errors.DataLossError:
      # This indicates a data corruption, where we will ignore the rest of the
      # current file and go on to the next file.
      pass

  result = dict()  # type: Dict[int, Dict[Text, List[float]]]
  for step, step_events in events.items():
    step_logits = dict()
    try:
      for key, current_logits in step_events.items():
        step_logits[key] = _dict_to_list(current_logits)
      result[step] = step_logits
    except ValueError:
      # Looks like some data is missing from the event files.
      pass
  return result
