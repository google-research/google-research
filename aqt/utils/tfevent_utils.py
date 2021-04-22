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

"""Functions to load a TFRecordDataset and extract TFEvents."""

import collections
import os
from typing import Dict, List, Optional

import dataclasses
import numpy as onp
import tensorflow as tf


@dataclasses.dataclass
class EventSeries:
  """To store series of events as coupled lists. All lists have same length."""
  # Name of the metric
  name: str
  # List of steps when metric was recorded, 1D list of integers.
  steps: onp.ndarray
  # Values recorded at steps, same length as steps, 1D list of floats.
  values: onp.ndarray
  # Time when value was recorded, same length as steps, 1D list of floats.
  wall_times: Optional[onp.ndarray] = None


def _sort_and_deduplicate_entries(event_series):
  """Sorts entries by (step, wall_time), deduplicates entries with same step.

  Will keep the one with the latest wall_time and remove wall_time from result.
  If resulting wall_times are not sorted, will raise error.

  Args:
    event_series: Series of (step, value, wall_time) tuples, stored as coupled
      lists in EventSeries dataclass instance. Doesn't need to be sorted, as
      this function will sort it.

  Returns:
    Sorted Series of (step, value) tuples with unique steps, stored as coupled
    lists in EventSeries dataclass instance.
  """
  # Sort lists first by step, then by wall time
  sort_indices = onp.lexsort((event_series.wall_times, event_series.steps))
  event_series.steps = event_series.steps[sort_indices]
  event_series.wall_times = event_series.wall_times[sort_indices]
  event_series.values = event_series.values[sort_indices]
  cur_step = None
  cur_value = None
  res_event_series = EventSeries(
      name=event_series.name,
      steps=onp.array([], dtype=int),
      values=onp.array([]),
      wall_times=onp.array([]))
  for i in range(len(event_series.steps)):
    if event_series.steps[i] != cur_step:
      if cur_value is not None:
        res_event_series.steps = onp.append(res_event_series.steps, cur_step)
        res_event_series.values = onp.append(res_event_series.values, cur_value)
        res_event_series.wall_times = onp.append(res_event_series.wall_times,
                                                 cur_wall_time)
    cur_step = event_series.steps[i]
    cur_value = event_series.values[i]
    cur_wall_time = event_series.wall_times[i]
  if cur_value is not None:
    res_event_series.steps = onp.append(res_event_series.steps, cur_step)
    res_event_series.values = onp.append(res_event_series.values, cur_value)
    res_event_series.wall_times = onp.append(res_event_series.wall_times,
                                             cur_wall_time)

  def _is_sorted(arr):
    return onp.all(onp.diff(arr) >= 0)

  if not _is_sorted(res_event_series.wall_times):
    raise ValueError(
        'Resulting EventSeries list after sorting by (step, wall_time)'
        ' has unsorted wall_times, likely caused by error in '
        'training.')
  # Remove wall_times data since not needed anymore.
  res_event_series.wall_times = None
  return res_event_series


def get_tfevent_paths(dir_path):
  """Read a TFRecordDataset from a directory path.

  The TFRecordDataset can be loaded from one or multiple files. E.g. if training
  restarts, multiple files would be saved.

  The events file(s) should be located either directly in dir_path, or nested
  one more level (in a subdirectory).

  Args:
    dir_path: Path to a directory containing file(s) readable as a
      TFRecordDataset.

  Returns:
    A list of tf event file paths.
  """
  if not tf.io.gfile.isdir(dir_path):
    raise ValueError(f'dir_path {dir_path} must point to a directory.')

  def _is_events_file(filename):
    return 'events.out.tfevents' in filename

  tfevent_file_paths = []
  for filename in tf.io.gfile.listdir(dir_path):
    if tf.io.gfile.isdir(filename):
      for nested_filename in filename.iterdir():
        if _is_events_file(nested_filename):
          tfevent_file_paths.append(
              os.path.join(dir_path, filename, nested_filename))
      break
    else:
      if _is_events_file(filename):
        tfevent_file_paths.append(os.path.join(dir_path, filename))
  return tfevent_file_paths


def get_parsed_tfevents(dir_path,
                        tags_to_include):
  """Retrieves (step, value) tuples from TFEvents files, filtered by tags.

  Args:
    dir_path: Path to a directory containing TFEvent files.
    tags_to_include: List of tags (attributes), e.g. 'loss' that should be
      included in the report.

  Returns:
    A dictionary mapping tag name to a EventSeries dataclass.
  """
  tfevent_file_paths = get_tfevent_paths(dir_path)

  event_series_dict = collections.defaultdict(list)
  for path in tfevent_file_paths:
    for event in tf.compat.v1.train.summary_iterator(str(path)):
      # There is no v2 API for summary_iterator, so using tf.compat.v1 instead.
      for value in event.summary.value:
        if value.tag in tags_to_include:
          # Some tag names have spaces.
          tag_str = str(value.tag).replace(' ', '_')
          if tag_str not in event_series_dict:
            event_series_dict[tag_str] = EventSeries(
                name=tag_str,
                steps=onp.array([], dtype=int),
                values=onp.array([]),
                wall_times=onp.array([]))
          event_series_dict[tag_str].steps = onp.append(
              event_series_dict[tag_str].steps, event.step)
          event_series_dict[tag_str].values = onp.append(
              event_series_dict[tag_str].values,
              float(tf.make_ndarray(value.tensor)))
          event_series_dict[tag_str].wall_times = onp.append(
              event_series_dict[tag_str].wall_times, event.wall_time)

  for tag_str in event_series_dict:
    # Sort and deduplicate entries with the same step, keeping one with latest
    # wall_time
    event_series_dict[tag_str] = _sort_and_deduplicate_entries(
        event_series_dict[tag_str])
  return event_series_dict
