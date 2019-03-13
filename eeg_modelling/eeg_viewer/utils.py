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

"""Utility functions for parsing model input and output protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import logging

import gviz_api


def TimestampPb2ToSeconds(timestamp):
  return timestamp.seconds + timestamp.nanos * (1e-9)


def GetSubsamplingRate(num_samples, max_samples):
  subsampling_rate = int(math.ceil(num_samples / max_samples))
  if subsampling_rate != 1:
    logging.warn('Data may be damaged, uniform downsampling at rate of %s',
                 subsampling_rate)
  return subsampling_rate


def GetSampleRange(freq, duration_sec, start_sec):
  """Creates index range [start, end) for a chunk given span and frequency."""
  chunk_start_sample = int(math.floor(freq * start_sec))
  chunk_end_sample = int(math.floor(freq * duration_sec + chunk_start_sample))
  return (chunk_start_sample, chunk_end_sample)


def InitDataTableInputsWithTimeAxis(freq, chunk_duration_sec, chunk_start,
                                    max_samples):
  """Initializes time axis of the DataTable inputs for given time chunk.

  Args:
    freq: Maximum sampling frequency of data to be added to graph.
    chunk_duration_sec: Number of seconds of data to load the table with.
    chunk_start: The time in seconds that the chunk will start at.
    max_samples: Maximum number of samples per series in a DataTable
  Returns:
    output_data: Data to be graphed.
  """
  start, end = GetSampleRange(freq, chunk_duration_sec, chunk_start)
  subsampling = GetSubsamplingRate(end - start, max_samples)
  output_data = [{'seconds': float(i)/freq}
                 for i in xrange(start, end, subsampling)]
  return (output_data, freq / subsampling)


def ConvertToDataTableJSon(output_data, columns_order):
  """Loads data into DataTable format.

  Args:
    output_data: Table data to be loaded into a DataTable object.
    columns_order: Ordered list of the column names.
  Returns:
    JSON format DataTable object to send to client.
  Raises:
    ValueError: Given empty dataset to load the DataTable with.
  """
  if len(output_data) < 1:
    raise ValueError('No data given to the load the DataTable')
  description = {col: ('number', col) for col in columns_order}
  data_table = gviz_api.DataTable(description)
  data_table.LoadData(output_data)
  return data_table.ToJSon(columns_order=columns_order)


def CreateEmptyTable(duration, max_samples, start=0):
  """Creates a DataTable with only the time axis intialized."""
  output_data, _ = InitDataTableInputsWithTimeAxis(1, duration + 1, start,
                                                   max_samples)
  return ConvertToDataTableJSon(output_data, ['seconds'])
