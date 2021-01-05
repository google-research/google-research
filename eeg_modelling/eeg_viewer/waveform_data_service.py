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

"""Contains methods for packaging data from a DataSource for the API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np

from scipy import signal

from eeg_modelling.eeg_viewer import utils
from eeg_modelling.pyprotos import data_pb2

# The double banana refers to a common montage used to display EEG data.
# Each tuple represents a 'standard' in the montage, which is a subtraction of
# the signals from two EEG leads placed on the scalp.
# Elements containing '|' allow for differences in lead naming conventions
# between datasets.
_DOUBLE_BANANA = [('FP1', 'F7'), ('F7', 'T3|T7'), ('T3|T7', 'T5|P7'),
                  ('T5|P7', 'O1'),
                  ('FP2', 'F8'), ('F8', 'T4|T8'), ('T4|T8', 'T6|P8'),
                  ('T6|P8', 'O2'),
                  ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
                  ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
                  ('FZ', 'CZ'), ('CZ', 'PZ'),
                  ('EKG1', 'EKG2')]

# The standard set of leads for a 12 lead ECG
_ECG_12_LEAD = [('I',), ('II',), ('III',), ('AVR',), ('AVL',), ('AVF',),
                ('V1',), ('V2',), ('V3',), ('V4',), ('V5',), ('V6',)]


def _FilterData(row_data, index, data_source, low_cut, high_cut, notch):
  """Runs full segment data through low and high pass filters.

  Args:
    row_data: Full segment data for a single channel.
    index: The index for a single channel.
    data_source: The DataSource for the waveform data.
    low_cut: lower frequency to apply a band-pass filter.
    high_cut: higher frequency to apply a band-pass filter.
    notch: frequency to apply a notch filter.
  Returns:
    Filtered input row data.
  """
  nyquist_freq = data_source.GetChannelSamplingFrequency(index) / 2
  low_val = low_cut / nyquist_freq
  high_val = high_cut / nyquist_freq
  notch_val = notch / nyquist_freq
  pad_len = int(nyquist_freq) if int(nyquist_freq) else 1
  padded_data = np.pad(row_data, (pad_len, pad_len), 'symmetric')
  if low_val > 0 and low_val < 1:
    # Using a 1st-order forward pass filter to match NK viewer
    b, a = signal.butter(1, [low_val], btype='high', analog=False)
    padded_data = signal.lfilter(b, a, padded_data)
  if high_val > 0 and high_val < 1:
    # Using a 1st-order forward pass filter to match NK viewer
    b, a = signal.butter(1, [high_val], btype='low', analog=False)
    padded_data = signal.lfilter(b, a, padded_data)
  if notch_val > 0 and notch_val < 1:
    b, a = signal.iirnotch(notch_val, 30)
    padded_data = signal.lfilter(b, a, padded_data)
  return padded_data[pad_len:-pad_len]


def _GetChannelIndicesInChannelDataIdList(id_list):
  """Returns a list of all the unique channel indices requested."""
  channel_indices = []
  for ch in id_list:
    if ch.HasField('bipolar_channel'):
      request_indices = [
          ch.bipolar_channel.index, ch.bipolar_channel.referential_index
      ]
    else:
      request_indices = [ch.single_channel.index]
    channel_indices = channel_indices + request_indices
  return list(set(channel_indices))


def _CreateChannelData(data_source,
                       channel_data_ids,
                       low_cut,
                       high_cut,
                       notch,
                       start=0,
                       duration=None,
                       max_samples=None):
  """Returns a list of channel names and a dictionary of their values.

  Args:
    data_source: The DataSource for the waveform data.
    channel_data_ids: ChannelDataIds list.
    low_cut: lower frequency to apply a band-pass filter.
    high_cut: higher frequency to apply a band-pass filter.
    notch: frequency to apply a notch filter.
    start: start time to crop the data, relative to the start of the file (in
      seconds). Defaults to the start of the file.
    duration: duration to crop from the data, in seconds. If None, will get the
      whole file data.
    max_samples: The maximum number of samples in one channel response.
      If None, there is no maximum limit.
  Returns:
    A dictionary of channel names mapped to the requested time slice of their
    data and an ordered list of the channel names.
  Raises:
    ValueError: Too many feature keys provided (only handles raw features or
    subtraction of two features).
  """
  if duration is None:
    duration = data_source.GetLength()

  channel_indices = _GetChannelIndicesInChannelDataIdList(channel_data_ids)
  single_channel_data = data_source.GetChannelData(
      channel_indices, start, duration)

  subsampling = 1 if max_samples is None else utils.GetSubsamplingRate(
      len(list(single_channel_data.values())[0]), max_samples)

  def _GetFilteredData(index):
    """Wrapper to call _FilterData function.

    Args:
      index: the index for the selected channel.
    Returns:
      Filtered data for the selected channel.
    """
    return _FilterData(single_channel_data[str(index)],
                       index,
                       data_source,
                       low_cut,
                       high_cut,
                       notch)

  req_channel_data = {}
  channel_names = []
  for channel_data_id in channel_data_ids:
    if channel_data_id.HasField('bipolar_channel'):
      primary_index = channel_data_id.bipolar_channel.index
      primary_data = _GetFilteredData(primary_index)
      ref_index = channel_data_id.bipolar_channel.referential_index
      ref_data = _GetFilteredData(ref_index)
      channel_data = [reference - primary for (primary, reference) in
                      zip(primary_data, ref_data)]
      channel_name = '-'.join(data_source.GetChannelName(index)
                              for index in [primary_index, ref_index])
    elif channel_data_id.HasField('single_channel'):
      index = channel_data_id.single_channel.index
      channel_data = _GetFilteredData(index)
      channel_name = data_source.GetChannelName(index)
    else:
      raise ValueError('Unfamiliary channel type %s' % channel_data_id)
    req_channel_data[channel_name] = channel_data[::subsampling]
    channel_names.append(channel_name)

  return req_channel_data, channel_names


def _AddDataTableSeries(channel_data, output_data):
  """Adds series to the DataTable inputs.

  Args:
    channel_data: A dictionary of channel names to their data.  Each value in
    the dictionary has the same sampling frequency and the same time slice.
    output_data: Current graph data for DataTable API.
  Returns:
    The edited output_data dictionary where the first index represents the
    time axis value and the second the series value.
  """
  for i in range(len(list(channel_data.values())[0])):
    output_data[i].update({channel_name: data[i]
                           for channel_name, data in channel_data.items()})
  return output_data


def GetSamplingFrequency(data_source, channel_data_ids):
  """Returns the sampling frequency for a group of channels.

  Args:
    data_source: DataSource instance.
    channel_data_ids: Channels to get the sampling freq from.
  Returns:
    Sampling frequency for all the channels (must be the same).
  """
  channel_indices = _GetChannelIndicesInChannelDataIdList(channel_data_ids)
  return data_source.GetSamplingFrequency(channel_indices)


def _CreateChunkDataTableJSon(data_source, request, max_samples):
  """Creates a DataTable in JSON format which contains the data specified.

  Data can be specified by a list of minuends and subtrahends of montage
  standards and/or a list of channel keys.
  Args:
    data_source: The DataSource for the waveform data.
    request: A DataRequest proto instance.
    max_samples: The maximum number of samples in one channel response.
  Returns:
    JSON format DataTable loaded with montage data.
  Raises:
    ValueError: The requested channels have multiple frequency types.
  """

  sample_freq = GetSamplingFrequency(data_source, request.channel_data_ids)

  # Initialize Dygraph data with a time axis of sampling frequency.
  output_data, _ = utils.InitDataTableInputsWithTimeAxis(
      sample_freq, request.chunk_duration_secs, request.chunk_start,
      max_samples)
  columns_order = ['seconds']

  channel_data, channel_names = _CreateChannelData(
      data_source,
      request.channel_data_ids,
      request.low_cut,
      request.high_cut,
      request.notch,
      start=request.chunk_start,
      duration=request.chunk_duration_secs,
      max_samples=max_samples)
  output_data = _AddDataTableSeries(channel_data, output_data)
  columns_order.extend(channel_names)

  return (utils.ConvertToDataTableJSon(output_data, columns_order),
          sample_freq)


def GetMetadata(data_source, max_samples):
  """Returns metadata consistent across the predictions.

  Args:
    data_source: The DataSource for the waveform data.
    max_samples: The maximum number of samples in one channel response.
  Returns:
    A PredictionMetadata instance filled with PredictionOutput data.
  """
  response = data_pb2.WaveformMetadata()
  response.abs_start = data_source.GetStartTime()
  response.labels.extend(data_source.GetAnnotations())
  for index, channel in data_source.GetChannelIndexDict().iteritems():
    response.channel_dict[index] = channel
  response.file_type = data_source.GetFileType()
  response.nav_timeline_datatable = utils.CreateEmptyTable(
      data_source.GetLength(), max_samples)
  response.num_secs = data_source.GetLength()
  response.patient_id = data_source.GetPatientId()
  response.sstable_key = data_source.GetFileKey()
  return response


def _GetChannelIndexFromNameOptions(channel_opts, data_source):
  indices = [data_source.GetChannelIndexFromName(opt)
             for opt in channel_opts.split('|')
             if data_source.GetChannelIndexFromName(opt)]
  return indices[0] if indices else None


def _GetChannelDataIdFromNameOptions(channel_name, data_source):
  """Creates a ChannelDataId for a channel name string with name options.

  Sometimes channel naming conventions for the same electrode placement vary
  between institutions, so the options allow us to cover all cases.
  Args:
    channel_name: A tuple of strings with channel name options joined on '|'.
    data_source: The DataSource for the waveform data.
  Returns:
    A ChannelDataId filled out with the indices for the given name tuple.
  """
  channel_id = None
  if len(channel_name) == 2:
    primary_index = _GetChannelIndexFromNameOptions(channel_name[0],
                                                    data_source)
    ref_index = _GetChannelIndexFromNameOptions(channel_name[1], data_source)
    if primary_index is not None and ref_index is not None:
      channel_id = data_pb2.ChannelDataId()
      channel_id.bipolar_channel.index = int(primary_index)
      channel_id.bipolar_channel.referential_index = int(ref_index)
  if len(channel_name) == 1:
    index = _GetChannelIndexFromNameOptions(channel_name[0], data_source)
    if index is not None:
      channel_id = data_pb2.ChannelDataId()
      channel_id.single_channel.index = int(index)
  return channel_id


def _GetDefaultChannelDataIdList(data_source):
  """Returns the list of default features when a request does not specify.

  When a data request is made for the first time with a set of file
  parameters, the client does not have the lookup table with the channel
  indices, therefore the client cannot specify channels until after the
  initial load.  To deal with this case, when no channel indices are provided,
  we generate a list of channel indices using the lookup table and default
  channel requests that are hardcoded for each medical waveform data type.
  Those channel indices will be used as the request indices.
  Args:
    data_source: The DataSource for the waveform data.
  """
  default_channel_names = []
  if data_source.GetFileType() == 'EEG':
    default_channel_names = _DOUBLE_BANANA
  elif (data_source.GetFileType() == 'EKG' or
        data_source.GetFileType() == 'ECG'):
    default_channel_names = _ECG_12_LEAD

  default_channel_ids = [_GetChannelDataIdFromNameOptions(x, data_source)
                         for x in default_channel_names]

  return [channel_id for channel_id in default_channel_ids if channel_id]


def GetChunk(data_source, request, max_samples):
  """Returns all graph data for current chunk.

  Args:
    data_source: The DataSource for the waveform data.
    request: A DataRequest proto instance.
    max_samples: The maximum number of samples in one channel response.
  Returns:
    A WaveformChunkResponse specified by the Request proto.
  Raises:
    ValueError: If chunk duration is not a positive integer.
  """
  if (request.chunk_start >= data_source.GetLength() or
      request.chunk_start + request.chunk_duration_secs <= 0):
    raise ValueError('Chunk starting at %s is out of bounds'
                     % request.chunk_start)
  if not request.channel_data_ids:
    default_channels = _GetDefaultChannelDataIdList(data_source)
    logging.info('Loading default channels')
    request.channel_data_ids.extend(default_channels)
  response = data_pb2.WaveformChunk()
  waveform_datatable, sampling_freq = _CreateChunkDataTableJSon(data_source,
                                                                request,
                                                                max_samples)
  response.waveform_datatable = waveform_datatable
  response.sampling_freq = sampling_freq
  response.channel_data_ids.extend(request.channel_data_ids)
  return response


def GetChunkDataAsNumpy(data_source,
                        channel_data_ids,
                        low_cut,
                        high_cut,
                        notch):
  """Extract data from a data source as a numpy array.

  Args:
    data_source: A DataSource instance.
    channel_data_ids: ChannelDataIds list.
    low_cut: lower frequency to apply a band-pass filter.
    high_cut: higher frequency to apply a band-pass filter.
    notch: frequency to apply a notch filter.
  Returns:
    Numpy array of shape (n_channels, n_data) with the waveform data.
  """
  channel_data, channel_names = _CreateChannelData(data_source,
                                                   channel_data_ids, low_cut,
                                                   high_cut, notch)

  data = [channel_data[channel_name] for channel_name in channel_names]
  data = np.array(data, dtype=np.float32)

  return data
