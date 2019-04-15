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

"""Contains DataSources that extract channel data from various file types."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import re

from google.protobuf import timestamp_pb2
from eeg_modelling.eeg_viewer import lookup
from eeg_modelling.eeg_viewer import utils
from eeg_modelling.pyprotos import data_pb2
from eeg_modelling.pyprotos import event_pb2


class DataSource(object):
  """Source of waveform data for WaveformsExample to create data responses."""

  _CHANNEL_MATCHERS = []

  def __init__(self, file_type, key):
    self._file_type = file_type
    self._key = key
    matchers = [matcher for matcher_pair in self._CHANNEL_MATCHERS
                for matcher in matcher_pair[0]]
    self._lookup = lookup.Lookup(self.GetChannelList(), matchers)

  def GetFileKey(self):
    """Returns the key or ID used to access the given file."""
    return self._key

  def GetFileType(self):
    """Returns the waveform type in the file."""
    return self._file_type

  @abc.abstractmethod
  def GetChannelList(self):
    """Returns the list of features in the file."""
    pass

  @abc.abstractmethod
  def GetLength(self):
    """Returns number of seconds of data in the source."""
    pass

  @abc.abstractmethod
  def GetStartTime(self):
    """Get global start time of the file."""
    pass

  def GetSamplingFrequency(self, channel_indices):
    """Returns the sampling frequency for a group of channels.

    Args:
      channel_indices: List of channels indices to get the sampling freq from.
    Returns:
      Sampling frequency for all the channels (must be the same).
    Raises:
      ValueError: if the channels don't have the same frequency.
    """
    freqs = list(set(
        self.GetChannelSamplingFrequency(index)
        for index in channel_indices
    ))
    if len(freqs) != 1:
      raise ValueError('The requested channels do not have the same frequency')

    return freqs[0]

  @abc.abstractmethod
  def GetChannelSamplingFrequency(self, index):
    """Get the frequency of the data with the given index."""
    pass

  @abc.abstractmethod
  def GetChannelData(self, index_list, start, duration):
    """Returns the feature data associated with the given index.

    Args:
      index_list: The numerical indices for the requested channels.
      start: The start of the requested slice in seconds relative to the
      beginning of the data.
      duration: The duration of the requested slice in seconds.
    """
    pass

  def GetChannelName(self, index):
    """Returns the feature name for display that maps to index."""
    return self._lookup.GetShorthandFromIndex(index)

  def GetChannelIndexFromKey(self, key):
    """Returns the numerical index associated with the channel key.

    The key given corresponds to a single key for the given channel data in the
    data structure.  The index is the assigned numerical index generated for the
    data source on construction by the 3-way lookup (e.g. if the TF Example key
    is 'eeg_channel/EEG FP1-REF/samples', this function might return 21).
    Args:
      key: The key used to access the channel in the underlying data structure.
    """
    return self._lookup.GetIndexFromKey(key)

  def GetChannelIndexFromName(self, name):
    """Returns the numerical index associated with the channel name."""
    return self._lookup.GetIndexFromShorthand(name)

  @abc.abstractmethod
  def GetAnnotations(self):
    """Returns a list of Waveforms Viewer Annotations."""

  @abc.abstractmethod
  def GetPatientId(self):
    """Returns the patient ID."""

  def GetChannelIndexDict(self):
    """Dict of available features to render in the data source.

    Each key is an index and its value is the shorthand name for the feature.
    Returns:
      Dictionary that maps between the feature index and shorthand.
    """
    return self._lookup.GetIndexToShorthandDict()


class TfExampleDataSource(DataSource):
  """DataSource that extracts data from a TF Example proto instance."""

  # These values are keys that will always be present in a TF Example from the
  # Medical Waveforms sandbox
  _NUM_SAMPLES_KEY = 'eeg_channel/num_samples'
  _FREQ_KEY = 'eeg_channel/sampling_frequency_hz'
  _RESAMPLED_NUM_SAMPLES_KEY = 'eeg_channel/resampled_num_samples'
  _RESAMPLED_FREQ_KEY = 'eeg_channel/resampled_sampling_frequency_hz'

  _START_TIME_KEY = 'start_time'
  _PATIENT_ID_KEY = 'segment/patient_id'
  _ANNOTATIONS_KEY = 'raw_label_events'

  def __init__(self, tf_example, key, file_type):
    self._tf_example = tf_example
    self._feature = self._tf_example.features.feature
    super(TfExampleDataSource, self).__init__(file_type, str(key))

  def GetChannelList(self):
    return self._feature.keys()

  def GetLength(self):
    num_samples = self._feature[self._NUM_SAMPLES_KEY].int64_list.value[0]
    sample_freq = self._feature[self._FREQ_KEY].float_list.value[0]
    return math.ceil(float(num_samples) / sample_freq)

  def GetStartTime(self):
    start_timestamp = timestamp_pb2.Timestamp.FromString(
        self._feature[self._START_TIME_KEY].bytes_list.value[0])
    return utils.TimestampPb2ToSeconds(start_timestamp)

  def GetChannelSamplingFrequency(self, index):
    key = self._lookup.GetKeyFromIndex(index)
    for matcher_set, freq_key in self._CHANNEL_MATCHERS:
      if any(matcher.match(key) for matcher in matcher_set) and freq_key:
        return self._feature[freq_key].float_list.value[0]
    return 1

  def GetChannelData(self, index_list, start, duration):
    freq = self.GetSamplingFrequency(index_list)

    chunk_start_index, chunk_end_index = utils.GetSampleRange(freq, duration,
                                                              start)
    channel_dict = {}
    for index in index_list:
      key = self._lookup.GetKeyFromIndex(index)
      if self._feature[key].HasField('float_list'):
        channel_data = self._feature[key].float_list.value
      else:
        raise ValueError('Channel %s is not a float value.' % key)
      channel_dict[str(index)] = channel_data[chunk_start_index:chunk_end_index]
    return channel_dict

  def GetAnnotations(self):
    annotation_strings = self._feature[self._ANNOTATIONS_KEY].bytes_list.value
    annotations = []
    for annotation_string in annotation_strings:
      event = event_pb2.Event.FromString(annotation_string)
      annotation = data_pb2.WaveformMetadata.Label()
      annotation.label_text = event.label
      annotation.start_time = event.start_time_sec
      annotations.append(annotation)
    return annotations

  def GetPatientId(self):
    return self._feature[self._PATIENT_ID_KEY].bytes_list.value[0]


class TfExampleEegDataSource(TfExampleDataSource):
  """Data source that extracts EEG data from a TF Example."""

  _CHANNEL_MATCHERS = [
      ([
          # EEG channel pattern
          re.compile(r'eeg_channel/EEG (\w+)(-\w+)*/samples'),
      ], 'eeg_channel/sampling_frequency_hz'),
      ([
          # EEG channel pattern for training data
          re.compile(r'eeg_channel/EEG (\w+)(-\w+)*/resampled_samples'),
      ], 'eeg_channel/resampled_sampling_frequency_hz'),
      ([
          # 'seizure bin' used at the shorthand for this key.
          re.compile(r'(seizure_bin)ary_per_sec'),  # Derived feature pattern.
      ], None),
  ]

  def __init__(self, tf_example, key):
    super(TfExampleEegDataSource, self).__init__(tf_example, key, 'EEG')


class TfExampleEkgDataSource(TfExampleDataSource):
  """Data source that extracts EEG data from a TF Example."""

  _CHANNEL_MATCHERS = [
      ([
          # EKG channel pattern
          re.compile(r'eeg_channel/POL (EKG\w+)/samples'),
          # ECG channel pattern
          re.compile(r'eeg_channel/(\w+)/samples')
      ], 'eeg_channel/sampling_frequency_hz'),
  ]

  def __init__(self, tf_example, key):
    super(TfExampleEkgDataSource, self).__init__(tf_example, key, 'EKG')
