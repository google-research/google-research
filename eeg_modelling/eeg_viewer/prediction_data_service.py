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

"""Handles prediction output data from Medical Waveforms experiments.

Contains the PredictionDataService class that grabs and formats prediction data
to be sent to and rendered by the client.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import logging
import numpy as np

from eeg_modelling.eeg_viewer import signal_helper
from eeg_modelling.eeg_viewer import utils
from eeg_modelling.pyprotos import data_pb2


class PredictionDataService(object):
  """Extracts experiment prediction data from a PredictionOutputs proto.

  See prediction_output.proto
  """

  def __init__(self, prediction_outputs, data_source, max_samples):
    self._prediction_outputs = prediction_outputs
    self._data_source = data_source
    self._abs_start = data_source.GetStartTime()
    self._max_samples = max_samples

  def _GetChunkTiming(self, prediction_output):
    chunk_info = prediction_output.chunk_info
    start = utils.TimestampPb2ToSeconds(chunk_info.chunk_start_time)
    start = round(start - self._abs_start)
    duration = round(chunk_info.chunk_size_sec)
    return start, duration

  def _PreprocessAttributionData(self, prediction_output):
    """Thresholds and normalizes the attribution in PredictionOutput.

    Args:
      prediction_output: PredictionOutput Message.

    Returns:
      PredictionOutput Message with thresholded and normalized attribution
      values.
    """
    for label in prediction_output.label:
      if not label.HasField('attribution_map'):
        continue
      attribution = label.attribution_map.attribution
      # Threshold and normalize over flattened array
      attribution = np.absolute(np.array(attribution))
      attribution = signal_helper.threshold_attribution(attribution)
      attribution = (attribution / np.max(attribution)).tolist()
      label.attribution_map.attribution[:] = attribution

    return prediction_output

  def _ConvertChannelDataIdToIndexStr(self, channel_data_id):
    """Converts a ChannelDataId proto to a string.

    For single channels, the resulting string is the index and for bipolar
    channels, the resulting string is the primary and referential indices joined
    on '-'.  The resulting string will be compared to the list of attribution
    feature_names that has been converted to index format.
    Args:
      channel_data_id: A ChannelDataId proto.
    Returns:
      A string of channel indices joined by '-'.
    """
    if channel_data_id.HasField('bipolar_channel'):
      indices = [
          str(channel_data_id.bipolar_channel.index),
          str(channel_data_id.bipolar_channel.referential_index)
      ]
    else:
      indices = [str(channel_data_id.single_channel.index)]
    return '-'.join(indices)

  def _SliceAttributionData(self, preprocessed_prediction_output, request):
    """Slices out only the requested time frame and channels.

    Args:
      preprocessed_prediction_output: PredictionOutput Message.
      request: DataRequest instance.
    Returns:
      A dictionary of string prediction types to AttributionMaps.
    """
    map_dict = {}
    for label in preprocessed_prediction_output.label:
      if not label.HasField('attribution_map'):
        continue
      if label.name not in map_dict:
        map_dict[label.name] = data_pb2.PredictionChunk.AttributionMap()
      attribution_map = label.attribution_map
      # Get index format for feature names in the attribution map
      attr_indices = [
          '-'.join([
              self._data_source.GetChannelIndexFromKey(key)
              for key in feat.split('#')
              if self._data_source.GetChannelIndexFromKey(key)
          ])
          for feat in attribution_map.feature_names
      ]

      # Width of row if the 1D attribution map were transformed to 2D
      width = attribution_map.width
      for channel_data_id in request.channel_data_ids:
        ch_str = self._ConvertChannelDataIdToIndexStr(channel_data_id)
        if ch_str in attr_indices:
          attribution_row = attr_indices.index(ch_str)
          attr_start = width * attribution_row
          attr_end = attr_start + width
          channel_attr = attribution_map.attribution[attr_start:attr_end]
          map_dict[label.name].attribution_map[ch_str].attribution.extend(
              channel_attr)
        else:
          map_dict[label.name].attribution_map[ch_str].attribution.extend(
              [0] * width)

    return map_dict

  def _DownsampleAttributionDictionary(self, map_dict):
    """Downsample attribution by row given a max number of samples per row.

    Args:
      map_dict: A dictionary of prediction types mapped to AttributionMaps.
    Returns:
      The original map_dict with the attribution values downsampled.
    """
    sample_attribution = []
    for data_key in map_dict:
      attribution_map = map_dict[data_key].attribution_map
      for map_key in attribution_map:
        sample_attribution = attribution_map[map_key].attribution
        break
    if not sample_attribution:
      return map_dict
    ratio = int(math.ceil(float(len(sample_attribution)) / self._max_samples))

    def _Downsample(attribution):
      return signal_helper.downsample_attribution(attribution,
                                                  ratio=ratio).tolist()

    if ratio != 1:
      logging.info('Downsampling attribution at a rate of %s', ratio)
      for label in map_dict:
        for feature in map_dict[label].attribution_map:
          downsampled = _Downsample(
              map_dict[label].attribution_map[feature].attribution)
          map_dict[label].attribution_map[feature].attribution[:] = downsampled

    return map_dict

  def _FormatAttributionMap(self, prediction_output, request):
    """Formats the attribution map in the PredictionOutput proto for viewing.

    Args:
      prediction_output: PredictionOutput Message.
      request: DataRequest instance.

    Returns:
      An nested dict containing attribution lists grouped first by prediction
      label and then by input channel within each label.
    """
    if not prediction_output:
      return {}
    preprocessed = self._PreprocessAttributionData(prediction_output)
    sliced = self._SliceAttributionData(preprocessed, request)
    downsampled = self._DownsampleAttributionDictionary(sliced)
    return downsampled

  def _GetChunkScores(self, prediction_output):
    """Returns actual and predicted score for each label in a PredictionOutput.

    Args:
      prediction_output: PredictionOutput Message for one 96 second data chunk.
    Returns:
      A ChunkScoreData instance filled with PredictionOutputs data.
    """
    start, duration = self._GetChunkTiming(prediction_output)

    chunk_score_data = data_pb2.PredictionMetadata.ChunkScoreData()
    chunk_score_data.start_time = start
    chunk_score_data.duration = duration

    for label in prediction_output.label:
      score_data = chunk_score_data.score_data[label.name]
      score_data.actual_value = label.actual_value.score
      score_data.predicted_value = label.predicted_value.score
      score_data.prediction_probability = label.predicted_value.probability

    return chunk_score_data

  def GetMetadata(self):
    """Gathers prediction data consistent across a file.

    Returns:
      Chunk scores for 96-second chunks for each prediction type.
    """
    response = data_pb2.PredictionMetadata()

    response.chunk_scores.extend([self._GetChunkScores(prediction_output) for
                                  prediction_output in
                                  self._prediction_outputs.prediction_output])

    return response

  def GetChunk(self, request):
    """Returns PredictionOutputs data contained inside chunk.

    Args:
      request: DataRequest instance.

    Returns:
      A PredictionChunkResponse instance per the DataRequest.
    """
    prediction_outputs_containing_chunk = []
    # Iterating through predictions by modeling chunk
    for prediction_output in self._prediction_outputs.prediction_output:
      start, duration = self._GetChunkTiming(prediction_output)
      # Gathering only attribution maps that overlap with the requested time
      has_attr = any(label.HasField('attribution_map')
                     for label in prediction_output.label)
      if (has_attr and start <= request.chunk_start and start + duration >=
          (request.chunk_start + request.chunk_duration_secs)):
        prediction_outputs_containing_chunk.append(prediction_output)

    response = data_pb2.PredictionChunk()

    if prediction_outputs_containing_chunk:
      prediction_output = prediction_outputs_containing_chunk[0]
      for key, value in self._FormatAttributionMap(prediction_output,
                                                   request).items():
        response.attribution_data[key].CopyFrom(value)
      start, duration = self._GetChunkTiming(prediction_output)
      response.chunk_start = int(start)
      response.chunk_duration = int(duration)

    return response
