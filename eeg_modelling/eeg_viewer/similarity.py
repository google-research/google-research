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

"""Handles similar patterns operations.

Provide functions to search similar patterns within a waveforms file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from eeg_modelling.pyprotos import similarity_pb2
import cv2


def _FilterOverlappedResults(sims, target_start_index, target_duration_index):
  """Filters out the similar patterns overlapped with the target pattern.

  Args:
    sims: Array of similarity scores.
    target_start_index: The start index of the target pattern.
    target_duration_index: The duration of the target pattern (measured in
      amount of array elements, not in seconds).
  Returns:
    Array of similarity scores without the ones that overlap with the target.
  """
  filter_start = max(target_start_index - target_duration_index + 1, 0)
  filter_end = min(target_start_index + target_duration_index, len(sims))

  return sims[:filter_start] + sims[filter_end:]


def _GetTopNonOverlappingResults(sims, top_n, duration_index):
  """Retrieves the top n similarity scores that do not overlap each other.

  Args:
    sims: Array of pairs (index, sim_score).
    top_n: Number of top scores to retrieve.
    duration_index: Duration of the target pattern (measured in amount of
      array elements, not in seconds).
  Returns:
    Array of pairs (index, sim_score), considering just the top_n scores.
  """

  def _IsBetween(target, left, right):
    """Indicate if target is between left and right."""
    return left <= target and target <= right

  top_sims = []
  for index, score in sims:
    left = index - duration_index + 1
    right = index + duration_index - 1

    overlap = any(_IsBetween(index, left, right) for index, _ in top_sims)

    if not overlap:
      top_sims.append((index, score))
      if len(top_sims) >= top_n:
        break

  return top_sims


def SearchSimilarPatterns(full_data,
                          window_start,
                          window_duration,
                          sampling_freq=200,
                          top_n=5):
  """Searches similar patterns for a target window in a 2d array.

  Args:
    full_data: numpy array that holds the full data to analyze.
      Must have shape (n_channels, n_data_points).
    window_start: the start of the window in seconds.
    window_duration: the duration of the window in seconds.
    sampling_freq: sampling frequency used in the data.
    top_n: Amount of similar results to return.
  Returns:
    Array of SimilarPattern proto objects, holding the most similar patterns
      found.
  """
  window_start_index = int(sampling_freq * window_start)

  window_duration_index = int(sampling_freq * window_duration)
  window_end_index = window_start_index + window_duration_index

  window_data = full_data[:, window_start_index:window_end_index]

  _, n_samples = window_data.shape
  if window_end_index <= window_start_index or n_samples == 0:
    raise ValueError(
        'Window must have positive duration: found %s-%s, (%s samples)' %
        (window_end_index, window_start_index, n_samples))

  sims = cv2.matchTemplate(full_data, window_data, cv2.TM_CCORR_NORMED)[0]

  sims = list(enumerate(sims))
  sims = _FilterOverlappedResults(sims, window_start_index,
                                  window_duration_index)
  sims = sorted(sims, key=lambda x: x[1], reverse=True)
  sims = _GetTopNonOverlappingResults(sims, top_n, window_duration_index)

  sim_patterns = []

  for index, score in sims:
    sim_pattern = similarity_pb2.SimilarPattern()
    sim_pattern.score = score
    sim_pattern.duration = window_duration
    sim_pattern.start_time = float(index) / sampling_freq

    sim_patterns.append(sim_pattern)

  return sim_patterns


def CreateSimilarPatternsResponse(array, start_time, duration, sampling_freq):
  """Searches similar patterns in an array of data.

  Args:
    array: Numpy array with shape (n_channels, n_data).
    start_time: seconds to start the window.
    duration: duration of the window.
    sampling_freq: Sampling frequency used in the data, in hz.
  Returns:
    SimilarPatternsResponse with the results found.
  """

  response = similarity_pb2.SimilarPatternsResponse()

  similar_patterns = SearchSimilarPatterns(
      array,
      start_time,
      duration,
      sampling_freq=sampling_freq)
  response.similar_patterns.extend(similar_patterns)

  return response
