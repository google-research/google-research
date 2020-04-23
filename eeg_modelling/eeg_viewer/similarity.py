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

# Lint as: python2, python3
"""Handles similar patterns operations.

Provide functions to search similar patterns within a waveforms file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from eeg_modelling.pyprotos import similarity_pb2
import cv2


def _GetPatternsDistance(pattern_a, pattern_b):
  """Calculates the distance between two patterns.

  Args:
    pattern_a: a SimilarPattern or TimeSpan.
    pattern_b: a SimilarPattern or TimeSpan.
  Returns:
    Distance in seconds between the two patterns. If they are overlapped, the
      distance is 0.
  """
  end_a = pattern_a.start_time + pattern_a.duration
  end_b = pattern_b.start_time + pattern_b.duration

  b_falls_in_a = pattern_a.start_time <= end_b and end_b <= end_a
  a_falls_in_b = pattern_b.start_time <= end_a and end_a <= end_b

  if b_falls_in_a or a_falls_in_b:
    return 0
  elif pattern_a.start_time < pattern_b.start_time:
    return pattern_b.start_time - end_a
  else:
    return pattern_a.start_time - end_b


def _FilterSeenResults(similar_patterns, seen_events):
  """Filters out the similar patterns overlapped with the previous events seen.

  Note that the seconds are approximated with int(), to simplify the algorithm.
  This means that two patterns that are less than a second away will be treated
  as they are overlapped.

  Args:
    similar_patterns: Array of SimilarPatterns.
    seen_events: Array of TimeSpans.
  Returns:
    Array of SimilarPattern without the ones that overlap with the seen events.
  """
  seen_mask = set()
  for seen_event in seen_events:
    start_time = int(seen_event.start_time)
    end_time = int(seen_event.start_time + seen_event.duration)
    for second in range(start_time, end_time + 1):
      seen_mask.add(second)

  filtered_patterns = []
  for pattern in similar_patterns:
    start_time = int(pattern.start_time)
    end_time = int(pattern.start_time + pattern.duration)

    overlaps = any(
        second in seen_mask for second in range(start_time, end_time + 1))
    if not overlaps:
      filtered_patterns.append(pattern)

  return filtered_patterns


def _SortSimilarPatterns(similar_patterns):
  """Sorts a list of SimilarPatterns, leaving the highest score first."""
  return sorted(similar_patterns, key=lambda x: x.score, reverse=True)


def _MergePatterns(base_pattern, patterns):
  """Merges a list of patterns into a base pattern.

  Keeps the longest span possible and the maximum score.
  Args:
    base_pattern: a SimilarPattern instance.
    patterns: array of SimilarPatterns.
  Returns:
    A SimilarPattern with the merged information.
  """
  start_time = base_pattern.start_time
  end_time = base_pattern.start_time + base_pattern.duration
  score = base_pattern.score
  for other_pattern in patterns:
    start_time = min(start_time, other_pattern.start_time)
    end_time = max(end_time, other_pattern.start_time + other_pattern.duration)
    score = max(score, other_pattern.score)

  merged_pattern = similarity_pb2.SimilarPattern()
  merged_pattern.start_time = start_time
  merged_pattern.duration = end_time - start_time
  merged_pattern.score = score
  return merged_pattern


def _SplitOverlappingAndNonOverlapping(base_pattern, patterns, threshold):
  """Splits a list of patterns into two: overlapping or not with a base pattern.

  Args:
    base_pattern: A SimilarPattern to use as reference.
    patterns: A list of SimilarPatterns to split.
    threshold: Maximum amount of seconds to consider as overlapped between two
      patterns.
  Returns:
    Two lists of patterns, first one with the patterns that do not overlap with
      the base_pattern; second one with the patterns that do overlap.
  """

  overlapped = []
  no_overlapped = []

  for other_pattern in patterns:
    if _GetPatternsDistance(base_pattern, other_pattern) <= threshold:
      overlapped.append(other_pattern)
    else:
      no_overlapped.append(other_pattern)

  return no_overlapped, overlapped


def _GetTopNonOverlappingResults(similar_patterns,
                                 top_n,
                                 merge,
                                 merge_threshold):
  """Retrieves the top_n SimilarPattern that do not overlap each other.

  Args:
    similar_patterns: Array of SimilarPatterns.
    top_n: Number of top scores to retrieve.
    merge: Boolean indicating if merging should be performed
    merge_threshold: Number of seconds to use as threshold to decide a merge.
  Returns:
    Array of SimilarPatterns, considering just the top_n scores.
  """

  top_results = []
  for pattern in _SortSimilarPatterns(similar_patterns):
    no_overlapped, overlapped = _SplitOverlappingAndNonOverlapping(
        pattern, top_results, merge_threshold)

    if merge:
      pattern = _MergePatterns(pattern, overlapped)
      top_results = no_overlapped
    elif overlapped:
      continue

    top_results.append(pattern)
    if len(top_results) >= top_n:
      break

  # Merging could have disordered the top_results
  return _SortSimilarPatterns(top_results) if merge else top_results


def _GetSliceWindow(array, start, duration, sampling_freq):
  """Extracts a slice from an array.

  Args:
    array: numpy array of shape (n_channels, n_samples).
    start: start seconds to perform the slice.
    duration: duration in seconds of the slice.
    sampling_freq: sampling frequency of the array (Hz).

  Returns:
    Numpy array of shape (n_channels, window_samples).
  Raises:
    ValueError: if the window specified has negative or zero length.
  """
  start_index = int(sampling_freq * start)
  end_index = start_index + int(sampling_freq * duration)

  window_data = array[:, start_index:end_index]

  _, n_samples = window_data.shape
  if end_index <= start_index or n_samples == 0:
    raise ValueError(
        'Window must have positive duration: found %s-%s, (%s samples)' %
        (end_index, start_index, n_samples))

  return window_data


def SearchSimilarPatterns(full_data,
                          window_start,
                          window_duration,
                          seen_events,
                          sampling_freq,
                          top_n=5,
                          merge_close_results=False,
                          merge_threshold=1):
  """Searches similar patterns for a target window in a 2d array.

  Args:
    full_data: numpy array that holds the full data to analyze.
      Must have shape (n_channels, n_data_points).
    window_start: the start of the window in seconds.
    window_duration: the duration of the window in seconds.
    seen_events: array of TimeSpan marking events already seen.
    sampling_freq: sampling frequency used in the data.
    top_n: Amount of similar results to return.
    merge_close_results: Boolean indicating if merge between near results should
      be performed.
    merge_threshold: Amount of seconds to use as merge_threshold.
  Returns:
    Array of SimilarPatterns, holding the most similar patterns found.
  """
  window_data = _GetSliceWindow(full_data, window_start, window_duration,
                                sampling_freq)

  scores = cv2.matchTemplate(full_data, window_data, cv2.TM_CCORR_NORMED)[0]
  similar_patterns = []
  for index, score in enumerate(scores):
    sim_pattern = similarity_pb2.SimilarPattern()
    sim_pattern.score = score
    sim_pattern.duration = window_duration
    sim_pattern.start_time = float(index) / sampling_freq

    similar_patterns.append(sim_pattern)

  similar_patterns = _FilterSeenResults(similar_patterns, seen_events)
  top_results = _GetTopNonOverlappingResults(similar_patterns, top_n,
                                             merge_close_results,
                                             merge_threshold)

  return top_results


def CreateSimilarPatternsResponse(array, start_time, duration,
                                  seen_events, sampling_freq, settings):
  """Searches similar patterns in an array of data.

  Args:
    array: Numpy array with shape (n_channels, n_data).
    start_time: seconds to start the window.
    duration: duration of the window.
    seen_events: Array of TimeSpan protos, representing previously seen events.
      The algorithm will avoid results that are already seen.
    sampling_freq: Sampling frequency used in the data, in hz.
    settings: SimilaritySettings instance.
  Returns:
    SimilarPatternsResponse with the results found.
  """
  response = similarity_pb2.SimilarPatternsResponse()

  similar_patterns = SearchSimilarPatterns(
      array,
      start_time,
      duration,
      seen_events,
      sampling_freq,
      top_n=settings.top_n,
      merge_close_results=settings.merge_close_results,
      merge_threshold=settings.merge_threshold)
  response.similar_patterns.extend(similar_patterns)

  return response


def CalculateRollingMax(raw_scores, total_samples, window_samples):
  """Calculates a rolling maximum across the array, in windows of the same size.

  The scores returned from open-cv matchTemplate are calculated as if each
  value in the array matches with the first position in the window. Here,
  for each position in the original array, scores across the window are
  aggregated, to take into account the match with the first, second, third, ...
  position in the window. The aggegation method is max(), it performed better
  than avg().
  Args:
    raw_scores: array of scores calculated with open-cv matchTemplate function.
      It should have length total_samples - window_samples + 1.
    total_samples: total number of samples of the original data.
    window_samples: number of samples in the window.

  Returns:
    Array of scores calculated aggregating with the maximum.
  """
  scores = []

  left = 0
  current_max = 0
  for right in range(total_samples):
    if right < len(raw_scores):
      if raw_scores[right] > current_max:
        current_max = raw_scores[right]

    current_size = right - left + 1
    if current_size > window_samples:
      pop_value = raw_scores[left]
      left += 1

      if pop_value >= current_max:
        current_max = 0
        for c in range(left, min(right + 1, len(raw_scores))):
          if raw_scores[c] > current_max:
            current_max = raw_scores[c]
    score = current_max

    scores.append(score)

  return scores


def CalculateSimilarityCurve(full_data, window_start, window_duration,
                             sampling_freq):
  """Calculates a similarity score for a target window in a 2d array.

  Args:
    full_data: numpy array that holds the full data to analyze. Must have shape
      (n_channels, n_data_points).
    window_start: the start of the window in seconds.
    window_duration: the duration of the window in seconds.
    sampling_freq: sampling frequency used in the data.

  Returns:
    List of floats of length n_data_points, holding the scores of similarity
      across the full_data array.
  """
  window_data = _GetSliceWindow(full_data, window_start, window_duration,
                                sampling_freq)

  raw_scores = cv2.matchTemplate(full_data, window_data, cv2.TM_CCORR_NORMED)[0]

  total_samples = full_data.shape[1]
  window_size = window_data.shape[1]
  scores = CalculateRollingMax(raw_scores, total_samples, window_size)

  return scores


def CreateSimilarityCurveResponse(array, start_time, duration, sampling_freq):
  """Calculates the similarity curve across an array of data.

  Note that there could be a single request to get both similar patterns and a
  similarity curve, though as it is an experimental feature, is left separated.

  Args:
    array: Numpy array with shape (n_channels, n_data).
    start_time: seconds to start the window.
    duration: duration of the window.
    sampling_freq: Sampling frequency used in the data, in hz.

  Returns:
    SimilarityCurveResponse with the scores from the curve.
  """
  response = similarity_pb2.SimilarityCurveResponse()

  scores = CalculateSimilarityCurve(array, start_time, duration, sampling_freq)
  response.scores.extend(scores)

  return response
