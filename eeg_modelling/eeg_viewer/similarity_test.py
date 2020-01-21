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

# Lint as: python2, python3
"""Tests for similarity operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from eeg_modelling.eeg_viewer import similarity
from eeg_modelling.pyprotos import similarity_pb2

sampling_freq = 120
n_leads = 18
total_seconds = 20
n_samples = total_seconds * sampling_freq


def overlaps(pattern_a, pattern_b):
  """Returns a boolean indicating if two patterns are overlapped.

  Args:
    pattern_a: SimilarPattern or TimeSpan.
    pattern_b: SimilarPattern or TimeSpan.
  Returns:
    boolean indicating if the patterns are overlapped.
  """
  start_a = pattern_a.start_time
  start_b = pattern_b.start_time

  end_a = pattern_a.start_time + pattern_a.duration
  end_b = pattern_b.start_time + pattern_b.duration

  a_falls_in_b = start_b < end_a and end_a < end_b
  b_falls_in_a = start_a < end_b and end_b < end_a

  return a_falls_in_b or b_falls_in_a


def set_slice_value(base_array, new_value_array, start_seconds, samp_freq):
  """Sets a slice of a numpy array with a new_value_array.

  Helper function used in the tests. It modifies the base_array in place.

  Args:
    base_array: numpy array of shape (n_channels, n_samples).
    new_value_array: numpy array of shape (n_channels, m_samples),
      where m_samples <= n_samples.
    start_seconds: starting seconds to set the new_value_array into the
      base_array.
    samp_freq: sampling frequency used in the data.
  """

  _, m_samples = new_value_array.shape

  start_index = int(start_seconds * samp_freq)
  end_index = start_index + m_samples
  base_array[:, start_index:end_index] = new_value_array


class SimilarityTest(absltest.TestCase):

  def setUp(self):
    super(SimilarityTest, self).setUp()

    self.base_data = np.zeros((n_leads, n_samples), dtype=np.float32)

  def testCreateSimilarPatternsResponse(self):
    settings = similarity_pb2.SimilaritySettings()
    settings.top_n = 7
    settings.merge_close_results = False

    response = similarity.CreateSimilarPatternsResponse(self.base_data,
                                                        1,
                                                        2,
                                                        [],
                                                        sampling_freq,
                                                        settings)

    self.assertIsInstance(response, similarity_pb2.SimilarPatternsResponse)
    self.assertLen(response.similar_patterns, 7)

  def testSearchSimilarPatterns(self):
    template_start_time = 1
    template_duration = 2
    template = np.ones((n_leads, template_duration * sampling_freq))

    set_slice_value(self.base_data, template, template_start_time,
                    sampling_freq)

    target_start_time = 5
    set_slice_value(self.base_data, template, target_start_time, sampling_freq)

    patterns_found = similarity.SearchSimilarPatterns(self.base_data,
                                                      template_start_time,
                                                      template_duration,
                                                      [],
                                                      sampling_freq,
                                                      top_n=3)
    self.assertLen(patterns_found, 3)

    target_similar_pattern = similarity_pb2.SimilarPattern()
    target_similar_pattern.score = 1
    target_similar_pattern.start_time = target_start_time
    target_similar_pattern.duration = template_duration
    self.assertIn(target_similar_pattern, patterns_found)

  def testSearchSimilarPatterns_ignoreSeen(self):
    template_start_time = 1
    template_duration = 1

    seen_event = similarity_pb2.TimeSpan()
    seen_event.start_time = 10
    seen_event.duration = 2.5

    patterns_found = similarity.SearchSimilarPatterns(self.base_data,
                                                      template_start_time,
                                                      template_duration,
                                                      [seen_event],
                                                      sampling_freq,
                                                      top_n=10)

    for pattern in patterns_found:
      end_time = pattern.start_time + pattern.duration
      message = 'Overlaps with event between %s-%s' % (pattern.start_time,
                                                       end_time)
      self.assertFalse(overlaps(seen_event, pattern), message)

  def testSearchSimilarPatterns_merge(self):
    template_start_time = 1
    template_duration = 2
    template = np.ones((n_leads, template_duration * sampling_freq))

    template_span = similarity_pb2.TimeSpan()
    template_span.start_time = template_start_time
    template_span.duration = template_duration
    seen_events = [template_span]

    set_slice_value(self.base_data, template, template_start_time,
                    sampling_freq)

    target_1_start_time = 5
    set_slice_value(self.base_data, template, target_1_start_time,
                    sampling_freq)

    target_2_start_time = 8.5
    set_slice_value(self.base_data, template, target_2_start_time,
                    sampling_freq)

    patterns_found = similarity.SearchSimilarPatterns(self.base_data,
                                                      template_start_time,
                                                      template_duration,
                                                      seen_events,
                                                      sampling_freq,
                                                      top_n=2,
                                                      merge_close_results=True,
                                                      merge_threshold=2)

    target_2_end_time = target_2_start_time + template_duration
    merged_duration = target_2_end_time - target_1_start_time

    merged_targets_span = similarity_pb2.TimeSpan()
    merged_targets_span.start_time = target_1_start_time
    merged_targets_span.duration = merged_duration
    self.assertTrue(overlaps(merged_targets_span, patterns_found[0]))

  def testCreateSimilarityCurveResponse(self):
    response = similarity.CreateSimilarityCurveResponse(self.base_data,
                                                        1,
                                                        2,
                                                        sampling_freq)

    self.assertIsInstance(response, similarity_pb2.SimilarityCurveResponse)
    self.assertLen(response.scores, self.base_data.shape[1])

if __name__ == '__main__':
  absltest.main()
