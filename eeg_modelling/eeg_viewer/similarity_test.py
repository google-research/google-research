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


class SimilarityTest(absltest.TestCase):

  def setUp(self):
    super(SimilarityTest, self).setUp()

    self.base_data = np.zeros((n_leads, n_samples), dtype=np.float32)

  def testCreateResponse(self):
    response = similarity.CreateSimilarPatternsResponse(self.base_data,
                                                        1,
                                                        2,
                                                        [],
                                                        sampling_freq)

    self.assertIsInstance(response, similarity_pb2.SimilarPatternsResponse)
    self.assertLen(response.similar_patterns, 5)

  def testSearchSimilarPatterns(self):
    template_start_time = 1
    template_duration = 2
    template_end_time = template_start_time + template_duration

    template = np.ones((n_leads, template_duration * sampling_freq))

    template_start_index = template_start_time * sampling_freq
    template_end_index = template_end_time * sampling_freq
    self.base_data[:, template_start_index:template_end_index] = template

    target_start_time = 5
    target_end_time = target_start_time + template_duration

    target_start_index = target_start_time * sampling_freq
    target_end_index = target_end_time * sampling_freq
    self.base_data[:, target_start_index:target_end_index] = template

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


if __name__ == '__main__':
  absltest.main()
