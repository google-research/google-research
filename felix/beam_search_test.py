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

"""Tests for felix.beam_search."""
from absl.testing import absltest
from absl.testing import parameterized

from felix import beam_search


class BeamSearchTest(parameterized.TestCase):

  @parameterized.parameters(
      # Straightforward.
      {
          "good_indexes": [0, 1, 2, 3, 4, 5],
          "sep_indexes": set([5]),
          "end_index": 5,
          "predicted_points_logits": [
              [0, 10, 0, 0, 0, 0],
              [0, 0, 10, 0, 0, 0],
              [0, 0, 0, 10, 0, 0],
              [0, 0, 0, 0, 10, 0],
              [10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
          ],
          "best_sequence": [0, 1, 2, 3, 4, 5],
      },
      # Go backwards.
      {
          "good_indexes": [0, 1, 2, 3, 4, 5],
          "sep_indexes": set([5]),
          "end_index": 5,
          "predicted_points_logits": [
              [0, 0, 0, 10, 0, 0],
              [0, 0, 0, 0, 10, 0],
              [0, 10, 0, 0, 0, 0],
              [0, 0, 10, 0, 0, 0],
              [10, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
          ],
          "best_sequence": [0, 3, 2, 1, 4, 5],
      },
  )
  def test_beam_search_single_tagging(self, predicted_points_logits,
                                      good_indexes, sep_indexes, end_index,
                                      best_sequence):
    prediction = beam_search.beam_search_single_tagging(predicted_points_logits,
                                                        good_indexes,
                                                        sep_indexes, end_index)
    self.assertEqual(prediction, best_sequence)


if __name__ == "__main__":
  absltest.main()
