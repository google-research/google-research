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

"""Tests for felix.pointing_converter."""
import random
import string

from absl.testing import absltest
from absl.testing import parameterized

from felix import pointing_converter


class PointingConverterTest(parameterized.TestCase):

  @parameterized.parameters(
      # A simple test.
      {
          'input_texts': 'A B C D'.split(),
          'target': 'A B C D',
          'phrase_vocabulary': ['and'],
          'target_points': [1, 2, 3, 0],
          'target_phrase': ['', '', '', '']
      },
      # Missing a middle token
      {
          'input_texts': 'A B D'.split(),
          'target': 'A b C D',
          'phrase_vocabulary': ['c'],
          'target_points': [1, 2, 0],
          'target_phrase': ['', 'c', '']
      },

      # An additional source token
      {
          'input_texts': 'A B E D'.split(),
          'target': 'A B D',
          'phrase_vocabulary': ['and'],
          'target_points': [1, 3, 0, 0],
          'target_phrase': ['', '', '', '']
      },

      # Missing a middle token + an additional source token
      {
          'input_texts': 'A B E D'.split(),
          'target': 'A B C D',
          'phrase_vocabulary': ['c'],
          'target_points': [1, 3, 0, 0],
          'target_phrase': ['', 'c', '', '']
      },

      # duplicate target token
      {
          'input_texts': 'A B E D'.split(),
          'target': 'A B C D D',
          'phrase_vocabulary': ['c'],
          'target_points': [],
          'target_phrase': []
      },
      # duplicate target and source token
      {
          'input_texts': 'A D B E D'.split(),
          'target': 'A B D D',
          'phrase_vocabulary': ['c'],
          'target_points': [2, 4, 1, 0, 0],
          'target_phrase': ['', '', '', '', '']
      })
  def test_matching_conversion(self, input_texts, target, phrase_vocabulary,
                               target_points, target_phrase):

    converter = pointing_converter.PointingConverter(phrase_vocabulary)
    points = converter.compute_points(input_texts, target)

    if not points:
      self.assertEqual(points, target_phrase)
      self.assertEqual(points, target_points)
    else:
      self.assertEqual([x.added_phrase for x in points], target_phrase)
      self.assertEqual([x.point_index for x in points], target_points)

  def test_no_match(self):
    input_texts = 'Turing was born in 1912 . Turing died in 1954 .'.split()
    target = 'Turing was born in 1912 and died in 1954 .'

    phrase_vocabulary = ['but']
    converter = pointing_converter.PointingConverter(phrase_vocabulary)
    points = converter.compute_points(input_texts, target)
    # Vocabulary doesn't contain "and" so the inputs can't be converted to the
    # target.
    self.assertEqual(points, [])

  def test_match(self):
    input_texts = 'Turing was born in 1912 . Turing died in 1954 .'.split()
    target = 'Turing was born in 1912 and died in 1954 .'

    phrase_vocabulary = ['but', 'KEEP|and']
    converter = pointing_converter.PointingConverter(phrase_vocabulary)
    points = converter.compute_points(input_texts, target)
    # Vocabulary match.
    target_points = [1, 2, 3, 4, 7, 0, 0, 8, 9, 10, 0]
    target_phrases = ['', '', '', '', 'and', '', '', '', '', '', '']
    self.assertEqual([x.point_index for x in points], target_points)
    self.assertEqual([x.added_phrase for x in points], target_phrases)

  ## Unlimited vocab
  def test_match_all(self):
    random.seed(1337)
    phrase_vocabulary = set()
    converter = pointing_converter.PointingConverter(phrase_vocabulary)
    for _ in range(10):
      input_texts = [
          random.choice(string.ascii_uppercase + string.digits)
          for _ in range(10)
      ]
      ## One token needs to match.
      input_texts.append('eos')
      target = ' '.join([
          random.choice(string.ascii_uppercase + string.digits)
          for _ in range(11)
      ])
      target += ' eos'
      points = converter.compute_points(input_texts, target)
      self.assertTrue(points)


if __name__ == '__main__':
  absltest.main()
