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

"""Tests for NarrativeQA extractive oracle."""

from absl.testing import absltest
from readtwice.models.narrative_qa import extractive_oracle


class ExtractiveOracleTest(absltest.TestCase):

  def test_extractive_oracle(self):
    oracle = extractive_oracle.ExtractiveOracle(0, 1.0, 100)
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa'), ['aaa'])
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa bbb'), ['aaa bbb'])
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa bbb ccc'),
        ['aaa bbb ccc'])
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa ddd'), ['aaa'])
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa ddd eee'), ['aaa'])
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb aaa ccc', 'aaa bbb'),
        ['aaa bbb'])
    self.assertEqual(oracle.find_approximate_answers('aaa aaa', 'aaa'), ['aaa'])

    oracle = extractive_oracle.ExtractiveOracle(0, 0.0, 100)
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb aaa ccc', 'aaa bbb'),
        ['aaa bbb', 'aaa'])

    oracle = extractive_oracle.ExtractiveOracle(0, 1.0, 1)
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb aaa ccc', 'aaa bbb'),
        ['aaa bbb'])

    oracle = extractive_oracle.ExtractiveOracle(1.0, 1.0, 100)
    self.assertEqual(
        oracle.find_approximate_answers('aaa bbb ccc', 'aaa ddd eee'), [])

    oracle = extractive_oracle.ExtractiveOracle(0, 1.0, 100)
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the', 'is', remove_all_stopwords_answers=True), [])
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the', 'is the', remove_all_stopwords_answers=True), [])
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the blabla', 'is the', remove_all_stopwords_answers=True), [])
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the', 'is', remove_all_stopwords_answers=False), ['is'])
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the', 'is the', remove_all_stopwords_answers=False), ['is the'])
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the blabla', 'is the', remove_all_stopwords_answers=False),
        ['is the'])

    oracle = extractive_oracle.ExtractiveOracle(0, 0, 100)
    self.assertEqual(
        oracle.find_approximate_answers(
            'is the blabla', 'is the', remove_all_stopwords_answers=True),
        ['is the blabla'])


if __name__ == '__main__':
  absltest.main()
