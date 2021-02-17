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

from absl.testing import absltest
from felix import converter_for_felix_insert as converter
from felix import felix_constants as constants


class FelixinsertConverterTest(absltest.TestCase):
  """Tests when `insert_after_token == True`."""

  def test_compute_edits_and_insertions(self):
    # pylint: disable=bad-whitespace
    source = ['A',      'B',  'c',  'D']
    target = ['A', 'Z', 'B',        'D', 'W']
    #          K    I  | K |   D |   K    I
    # pylint: enable=bad-whitespace
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1)
    self.assertEqual(
        edits,
        [constants.KEEP, constants.KEEP, constants.DELETE, constants.KEEP])
    self.assertEqual(insertions, [['Z'], [], [], ['W']])

  def test_compute_edits_and_insertions_for_replacement(self):
    source = ['A', 'b', 'C']
    target = ['A', 'B', 'C']

    # We should insert 'B' after 'b' (not after 'A' although the result is the
    # same).
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1)
    self.assertEqual(edits, [constants.KEEP, constants.DELETE, constants.KEEP])
    self.assertEqual(insertions, [[], ['B'], []])

  def test_compute_edits_and_insertions_for_long_insertion(self):
    # pylint: disable=bad-whitespace
    source = ['A',           'B']
    target = ['A', 'X', 'Y', 'B']
    # pylint: enable=bad-whitespace
    edits_and_insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1)
    self.assertIsNone(edits_and_insertions)
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2)
    self.assertEqual(edits, [constants.KEEP, constants.KEEP])
    self.assertEqual(insertions, [['X', 'Y'], []])

  def test_compute_edits_and_insertions_for_long_insertion_and_deletions(self):
    # pylint: disable=bad-whitespace
    source = ['A', 'b',              'c',          'D']
    target = ['A',     'X', 'Y', 'Z',    'U', 'V', 'D']
    # pylint: enable=bad-whitespace
    edits_and_insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2)
    self.assertIsNone(edits_and_insertions)
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=3)
    self.assertEqual(
        edits,
        [constants.KEEP, constants.DELETE, constants.DELETE, constants.KEEP])
    self.assertEqual(insertions, [[], ['X', 'Y', 'Z'], ['U', 'V'], []])

  def test_compute_edits_and_insertions_no_overlap(self):
    source = ['a', 'b']
    target = ['C', 'D']
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2)
    self.assertEqual(edits, [constants.DELETE, constants.DELETE])
    self.assertEqual(insertions, [['C', 'D'], []])


class FelixinsertConverterTestInsertBefore(absltest.TestCase):
  """Tests when `insert_after_token == False`."""

  def test_compute_edits_and_insertions(self):
    # pylint: disable=bad-whitespace
    source = [     'A',      'B',  'c',  'D']
    target = ['X', 'A', 'Z', 'B',        'D']
    #          I    K  | I    K |   D |   K
    # pylint: enable=bad-whitespace
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1, insert_after_token=False)
    self.assertEqual(
        edits,
        [constants.KEEP, constants.KEEP, constants.DELETE, constants.KEEP])
    self.assertEqual(insertions, [['X'], ['Z'], [], []])

  def test_compute_edits_and_insertions_for_replacement(self):
    source = ['A', 'b', 'C', 'D']
    target = ['A', 'B', 'C', 'D']

    # We should insert 'B' before 'b' (not before 'C' although the result is the
    # same).
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1, insert_after_token=False)
    self.assertEqual(
        edits,
        [constants.KEEP, constants.DELETE, constants.KEEP, constants.KEEP])
    self.assertEqual(insertions, [[], ['B'], [], []])

  def test_compute_edits_and_insertions_for_long_insertion(self):
    # pylint: disable=bad-whitespace
    source = ['A',           'B']
    target = ['A', 'X', 'Y', 'B']
    # pylint: enable=bad-whitespace
    edits_and_insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=1, insert_after_token=False)
    self.assertIsNone(edits_and_insertions)
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2, insert_after_token=False)
    self.assertEqual(edits, [constants.KEEP, constants.KEEP])
    self.assertEqual(insertions, [[], ['X', 'Y']])

  def test_compute_edits_and_insertions_for_long_insertion_and_deletions(self):
    # pylint: disable=bad-whitespace
    source = [         'a',              'b', 'C']
    target = ['X', 'Y',    'Z', 'U', 'V',     'C']
    # pylint: enable=bad-whitespace
    edits_and_insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2, insert_after_token=False)
    self.assertIsNone(edits_and_insertions)
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=3, insert_after_token=False)
    self.assertEqual(edits,
                     [constants.DELETE, constants.DELETE, constants.KEEP])
    self.assertEqual(insertions, [['X', 'Y'], ['Z', 'U', 'V'], []])

  def test_compute_edits_and_insertions_no_overlap(self):
    source = ['a', 'b']
    target = ['C', 'D']
    edits, insertions = converter.compute_edits_and_insertions(
        source, target, max_insertions_per_token=2)
    self.assertEqual(edits, [constants.DELETE, constants.DELETE])
    self.assertEqual(insertions, [['C', 'D'], []])


if __name__ == '__main__':
  absltest.main()
