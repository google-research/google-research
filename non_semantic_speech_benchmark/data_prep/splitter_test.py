# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for splitter.py."""
import functools
from absl.testing import absltest
from non_semantic_speech_benchmark.data_prep import splitter


class SplitterTest(absltest.TestCase):

  def test_compute_split_boundaries(self):
    # Split to two.
    self.assertEqual(
        splitter._compute_split_boundaries(
            [('A', 0.6), ('B', 0.4)], 10),
        [('A', 0, 6), ('B', 6, 10)]
    )

    # Split to three.
    self.assertEqual(
        splitter._compute_split_boundaries(
            [('train', 0.6), ('dev', 0.2), ('test', 0.2)], 100),
        [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)]
    )

    # Rounding number of items to nearest integer.
    self.assertEqual(
        splitter._compute_split_boundaries(
            [('train', 0.6), ('dev', 0.2), ('test', 0.2)], 9),
        [('train', 0, 5), ('dev', 5, 7), ('test', 7, 9)]
    )

    # Probs that don't sum up to 1.
    with self.assertRaises(ValueError):
      splitter._compute_split_boundaries(
          [('train', 0.6), ('dev', 0.2), ('test', 0.3)], 100)
    with self.assertRaises(ValueError):
      splitter._compute_split_boundaries([], 100)

  def test_get_inter_splits_by_group_correct_splits(self):
    items_and_groups = [  # 4 groups with 1 or 2 items each.
        (1, 'A'), (2, 'A'), (3, 'B'), (4, 'B'), (5, 'C'), (6, 'C'), (7, 'D'),
    ]
    split_probs = [('train', 0.5), ('dev', 0.25), ('test', 0.25)]

    for split_num in range(3):
      res = splitter.get_splits_by_group(
          items_and_groups, split_probs, split_num, 'inter')

      # Check we have the expected 3 splits in the result.
      self.assertEqual({'train', 'dev', 'test'}, res.keys())

      # Check the union of all splits is all items and there are no duplicates.
      self.assertCountEqual(
          [1, 2, 3, 4, 5, 6, 7],
          list(res['train']) + list(res['dev']) + list(res['test']))

      # Check items for each group appear in the same split.
      i2g = dict(items_and_groups)
      split_groups = {  # Get list of unique groups in each split.
          name: list({i2g[x] for x in items}) for name, items in res.items()
      }
      self.assertCountEqual(
          ['A', 'B', 'C', 'D'],
          split_groups['train'] + split_groups['dev'] + split_groups['test'])

      # Check number of groups in train/dev/test are as requested.
      self.assertLen(split_groups['train'], 2)
      self.assertLen(split_groups['dev'], 1)
      self.assertLen(split_groups['test'], 1)

  def test_get_intra_splits_invalid_groups(self):
    items_and_groups = []
    # 4 groups with 2, 3, 4, 5 items each.
    groups = ['A']*2 + ['B']*3 + ['C']*4 + ['D']*5
    items_and_groups = list(enumerate(groups))

    for split_num in range(3):
      two_split_groups = [(str(i), 1.0 / 2) for i in range(2)]
      res = splitter.get_splits_by_group(
          items_and_groups, two_split_groups, split_num, 'intra')
      self.assertEqual({'A', 'B', 'C', 'D'}, res.keys())

      three_split_groups = [(str(i), 1.0 / 3) for i in range(3)]
      res = splitter.get_splits_by_group(
          items_and_groups, three_split_groups, split_num, 'intra')
      # Now A shouldn't have a split
      self.assertEqual({'B', 'C', 'D'}, res.keys())

      four_split_groups = [(str(i), 1.0 / 4) for i in range(4)]
      res = splitter.get_splits_by_group(
          items_and_groups, four_split_groups, split_num, 'intra')
      # Now A and B shouldn't have a split
      self.assertEqual({'C', 'D'}, res.keys())

      five_split_groups = [(str(i), 1.0 / 5) for i in range(5)]
      res = splitter.get_splits_by_group(
          items_and_groups, five_split_groups, split_num, 'intra')
      # Now A, B and C shouldn't have a split
      self.assertEqual({'D'}, res.keys())

  def test_get_intra_splits_by_group_correct_splits(self):
    items_and_groups = []
    # 4 groups with 10, 15, 20, 25 items each.
    groups = ['A']*10 + ['B']*15 + ['C']*20 + ['D']*25
    items_and_groups = list(enumerate(groups))

    split_probs = [('train', 0.5), ('dev', 0.25), ('test', 0.25)]

    for split_num in range(3):
      res = splitter.get_splits_by_group(
          items_and_groups, split_probs, split_num, 'intra')

      # Check we have the expected 4 groups in the result.
      self.assertEqual({'A', 'B', 'C', 'D'}, res.keys())

      # Check that every group has a train dev and test, and that they have the
      # correct items.
      i2g = dict(items_and_groups)
      for group, l in [('A', 10), ('B', 15), ('C', 20), ('D', 25)]:
        cur_splits = res[group]
        # Check current split has train dev and test sets
        self.assertEqual({'train', 'dev', 'test'}, cur_splits.keys())
        # Check the current split has the correct number of items
        self.assertEqual(l, sum([len(items) for items in cur_splits.values()]))
        for items in cur_splits.values():
          for item in items:
            self.assertEqual(group, i2g[item])

  def test_get_splits_by_group_effect_of_split_number(self):
    items_and_groups = []
    for group in range(20):  # Many groups reduce chance of same random shuffle.
      for i in range(3):
        items_and_groups.append(('{}-{}'.format(group, i), group))
    split_probs = [('train', 0.6), ('dev', 0.2), ('test', 0.2)]

    get_split = functools.partial(
        splitter.get_splits_by_group, items_and_groups, split_probs)

    # Check running several times with same split number gives the same split.
    for split_num in range(10):
      split1 = get_split(split_num, 'inter')
      for _ in range(5):
        split2 = get_split(split_num, 'inter')
        self.assertEqual(split1, split2)

    # Check different split numbers give different splits.
    def freeze(split):
      """Return an equivalent split which is immutable and thus hashable."""
      return frozenset(
          (name, frozenset(items)) for name, items in split.items()
      )
    set_of_splits = {freeze(get_split(i, 'inter')) for i in range(15)}
    self.assertLen(set_of_splits, 15)


if __name__ == '__main__':
  absltest.main()
