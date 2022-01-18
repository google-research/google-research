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

"""Tests for multi_task."""

from absl.testing import absltest
from dedal import multi_task


class MultiTaskTest(absltest.TestCase):
  """Tests the generic NamedLists container and the Backbone specific case."""

  def test_namedlists(self):
    container = multi_task.NamedLists(
        {'first': (1, 2, 3), 'second': [], 'third': [4, 5]})
    self.assertEqual(container.shape, (3, 0, 2))
    self.assertSequenceEqual(list(container), [1, 2, 3, 4, 5])
    self.assertSequenceEqual(container.first, [1, 2, 3])
    self.assertIsInstance(container.first, list)
    with self.assertRaises(KeyError):
      _ = container.something

    constant = False
    copy = container.constant_copy(constant)
    self.assertEqual(copy.shape, container.shape)
    for level, num in zip(copy.levels, copy.shape):
      self.assertSequenceEqual(level, [constant] * num)

    copy = container.copy()
    self.assertSequenceEqual(copy.first, container.first)
    copy.first[0] = 8
    self.assertNotEqual(copy.first[0], container.first[0])
    self.assertSequenceEqual(copy.first[1:], container.first[1:])

    values = [0, 0, 0, 2, 2]
    packed = container.pack(values)
    self.assertEqual(packed.shape, container.shape)
    self.assertSequenceEqual(list(packed), values)
    self.assertSequenceEqual(packed.first, values[:3])
    self.assertSequenceEqual(packed.second, [])
    self.assertSequenceEqual(packed.third, values[3:])

    flattened = container.flatten()
    self.assertIsInstance(flattened, dict)
    self.assertLen(flattened, 6)
    self.assertIn('first/0', flattened)
    self.assertEqual(flattened['first/0'], 1)
    self.assertIn('second/', flattened)
    self.assertIsNone(flattened['second/'])
    unflattened = multi_task.NamedLists.unflatten(flattened)
    self.assertEqual(unflattened.shape, container.shape)

  def test_backbone(self):
    values = [1, 2, 4]
    container = multi_task.Backbone(values)
    self.assertSequenceEqual(container.embeddings, values)
    self.assertSequenceEqual(container.alignments, [])
    self.assertSequenceEqual(list(container), values)


class SwitchNamedListsTest(absltest.TestCase):
  """Tests the generic SwitchNamedLists class."""

  def setUp(self):
    super().setUp()
    self.inputs = [
        multi_task.NamedLists(
            {'l1': [1.0, 2.0], 'l2': [5.0, 6.0], 'l3': [7.0, 8.0, 9.0]}),
        multi_task.NamedLists(
            {'l1': [3.0], 'l2': [4.0], 'l3': []}),
    ]
    self.switch = multi_task.SwitchNamedLists(
        {'l1': [0, 0, 1], 'l2': [1, 0, 0], 'l3': [0, 0, 0]})
    self.merged = multi_task.NamedLists(
        {'l1': [1.0, 2.0, 3.0], 'l2': [4.0, 5.0, 6.0], 'l3': [7.0, 8.0, 9.0]})

  def test_n(self):
    self.assertEqual(2, self.switch.n)

  def test_filter(self):
    for i, inputs_i in enumerate(self.inputs):
      self.assertEqual(inputs_i, self.switch.filter(self.merged, i))

  def test_merge_and_merge_flattened(self):
    # Tests "original" merge.
    self.assertEqual(self.merged, self.switch.merge(self.inputs))
    # Tests "flattened" merge.
    flat_merged = self.merged.flatten()
    flat_inputs = [inputs_i.flatten() for inputs_i in self.inputs]
    self.assertEqual(flat_merged, self.switch.merge_flattened(flat_inputs))

  def test_get_selector(self):
    expected_output0 = multi_task.NamedLists({'l1': [True, True, False],
                                              'l2': [False, True, True],
                                              'l3': [True, True, True]})
    self.assertEqual(expected_output0, self.switch.get_selector(0))
    expected_output1 = multi_task.NamedLists({'l1': [False, False, True],
                                              'l2': [True, False, False],
                                              'l3': [False, False, False]})
    self.assertEqual(expected_output1, self.switch.get_selector(1))


if __name__ == '__main__':
  absltest.main()
