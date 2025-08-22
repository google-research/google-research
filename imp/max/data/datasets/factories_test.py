# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for dataset base."""

from absl.testing import absltest

from imp.max.data.datasets import factories


class TestDataset(factories.ThirdPartyDatasetsBase):

  # TODO(b/215205249): Add a real test dataset.
  def tables(self):
    return {'t1': ['a', 'b'], 't2': 'c'}


class BaseTest(absltest.TestCase):

  def test_thirdparty_dataset_invalid_table_selection(self):
    with self.assertRaises(ValueError):
      TestDataset('dir', 'table')

  def test_thirdparty_dataset_buildable_with_subset_list(self):
    TestDataset('dir', 't1')

  def test_thirdparty_dataset_buildable_with_subset_element(self):
    TestDataset('dir', 't2')
    TestDataset('dir', 't2', num_shards=1, shard_index=0)
    with self.assertRaises(ValueError):
      TestDataset('dir', 't2', num_shards=2)

  def test_thirdparty_dataset_modalities_provided(self):
    dataset = TestDataset('dir', 't1')
    self.assertEmpty(dataset.modalities_provided())

  def test_thirdparty_dataset_get_num_examples(self):
    dataset = TestDataset('dir', 't1')
    with self.assertRaises(AttributeError):
      dataset.get_num_examples('t1')

  def test_thirdparty_dataset_lookup(self):
    dataset = TestDataset('dir', 't2')
    small = dataset.lookup('key')
    self.assertEqual(dataset._source._key_prefix, '')
    self.assertEqual(small._source._key_prefix, 'key')


if __name__ == '__main__':
  absltest.main()
