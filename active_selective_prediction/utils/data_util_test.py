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

"""Unit tests for `data_util.py`."""

import unittest

from active_selective_prediction.utils import data_util
import numpy as np


class TestDatasetLoadingFunctions(unittest.TestCase):
  """Tests dataset loading functions."""

  def test_get_color_mnist_dataset(self):
    """Tests get_color_mnist_dataset function."""
    ds = data_util.get_color_mnist_dataset(
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 32, 32, 3))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_svhn_dataset(self):
    """Tests get_svhn_dataset function."""
    ds = data_util.get_svhn_dataset(
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 32, 32, 3))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_fmow_dataset(self):
    """Tests get_fmow_dataset function."""
    ds = data_util.get_fmow_dataset(
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 96, 96, 3))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_domainnet_dataset(self):
    """Tests get_domainnet_dataset function."""
    ds = data_util.get_domainnet_dataset(
        domain_name='real',
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 96, 96, 3))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_amazon_review_dataset(self):
    """Tests get_amazon_review_dataset function."""
    ds = data_util.get_amazon_review_dataset(
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 768))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_amazon_review_test_sub_dataset(self):
    """Tests get_amazon_review_test_sub_dataset function."""
    ds = data_util.get_amazon_review_test_sub_dataset(
        subset_index=1,
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 768))
    self.assertEqual(init_batch_y.shape, (2,))

  def test_get_otto_dataset(self):
    """Tests get_otto_dataset function."""
    ds = data_util.get_otto_dataset(
        split='train',
        batch_size=2,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 93))
    self.assertEqual(init_batch_y.shape, (2,))


class TestDataUtilFunctions(unittest.TestCase):
  """Tests data util functions."""

  def test_get_ds_data(self):
    """Tests get_ds_data function."""
    ds = data_util.get_color_mnist_dataset(
        split='test',
        batch_size=100,
        shuffle=False,
        drop_remainder=False,
        buffer_size=1000,
    )
    data_dict = data_util.get_ds_data(ds)
    self.assertEqual(data_dict['inputs'].shape, (10000, 32, 32, 3))
    self.assertEqual(data_dict['labels'].shape, (10000,))

  def test_construct_dataset(self):
    """Tests construct_dataset function."""
    inputs = np.ones((100, 5), dtype=np.float32)
    labels = np.zeros((100,), dtype=np.int32)
    data_dict = {'inputs': inputs, 'labels': labels}
    ds = data_util.construct_dataset(
        data_dict,
        batch_size=2,
        shuffle=False,
        include_label=True,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 5))
    self.assertEqual(init_batch_y.shape, (2,))
    unlabeled_ds = data_util.construct_dataset(
        data_dict,
        batch_size=2,
        shuffle=False,
        include_label=False,
        buffer_size=1000,
    )
    init_batch = next(iter(unlabeled_ds))
    self.assertEqual(isinstance(init_batch, tuple), False)
    self.assertEqual(init_batch.shape, (2, 5))

  def test_construct_sub_dataset(self):
    """Tests construct_sub_dataset function."""
    inputs = np.ones((100, 5), dtype=np.float32)
    labels = np.zeros((100,), dtype=np.int32)
    data_dict = {'inputs': inputs, 'labels': labels}
    selected_indices = np.array([0, 1], dtype=np.int32)
    # Makes the batch size equal to
    # the size of selected_indices.
    ds = data_util.construct_sub_dataset(
        data_dict,
        selected_indices,
        batch_size=selected_indices.shape[0],
        shuffle=False,
        include_label=True,
        buffer_size=1000,
    )
    init_batch_x, init_batch_y = next(iter(ds))
    self.assertEqual(init_batch_x.shape, (2, 5))
    self.assertEqual(init_batch_y.shape, (2,))
    unlabeled_ds = data_util.construct_sub_dataset(
        data_dict,
        selected_indices,
        batch_size=2,
        shuffle=False,
        include_label=False,
        buffer_size=1000,
    )
    init_batch = next(iter(unlabeled_ds))
    self.assertEqual(isinstance(init_batch, tuple), False)
    self.assertEqual(init_batch.shape, (2, 5))


if __name__ == '__main__':
  unittest.main()
