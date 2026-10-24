# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for uflow.uflow_data.combine_train_datasets.

`make_train_iterator` builds one `tf.data.Dataset` per entry in the `train_on`
string but used to take only `train_datasets[0]`, silently training on the
first format alone.
"""

import sys
import types

from absl.testing import absltest
import numpy as np
import tensorflow as tf

# `uflow_augmentation` imports `tf.estimator`, which TensorFlow removed in
# 2.16. The function under test does not use augmentation, so stub the module
# instead of forcing an old TensorFlow just to collect these tests.
if 'uflow.uflow_augmentation' not in sys.modules:
  _aug = types.ModuleType('uflow.uflow_augmentation')
  _aug.apply_augmentation = lambda *a, **k: None
  sys.modules['uflow.uflow_augmentation'] = _aug

from uflow import uflow_data  # pylint: disable=g-import-not-at-top


def _dataset(values):
  return tf.data.Dataset.from_tensor_slices(np.asarray(values, np.int64))


class CombineTrainDatasetsTest(absltest.TestCase):

  def _collect(self, ds, count):
    return [int(v) for v in ds.take(count).as_numpy_iterator()]

  def test_single_dataset_is_returned_unchanged(self):
    ds = _dataset([1, 2, 3])
    self.assertIs(uflow_data.combine_train_datasets([ds]), ds)

  def test_samples_from_every_dataset(self):
    """Each input dataset must contribute elements."""
    a = _dataset([0] * 100)
    b = _dataset([1] * 100)
    c = _dataset([2] * 100)
    combined = uflow_data.combine_train_datasets([a, b, c])

    seen = set(self._collect(combined, 300))
    self.assertEqual(seen, {0, 1, 2})

  def test_previously_dropped_datasets_are_used(self):
    """The second and later datasets are no longer discarded."""
    first = _dataset([10] * 200)
    second = _dataset([20] * 200)
    combined = uflow_data.combine_train_datasets([first, second])

    values = self._collect(combined, 200)
    self.assertIn(10, values)
    self.assertIn(20, values, 'elements from the second dataset are missing')
    # Neither dataset should dominate entirely.
    self.assertGreater(values.count(10), 10)
    self.assertGreater(values.count(20), 10)

  def test_empty_list_is_rejected(self):
    with self.assertRaises(ValueError):
      uflow_data.combine_train_datasets([])

  def test_preserves_element_structure(self):
    """Element structure of the sources is passed through unchanged."""
    ds_a = tf.data.Dataset.from_tensor_slices(
        {'imgs': np.zeros((4, 2, 2, 3), np.float32)})
    ds_b = tf.data.Dataset.from_tensor_slices(
        {'imgs': np.ones((4, 2, 2, 3), np.float32)})
    combined = uflow_data.combine_train_datasets([ds_a, ds_b])
    sample = next(iter(combined))
    self.assertIn('imgs', sample)
    self.assertEqual(sample['imgs'].shape, (2, 2, 3))


if __name__ == '__main__':
  absltest.main()
