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

"""Tests for preprocessors."""
from absl.testing import absltest
from seqio import test_utils
import tensorflow.compat.v2 as tf

from kl_guided_sampling import preprocessors

assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):

  def test_tf_negate_before_pattern_hit(self):
    arr = tf.constant([1, 2, 3, 4, 5, 6])
    pattern = tf.constant([3, 4])
    result = preprocessors.tf_negate_before_pattern(arr, pattern)
    self.assertEqual([-1, -2, 3, 4, 5, 6], result.numpy().tolist())

  def test_tf_negate_singleton_pattern(self):
    arr = tf.constant([1, 2, 3, 4, 5, 6])
    pattern = tf.constant([6])
    result = preprocessors.tf_negate_before_pattern(arr, pattern)
    self.assertEqual([-1, -2, -3, -4, -5, 6], result.numpy().tolist())

  def test_tf_negate_before_pattern_miss(self):
    arr = tf.constant([1, 2, 3, 4, 5, 6])
    pattern = tf.constant([3, 3])
    result = preprocessors.tf_negate_before_pattern(arr, pattern)
    self.assertEqual([1, 2, 3, 4, 5, 6], result.numpy().tolist())

  def test_tf_negate_before_pattern_hit_but_no_change(self):
    arr = tf.constant([1, 2, 3, 4, 5, 6])
    pattern = tf.constant([1, 2])
    result = preprocessors.tf_negate_before_pattern(arr, pattern)
    self.assertEqual([1, 2, 3, 4, 5, 6], result.numpy().tolist())

  def test_tf_negate_before_pattern_hit_many_and_last_used(self):
    arr = tf.constant([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
    pattern = tf.constant([1, 2])
    result = preprocessors.tf_negate_before_pattern(arr, pattern)
    self.assertEqual([-1, -2, -3, -4, -5, -6, 1, 2, 3, 4, 5, 6],
                     result.numpy().tolist())

  def test_tf_negate_inputs(self):
    self.tokenized_dataset = tf.data.Dataset.from_tensors({
        'inputs': [1, 2, 3, 4, 5, 6],
        'targets': [3, 3, 4, 5, 6],
    })

    assert_dataset(
        preprocessors.tf_negate_inputs(
            self.tokenized_dataset, tf.constant([3, 4])
        ),
        {
            'inputs': [-1, -2, 3, 4, 5, 6],
            'targets': [3, 3, 4, 5, 6],
        },
    )

if __name__ == '__main__':
  absltest.main()
