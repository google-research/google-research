# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
"""Tests for criteo.data_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import six
from six.moves import range
import tensorflow.compat.v2 as tf

from uq_benchmark_2019.criteo import data_lib


class DataLibTest(absltest.TestCase):

  def test_build_dataset(self):
    config = data_lib.DataConfig(split='train', fake_data=True)
    dataset = data_lib.build_dataset(config, batch_size=8,
                                     is_training=False, fake_training=False)

    # Check output_shapes.
    features_shapes, label_shape = dataset.output_shapes
    self.assertEqual([None], label_shape.as_list())
    expected_keys = [data_lib.feature_name(i)
                     for i in range(1, data_lib.NUM_TOTAL_FEATURES+1)]
    self.assertSameElements(expected_keys, list(features_shapes.keys()))
    for key, shape in six.iteritems(features_shapes):
      self.assertEqual([None], shape.as_list(), 'Unexpected shape at key='+key)

    # Check output_types.
    features_types, label_type = tf.compat.v1.data.get_output_types(dataset)
    self.assertEqual(tf.float32, label_type)
    for idx in data_lib.INT_FEATURE_INDICES:
      self.assertEqual(tf.float32, features_types[data_lib.feature_name(idx)])
    for idx in data_lib.CAT_FEATURE_INDICES:
      self.assertEqual(tf.string, features_types[data_lib.feature_name(idx)])


if __name__ == '__main__':
  absltest.main()
