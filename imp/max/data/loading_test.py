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

"""Tests for loading."""

import os

from absl.testing import absltest
import tensorflow as tf

from imp.max.data import loading


class LoadingTest(absltest.TestCase):

  def test_get_source(self):
    self.assertIs(loading.get_source('TFRecord'), loading.TFRecordSource)
    with self.assertRaises(NotImplementedError):
      loading.get_source('something')

  def test_get_latest_dir(self):
    path = self.create_tempdir()
    tf.io.gfile.makedirs(os.path.join(path, 'path/to/123/dir'))
    tf.io.gfile.makedirs(os.path.join(path, 'path/to/456/dir'))

    latest = loading.get_latest_dir(os.path.join(path, 'path/to/{latest}/dir'))
    self.assertEqual(latest, os.path.join(path, 'path/to/456/dir'))

    with self.assertRaises(ValueError):
      loading.get_latest_dir(os.path.join(path, 'path/to/123/dir/{latest}'))

  def test_example_custome_parser_builder_with_custom_extract_fn(self):
    builder = loading.ExampleCustomParserBuilder()
    builder.override_parse_fn(lambda x: x)
    inputs = 'test'
    result = builder._default_parse_fn(inputs)
    self.assertEqual(inputs, result)


if __name__ == '__main__':
  absltest.main()
