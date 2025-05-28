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

"""Tests for io_util."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized

from dense_representations_for_entity_retrieval.mel.mewsli_x import io_util


class IoUtilTest(parameterized.TestCase):

  TEST_STR = "こんにちは世界"

  def setUp(self):
    super().setUp()
    self.test_file = self.create_tempfile()
    self.test_file.write_text(self.TEST_STR)

  @parameterized.parameters((str,), (pathlib.Path,))
  def test_open_file_default(self, arg_converter):
    path_arg = arg_converter(self.test_file.full_path)
    with io_util.open_file(path_arg) as f:
      got = f.read()
    self.assertEqual(got, self.TEST_STR)

  @parameterized.parameters((str,), (pathlib.Path,))
  def test_open_file_read(self, arg_converter):
    path_arg = arg_converter(self.test_file.full_path)
    with io_util.open_file(path_arg, "r") as f:
      got = f.read()
    self.assertEqual(got, self.TEST_STR)

  @parameterized.parameters((str,), (pathlib.Path,))
  def test_open_file_write(self, arg_converter):
    out_file = self.create_tempfile()
    path_arg = arg_converter(out_file.full_path)
    with io_util.open_file(path_arg, "w") as f:
      f.write(self.TEST_STR)
    self.assertEqual(out_file.read_text(), self.TEST_STR)

  @parameterized.parameters((str,), (pathlib.Path,))
  def test_make_dirs(self, arg_converter):
    temp_dir = self.create_tempdir()
    target_dir = pathlib.Path(temp_dir.full_path) / "sub/dirs"
    path_arg = arg_converter(target_dir)
    io_util.make_dirs(path_arg)
    self.assertTrue(target_dir.exists())


if __name__ == "__main__":
  absltest.main()
