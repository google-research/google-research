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
"""Tests for clean_text_tester."""

from absl.testing import absltest

from smu.tools import clean_text_tester

smu_line = clean_text_tester.SmuLineForDiff


class DiffTester(absltest.TestCase):

  def test_simple_diff(self):
    self.assertNotEqual(smu_line('I am a teapot'), smu_line('I am a flower'))

  def test_simple_same(self):
    self.assertEqual(smu_line('I am a teapot'), smu_line('I am a teapot'))

  def test_diff_num_tokens(self):
    self.assertNotEqual(
        smu_line('token0 -0.00 token1 token 2'),
        smu_line('token0  0.00 token1'))

  def test_trailing_whitespace(self):
    self.assertNotEqual(smu_line('foo bar   '), smu_line('foo bar'))

  def test_neg_zero(self):
    self.assertEqual(smu_line('line -0.0 123'), smu_line('line -0.0 123'))
    self.assertEqual(smu_line('line  0.0 123'), smu_line('line -0.0 123'))
    self.assertEqual(smu_line('line -0.0 123'), smu_line('line  0.0 123'))
    self.assertEqual(smu_line('line  0.0 123'), smu_line('line  0.0 123'))
    self.assertEqual(smu_line('-0.000 123'), smu_line(' 0.000 123'))
    self.assertNotEqual(smu_line('line -0.00 123'), smu_line('line 0.00  123'))
    self.assertNotEqual(
        smu_line('-0.000not_float 123'), smu_line(' 0.000not_float 123'))
    self.assertNotEqual(smu_line('-1.23 foo'), smu_line(' 1.23 foo'))

  def test_rounding(self):
    self.assertNotEqual(smu_line('line 1.23 '), smu_line('line  1.23'))
    self.assertNotEqual(smu_line('line 1.23 '), smu_line('line 1.2  '))

    self.assertEqual(smu_line('line 1.23'), smu_line('line 1.23'))
    self.assertEqual(smu_line('line 1.23'), smu_line('line 1.24'))
    self.assertEqual(smu_line('line 1.24'), smu_line('line 1.23'))

    self.assertEqual(smu_line('line 1.299'), smu_line('line 1.300'))
    self.assertEqual(smu_line('line 1.300'), smu_line('line 1.299'))

    self.assertEqual(smu_line('line 1.32222'), smu_line('line 1.32221'))
    self.assertEqual(smu_line('line 1.322222'), smu_line('line 1.322221'))
    self.assertEqual(smu_line('line 1.3222222'), smu_line('line 1.3222221'))

    self.assertNotEqual(smu_line('line 1.123'), smu_line('line 1.121'))
    self.assertNotEqual(smu_line('line 1.123'), smu_line('line 1.125'))
    self.assertNotEqual(smu_line('line 1.121'), smu_line('line 1.123'))
    self.assertNotEqual(smu_line('line 1.125'), smu_line('line 1.123'))

    self.assertNotEqual(smu_line('line 1'), smu_line('line 2'))
    self.assertNotEqual(smu_line('line 2'), smu_line('line 1'))

  def test_real_example(self):
    self.assertNotEqual(
        smu_line(
            ' calc            warn 3        0            0            0                          x07_cn3o3h3.131334.013\n'
        ),
        smu_line(
            ' calc            warn 3        0            2            0                          x07_cn3o3h3.131334.013\n'
        ))


if __name__ == '__main__':
  absltest.main()
