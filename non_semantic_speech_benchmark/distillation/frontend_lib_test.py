# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for frontend."""

from absl.testing import absltest

from non_semantic_speech_benchmark.distillation import frontend_lib


class FrontendTest(absltest.TestCase):

  def test_default_shape(self):
    self.assertEqual(frontend_lib.get_frontend_output_shape(), [1, 96, 64])


if __name__ == '__main__':
  absltest.main()
