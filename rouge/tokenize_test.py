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

"""Tests for tokenize."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from rouge import tokenize


class TokenizeTest(absltest.TestCase):

  def test_give_me_a_name(self):
    self.assertEqual(['one', 'two', 'three'],
                     tokenize.tokenize('one Two three', None))
    self.assertEqual(['one', 'two', 'three'],
                     tokenize.tokenize('one\n Two \nthree', None))


if __name__ == '__main__':
  absltest.main()
