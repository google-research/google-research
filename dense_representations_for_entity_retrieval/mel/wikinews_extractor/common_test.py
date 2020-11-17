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

"""Tests for `common`."""

from absl.testing import absltest

from dense_representations_for_entity_retrieval.mel.wikinews_extractor import common


class CommonTest(absltest.TestCase):

  def test_wiki_encode(self):
    self.assertEqual(
        common.wiki_encode("http://website.com/Page1"),
        "http://website.com/Page1")
    self.assertEqual(common.wiki_encode("A B-C'i"), "A_B-C'i")
    self.assertEqual(common.wiki_encode("ABC"), "ABC")
    self.assertEqual(common.wiki_encode("x~y"), "x~y")
    self.assertEqual(common.wiki_encode("a.x!"), "a.x!")
    self.assertEqual(common.wiki_encode("a\\b"), "a%5Cb")
    self.assertEqual(
        common.wiki_encode("hastalığı"), "hastal%C4%B1%C4%9F%C4%B1")


if __name__ == "__main__":
  absltest.main()
