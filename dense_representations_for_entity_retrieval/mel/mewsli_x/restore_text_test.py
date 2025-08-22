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

"""Tests for restore_text."""

import pathlib

from absl.testing import absltest
from absl.testing import flagsaver

from dense_representations_for_entity_retrieval.mel.mewsli_x import restore_text
from dense_representations_for_entity_retrieval.mel.mewsli_x import schema


TESTDATA_DIR = (
    "dense_representations_for_entity_retrieval/mel/mewsli_x/testdata")


class RestoreTextTest(absltest.TestCase):

  def test_restore_text(self):
    testdata_dir = (
        pathlib.Path(absltest.get_default_test_srcdir()) / TESTDATA_DIR)
    input_file = testdata_dir / "example_mentions_no_text.jsonl"
    expected_file = testdata_dir / "example_mentions.jsonl"

    output_dir = self.create_tempdir()
    output_file = pathlib.Path(output_dir.full_path) / "restored.jsonl"
    with flagsaver.flagsaver(
        input=str(input_file),
        index_dir=str(testdata_dir),
        output=str(output_file)):
      restore_text.main([])

    self.assertTrue(output_file.exists(), msg=str(output_file))

    # Compare the dataclass representations of the output and expected files
    # rather than their string contents to be robust against fluctuations in the
    # serialization (e.g. due to dictionary order, etc).
    got = schema.load_jsonl(output_file, schema.ContextualMentions)
    expected = schema.load_jsonl(expected_file, schema.ContextualMentions)
    self.assertEqual(got, expected)


if __name__ == "__main__":
  with flagsaver.flagsaver(
      index_dir="ignored here", input="ignored here", output="ignored here"):
    absltest.main()
