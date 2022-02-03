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

"""Tests for downstream_sklearn_eval."""

import os
from absl.testing import absltest
from non_semantic_speech_benchmark.trillsson import downstream_sklearn_eval


class DownstreamSklearnEvalTest(absltest.TestCase):

  def test_most_recent_embeddings_dump(self):
    embedding_filenames = set(range(102))
    prefix = 'prefix'

    recent_filenames = downstream_sklearn_eval._most_recent_embeddings_dump(
        embedding_filenames, prefix)
    expected_filenames = [
        os.path.join(prefix, str(i)) for i in range(101, -1, -1)
    ]
    self.assertListEqual(recent_filenames, expected_filenames)


if __name__ == '__main__':
  absltest.main()
