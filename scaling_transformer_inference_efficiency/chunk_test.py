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

"""Tests for chunk."""

from absl.testing import absltest
import numpy as np

from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import FullChunkResult


class ChunkTest(absltest.TestCase):

  def test_to_chunk_result(self):
    chunk = Chunk(
        tokens=np.array([[
            0,
            1,
            2,
        ]], np.int32),
        lengths=np.array([3], np.int32))
    probs = np.array([[
        [0.0, 0.6, 0.3, 0.1],
        [0.0, 0.05, 0.85, 0.1],
        [0.1, 0.15, 0.2, 0.55],
    ]], np.float32)
    logits = np.log2(probs)
    full_result = FullChunkResult(logits=logits, kv_cache=None)
    result = full_result.to_chunk_result(None, chunk, do_top_k=True)
    np.testing.assert_allclose(
        result.per_token_scores,
        np.log(np.array([[1.0, 0.6, 0.85]], np.float32)),
        rtol=1e-6)
    np.testing.assert_allclose(
        result.top_token_ids,
        np.array([[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0]]], np.int32))
    np.testing.assert_allclose(
        result.top_token_probs,
        np.array([[[1.0, 0.0, 0.0, 0.0], [0.6, 0.3, 0.1, 0.0],
                   [0.85, 0.1, 0.05, 0.0]]], np.float32),
        rtol=1e-6)
    np.testing.assert_allclose(
        result.next_token_logits, logits[:, -1, :], rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
