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

"""Tests for attention."""

from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import attention

_EXAMPLE_NUM_HEADS = 2
_EXAMPLE_SEQLEN = 4
_EXAMPLE_QKV = 4

# Explanation of our query/key/value test tensors.
#
# We use num_heads=2, seqlen=4, d_qkv=4.
#
# To make the test data simple enough to manually verify correctness, we arrange
# for:
# * head 0 will attend to a 50/50 mix of the token i (the current token) and
#   token i-1 (the one immediately before), wrapping around when negative.
# * head 1 will attend to a 50/50 mix of the token i (the current token) and
#   token i-2 (the one immediately before), wrapping around when negative.
#
# We achieve this by having the keys be one-hot position encodings, i.e.
# encodes position 2. Queries can straightforwardly specify which positions to
# attend to via a 1/0 mask; we then divide the 1/0 mask by a small constant
# _SOFTMAX_TEMPERATURE to turn the "soft"max into a "hard"max.
#
# Values are chosen arbitrarily.

# _EXAMPLE_KEYS: float32[seqlen, d_qkv]
_EXAMPLE_KEYS = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], np.float32)

_SOFTMAX_TEMPERATURE = 0.01

# _EXAMPLE_QUERIES: float32[seqlen, num_heads, d_qkv]
_EXAMPLE_QUERIES = (1.0 / _SOFTMAX_TEMPERATURE) * np.array([
    [
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ],
    [
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ],
    [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
    ],
    [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ],
], np.float32)

# _EXAMPLE_VALUES: float32[seqlen, d_qkv]
_EXAMPLE_VALUES = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
], np.float32)

# _EXAMPLE_VALUES: float32[seqlen, num_heads, d_qkv]
_EXAMPLE_RESULTS = np.array([
    [
        _EXAMPLE_VALUES[0],  # Attention mask prevents attending to the future.
        _EXAMPLE_VALUES[0],  # Attention mask prevents attending to the future.
    ],
    [
        (_EXAMPLE_VALUES[0] + _EXAMPLE_VALUES[1]) / 2.0,
        _EXAMPLE_VALUES[1],  # Attention mask prevents attending to the future.
    ],
    [
        (_EXAMPLE_VALUES[1] + _EXAMPLE_VALUES[2]) / 2.0,
        (_EXAMPLE_VALUES[0] + _EXAMPLE_VALUES[2]) / 2.0,
    ],
    [
        (_EXAMPLE_VALUES[2] + _EXAMPLE_VALUES[3]) / 2.0,
        (_EXAMPLE_VALUES[1] + _EXAMPLE_VALUES[3]) / 2.0,
    ],
])


class AttentionTest(absltest.TestCase):

  def test_attend_nonincremental(self):
    batch = 2
    q = jnp.broadcast_to(
        _EXAMPLE_QUERIES,
        (batch, _EXAMPLE_SEQLEN, _EXAMPLE_NUM_HEADS, _EXAMPLE_QKV))
    k = jnp.broadcast_to(_EXAMPLE_KEYS, (batch, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    v = jnp.broadcast_to(_EXAMPLE_VALUES,
                         (batch, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    expected = jnp.broadcast_to(
        _EXAMPLE_RESULTS,
        (batch, _EXAMPLE_SEQLEN, _EXAMPLE_NUM_HEADS, _EXAMPLE_QKV))
    np.testing.assert_allclose(expected, attention.attend(q, k, v, [], 0))

  def test_attend_incremental(self):
    batch = 1
    beam = 2
    num_layers = 2
    layer = 0

    # full_cache_k, full_cache_v: float32[ 1, 1, seqlen, d_qkv]
    full_cache_k = _EXAMPLE_KEYS[np.newaxis, np.newaxis, :, :]
    full_cache_v = _EXAMPLE_VALUES[np.newaxis, np.newaxis, :, :]
    # full_cache_k, full_cache_v: float32[seqlen, num_layers, batch, d_qkv]
    full_cache_k = jnp.broadcast_to(
        full_cache_k, (num_layers, batch, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    full_cache_v = jnp.broadcast_to(
        full_cache_v, (num_layers, batch, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    print(full_cache_k.shape)

    full_q = jnp.broadcast_to(
        _EXAMPLE_QUERIES,
        (batch * beam, _EXAMPLE_SEQLEN, _EXAMPLE_NUM_HEADS, _EXAMPLE_QKV))
    full_k = jnp.broadcast_to(_EXAMPLE_KEYS,
                              (batch * beam, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    full_v = jnp.broadcast_to(_EXAMPLE_VALUES,
                              (batch * beam, _EXAMPLE_SEQLEN, _EXAMPLE_QKV))
    full_expected = jnp.broadcast_to(
        _EXAMPLE_RESULTS,
        (batch * beam, _EXAMPLE_SEQLEN, _EXAMPLE_NUM_HEADS, _EXAMPLE_QKV))

    for split_point in [1, 2, 3]:
      # Truncate kv cache at split_point.
      kv_cache = attention.KVCache(
          k=full_cache_k[:, :, :split_point, :],
          v=full_cache_v[:, :, :split_point, :],
          lengths=jnp.broadcast_to(split_point, (batch,)),
          offset=0,
      )
      result = attention.attend(full_q[:, split_point:, :, :],
                                full_k[:, split_point:, :],
                                full_v[:, split_point:, :], [kv_cache], layer)
      np.testing.assert_allclose(full_expected[:, split_point:, :, :], result)


if __name__ == '__main__':
  absltest.main()
