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

r"""Tests for fast attention module.

Tests sanity of returns from attention module for Performer attention.
"""

from absl.testing import absltest
from flax.linen import initializers
from jax import random
import jax.numpy as jnp
from performer.fast_attention.jax.attention_module import MultiHeadDotProductFastAttention
from performer.fast_attention.jax.fast_attention import make_fast_softmax_attention


class AttentionModuleTest(absltest.TestCase):

  def test_multihead_self_attention(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 8))
    attention_fn = make_fast_softmax_attention(
        8,
        lax_scan_unroll=16,
        nb_features=256,
        unidirectional=False,
        redraw_features=True,)
    sa_module = MultiHeadDotProductFastAttention(
        num_heads=1,
        qkv_features=8,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        attention_fn=attention_fn,
    )
    y, _ = sa_module.init_with_output(rng, x, x)
    self.assertEqual(y.shape, x.shape)

  def test_multihead_encoder_decoder_attention(self):
    rng = random.PRNGKey(0)
    q = jnp.ones((4, 2, 3, 8))
    kv = jnp.ones((4, 2, 3, 8))
    attention_fn = make_fast_softmax_attention(
        8,
        lax_scan_unroll=16,
        nb_features=256,
        unidirectional=False,
        redraw_features=True,)
    sa_module = MultiHeadDotProductFastAttention(
        num_heads=4,
        qkv_features=32,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        attention_fn=attention_fn,
    )
    y, _ = sa_module.init_with_output(rng, q, kv)
    self.assertEqual(y.shape, q.shape)

  def test_1D_mask(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 8))
    x = jnp.ones((4, 2, 3, 8))
    mask = jnp.ones((4, 2, 3))
    attention_fn = make_fast_softmax_attention(
        8,
        lax_scan_unroll=16,
        nb_features=256,
        unidirectional=False,
        redraw_features=True,)
    sa_module = MultiHeadDotProductFastAttention(
        num_heads=1,
        qkv_features=8,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        attention_fn=attention_fn,
    )
    y, _ = sa_module.init_with_output(rng, x, x, mask_k=mask)
    self.assertEqual(y.shape, x.shape)

if __name__ == "__main__":
  absltest.main()
