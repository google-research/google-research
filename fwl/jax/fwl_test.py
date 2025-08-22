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

from absl.testing import absltest
import flax
import flax.linen as nn
import fwl
import jax
from jax import numpy as jnp
from jax import random


class FWLTest(absltest.TestCase):
  def test_fwl(self):
    # create some random inputs and pick a token to check the FWL output on
    batch_size, seq_len, fwl_size, vocab_size = 2, 8, 3, 5
    i, j = 1, 5
    xrng, lrng, wrng, modelrng = random.split(random.PRNGKey(0), 4)
    x = random.normal(xrng, (batch_size, seq_len, fwl_size), dtype=jnp.float32)
    labels = nn.one_hot(
        random.randint(lrng, (batch_size, seq_len), 0, vocab_size),
        vocab_size, dtype=jnp.float32)
    weights = random.randint(
        wrng, (batch_size, seq_len), 0, 2).astype(jnp.float32)

    model = fwl.FWBlock(size=4, vocab_size=5, attn_chunks=2)
    params = model.init(modelrng, x, labels, weights)

    # update FWL params based on the losses of previous tokens then re-run
    # the slow weight pass; this should match the FWBlock's output
    @jax.grad
    def get_grads(params):
      logits = model.apply(params, x, method=model.slow_weight_fwd)
      log_probs = -jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
      losses = weights * log_probs / jnp.maximum(jnp.sum(weights), 1e-8)
      return jnp.sum(losses[i, :j])

    grads = get_grads(params)
    flat_grads = flax.traverse_util.flatten_dict(grads)
    flat_params = flax.traverse_util.flatten_dict(params)
    for k, g in flat_grads.items():
      if "unembed" not in k and ("LayerNorm_0" in k or "bias" not in k):
        flat_params[k] -= 0.01 * g
    updated_params = flax.traverse_util.unflatten_dict(flat_params)

    assert jnp.allclose(
        model.apply(params, x, labels, weights)[i, j],
        model.apply(updated_params, x, method=model.slow_weight_fwd)[i, j]
    )


if __name__ == "__main__":
  absltest.main()
