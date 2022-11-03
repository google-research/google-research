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

"""Tests for Flax modules."""

import functools

from absl.testing import parameterized
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow.compat.v1 as tf

from protein_lm import domains
from protein_lm import models
from protein_lm import modules

lm_cls = functools.partial(
    models.FlaxLM,
    num_layers=1,
    num_heads=1,
    emb_dim=64,
    mlp_dim=64,
    qkv_dim=64)


class ModulesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (modules.AddLearnedPositionalEncodings,),
      (modules.AddSinusoidalPositionalEncodings,))
  def test_positional_encodings(self, positional_encoding_module):
    """Tests that the model runs with both types of positional encodings."""
    domain = domains.FixedLengthDiscreteDomain(vocab_size=2, length=2)
    lm = lm_cls(domain=domain,
                positional_encoding_module=positional_encoding_module)
    lm.sample(1)

  def test_embeddings_for_one_hot(self):
    """Tests that the embeddings match for int and one-hot representations."""
    vocab_size = 10
    emb_dim = 7
    x_int = jnp.array([[1, 3], [2, 8]])
    module = modules.Embed.partial(
        num_embeddings=vocab_size, num_features=emb_dim)
    _, params = module.init(jrandom.PRNGKey(0), x_int)
    emb_int = module.call(params, x_int)
    x_one_hot = jnp.eye(vocab_size)[x_int]
    emb_one_hot = module.call(params, x_one_hot)
    self.assertAllEqual(emb_int, emb_one_hot)

  def test_embeddings_for_dist(self):
    """Tests that the embeddings for soft inputs contain both tokens."""
    vocab_size = 5
    emb_dim = 7
    x_int = np.array([[1], [3]])
    module = modules.Embed.partial(
        num_embeddings=vocab_size, num_features=emb_dim)
    _, params = module.init(jrandom.PRNGKey(0), x_int)
    emb_int = module.call(params, x_int)
    x_dist = np.array([[[0, 0.25, 0, 0.75, 0]], [[0, 0.5, 0, 0.5, 0]]])
    emb_dist = np.array(module.call(params, x_dist))
    emb_expected = np.array([[emb_int[0, 0] * 0.25 + emb_int[1, 0] * 0.75],
                             [emb_int[0, 0] * 0.5 + emb_int[1, 0] * 0.5]])
    self.assertAllClose(emb_dist, emb_expected)

  @parameterized.parameters(
      ('logits', False),
      (['logits'], True),
      (('logits', 'output_emb'), True),
  )
  def test_output_head(self, output_head, multiple_heads):
    domain = domains.FixedLengthDiscreteDomain(vocab_size=2, length=2)
    inputs = domain.sample_uniformly(8)
    lm = lm_cls(domain=domain, pmap=False)
    outputs = models.predict_step(
        lm.optimizer.target,
        inputs,
        preprocess_fn=lm.preprocess,
        output_head=output_head)
    if multiple_heads:
      self.assertIsInstance(outputs, dict)
      self.assertLen(outputs, len(output_head))
    else:
      # We should have gotten a single output, the logits.
      self.assertEqual(outputs.shape,
                       (inputs.shape[0], inputs.shape[1], lm.vocab_size))


if __name__ == '__main__':
  tf.test.main()
