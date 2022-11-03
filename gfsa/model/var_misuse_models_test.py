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

"""Tests for gfsa.model.var_misuse_models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from gfsa.model import var_misuse_models

NUM_STATIC_NODES = 16
NUM_REAL_NODES = 13
NODE_EMBEDDING_DIM = 8


class VarMisuseModelsTest(parameterized.TestCase):

  @parameterized.parameters("two_pointer_output_head",
                            "bilinear_joint_output_head",
                            "bug_conditional_output_head")
  def test_output_head(self, head_name):
    head_fn = var_misuse_models.VAR_MISUSE_OUTPUT_HEADS[head_name]
    candidate_mask = np.zeros((NUM_STATIC_NODES,), dtype=bool)
    candidate_mask[:NUM_REAL_NODES] = True
    logits, _ = head_fn.init(
        jax.random.PRNGKey(0),
        node_embeddings=jnp.zeros((NUM_STATIC_NODES, NODE_EMBEDDING_DIM)),
        output_mask=candidate_mask)

    self.assertEqual(logits.shape, (NUM_STATIC_NODES, NUM_STATIC_NODES))
    joint_probs = np.exp(logits)
    np.testing.assert_allclose(np.sum(joint_probs), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.sum(joint_probs[:NUM_REAL_NODES, :NUM_REAL_NODES]), 1.0, atol=1e-6)
    self.assertTrue(np.all(joint_probs >= 0))


if __name__ == "__main__":
  absltest.main()
