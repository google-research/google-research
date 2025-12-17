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

"""Test for Heterogeneous GNN library."""

import jax
import jax.numpy as jnp
import tensorflow as tf
from sparse_deferred.gnn import heterogeneous_gnn

HeteroGraphTransformerBlock = heterogeneous_gnn.HeteroGraphTransformerBlock


class HeterogeneousGnnTest(tf.test.TestCase):

  def test_basic_heterogeneous_graph_transformer(self):
    h_source = jnp.array([
        [4.0, 5.0, 6.0, 7.0],
        [22.0, 45.0, 9.0, 0.0],
        [11.0, 12.0, 13.0, 14.0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    source_mask = jnp.array([1, 1, 1, 0, 0])

    h_target = jnp.array(
        [[1.0, 2.0, 3.0, 4.0], [25.0, 16.0, 76.0, 28.0], [0.0, 0.0, 0.0, 0.0]]
    )
    target_mask = jnp.array([1, 1, 0])

    edge_mask = HeteroGraphTransformerBlock.create_mask_from_edges(
        (
            h_source.shape[0],
            h_target.shape[0],
        ),
        # Sparse encoding of edge IDs with SD Pad convention.
        jnp.array([0, 0, 1, 2, 4]),  # Source IDs
        jnp.array([0, 1, 0, 0, 2]),  # Target IDs
    )

    self.assertAllEqual(
        edge_mask,
        jnp.array([
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ]),
    )

    model = HeteroGraphTransformerBlock(hidden_dim=4, num_heads=2)

    params = model.init(jax.random.PRNGKey(0), h_source, h_target)
    output = model.apply(
        params,
        h_source,
        h_target,
        source_mask,
        target_mask,
        edge_mask,
        training=False,
    )

    # Check that the output is all zeros for padded nodes in the target set..
    self.assertAllEqual(output[3:, :], jnp.zeros_like(output[3:, :]))


if __name__ == "__main__":
  tf.test.main()
