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

"""Tests for subgraph_extractors."""

import functools

from absl.testing import absltest

import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

import tree_math as tm

from jaxsel import agents
from jaxsel import subgraph_extractors
from jaxsel.tests import base_graph_test


def check_sparse_against_dense(sparse, dense):
  assert jnp.allclose(sparse.todense(), dense)


def diff_norm(tree, other_tree):
  tree_diff = jax.tree_multimap(lambda x, y: x - y, tree, other_tree)
  return jax.tree_map(jnp.linalg.norm, tree_diff).sum()


# TODO(gnegiar): add test comparing sparse implementation to
# dense implementation on a small problem with known solution.
# TODO(gnegiar): write test case where max_subgraph_size is too small
class ISTASubgraphExtractorsTest(base_graph_test.BaseGraphTest):

  def test_abs_top_k(self):
    u = jsparse.BCOO.fromdense(jnp.array([0., 0., 1., 10., -5., 2.]), nse=4)

    k = 3
    nse = 5

    topk_u = subgraph_extractors._abs_top_k(u, k, nse)
    expected = jnp.array([0., 0., 0., 10., -5., 2.])
    check_sparse_against_dense(topk_u, expected)

    assert topk_u.nse == nse

  def test_dense_submatrix_extraction(self):

    # Dense matrix, used to build the sparse matrix
    mat = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.9, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.9, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.9, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    n = 10
    assert n == mat.shape[0]
    k = 6  # Leave some wiggle room: nse > |true non zeros elements|.
    # Sparse matrix to extract from.
    mat_sp = jsparse.BCOO.fromdense(mat, nse=k**2)

    # Sparse vector, used to index the submatrix to extract.
    v = jnp.array([0.0, 0.0, 0.9, 0.7, 0.0, 0.9, 0.4, 0.0, 0.7, 0.0])
    v_sp = jsparse.BCOO.fromdense(v, nse=k)
    indices = v_sp.indices.flatten()
    assert jnp.allclose(indices, jnp.array([2, 3, 5, 6, 8, 10]))

    in_bounds_indices = indices[indices < n]
    assert jnp.allclose(in_bounds_indices, jnp.array([2, 3, 5, 6, 8]))

    # Extract submatrix.
    submat = self.extractor._extract_dense_submatrix(mat_sp, indices)
    # Check the extracted matrix's shape.
    assert submat.shape == (len(indices), len(indices))

    # Check that the extracted values are correct.
    expected = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.4],
        [0.4, 0.0, 0.0, 0.0, 0.0],
    ])

    assert jnp.allclose(
        submat[:len(in_bounds_indices)][:, :len(in_bounds_indices)], expected)

  def test_qstar(self):
    """Tests ability to extract a subgraph, with jit."""
    rng_extractor, self.rng = jax.random.split(self.rng)
    params = self.extractor.init(rng_extractor, self.start_node_id, self.graph)
    qstar, _, _, dense_submat, _, _, error = self.extractor.apply(
        params, self.start_node_id, self.graph)

    # Tests output shape
    assert qstar.shape == (self.extractor.config.max_subgraph_size,)

    s = self.extractor._s(self.start_node_id)
    # Tests convergence of the sparse PageRank
    assert error < 1e-4, subgraph_extractors._dense_fixed_point(
        qstar, dense_submat, s, self.extractor.config.alpha,
        self.extractor.config.rho)

  def test_backprop(self):
    """Tests ability to backpropagate through the Subgraph Selection layer.

    Verifies numerical value of the jax backprop vs finite differences.
    """

    rng_extractor, self.rng = jax.random.split(self.rng)
    params = self.extractor.init(rng_extractor, self.start_node_id, self.graph)

    def get_qstar_sum(params):
      return self.extractor.apply(params, self.start_node_id,
                                  self.graph)[0].sum()

    eps = 1e-5  # magnitude of the finite difference

    grad = jax.grad(get_qstar_sum)(params)
    delta_rng, self.rng = jax.random.split(self.rng)
    # Take a random direction
    delta_params = tm.Vector(
        self.extractor.init(delta_rng, self.start_node_id, self.graph))
    # Normalize
    delta_params = jax.tree_map(lambda x: x / max(1e-9, jnp.linalg.norm(x)),
                                delta_params).tree

    # Directional derivative given by jax
    deriv_jax = tm.Vector(jax.tree_multimap(jnp.vdot, delta_params, grad)).sum()

    # Directional derivative given by finite differences
    agent_plus_eps = (tm.Vector(params) + eps * tm.Vector(delta_params)).tree
    agent_minus_eps = (tm.Vector(params) - eps * tm.Vector(delta_params)).tree
    deriv_diff = ((tm.Vector(get_qstar_sum(agent_plus_eps)) -
                   tm.Vector(get_qstar_sum(agent_minus_eps))) / 2 * eps).tree

    err = diff_norm(deriv_jax, deriv_diff)
    assert jax.tree_util.tree_all(
        jax.tree_multimap(
            functools.partial(jnp.allclose, atol=1e-3), deriv_jax,
            deriv_diff)), f"Difference between FDM and autograd is {err}"

  def test_convert_to_bcoo_indices(self):
    node_id = 0
    n_neighbors = 5
    neighbor_node_ids = jnp.arange(n_neighbors)
    indices = agents._make_adjacency_mat_row_indices(node_id, neighbor_node_ids)
    assert (indices[0] == jnp.array([0, 0])).all()

  def test_make_dense_vector(self):
    q_dense = jnp.zeros(10).at[6].set(1.)
    q_sparse = jsparse.BCOO.fromdense(q_dense, nse=4)
    extracted_q = self.extractor._make_dense_vector(q_sparse, q_sparse.indices)
    assert (extracted_q == jnp.array([1., 0., 0., 0.])).all()

    full_q = self.extractor._make_dense_vector(q_sparse,
                                               jnp.arange(q_dense.shape[0]))
    assert (full_q == q_dense).all()


if __name__ == "__main__":
  absltest.main()
