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

r"""Tests for Fast Self Attention mechanism.

Tests Fast Self Attention mechanism based on random feature maps.
"""

import functools
import time
from typing import Iterable

from absl import logging
from absl.testing import absltest
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as onp

from performer.fast_attention.jax import fast_attention


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError("Attention axis must be between the batch "
                       "axis and the last-two axes.")
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


def kernel_feature_creator(data,
                           projection_matrix,
                           attention_dims_t,
                           batch_dims_t,
                           precision,
                           is_query,
                           normalize_data=True):
  del is_query
  return fast_attention.sincos_softmax_kernel_feature_creator(
      data, projection_matrix, attention_dims_t, batch_dims_t, precision,
      normalize_data)


class FSAAccuracyTest(absltest.TestCase):

  def test_evaluate_parameter(self):

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    dim1 = 2
    dim2 = 1
    dim3 = 1
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, dim1, dim2, dim3, num_heads, qk_dim)
    shape_key = (batch_size, dim1, dim2, dim3, num_heads, qk_dim)
    shape_value = (batch_size, dim1, dim2, dim3, num_heads, v_dim)
    query = random.normal(random.PRNGKey(0), shape_query)
    key = random.normal(random.PRNGKey(0), shape_key)
    value = random.normal(random.PRNGKey(0), shape_value)

    renormalize_attention = True
    numerical_stabilizer = 0.0
    redraw_features = False
    unidirectional = False

    unstructured_random_matrix_creator = functools.partial(
        fast_attention.GaussianUnstructuredRandomMatrix, nb_random_features,
        qk_dim)
    ortho_random_matrix_creator = functools.partial(
        fast_attention.GaussianOrthogonalRandomMatrix, nb_random_features,
        qk_dim)
    fast_unstruct_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
        unstructured_random_matrix_creator, kernel_feature_creator,
        renormalize_attention, numerical_stabilizer, redraw_features,
        unidirectional)
    fast_ortho_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
        ortho_random_matrix_creator, kernel_feature_creator,
        renormalize_attention, numerical_stabilizer, redraw_features,
        unidirectional)

    standard_attention_result = dot_product_attention(
        query, key, value)
    unstruct_rfm_attention_result = fast_unstruct_rfm_dot_product_attention.dot_product_attention(
        query, key, value)
    ortho_rfm_attention_result = fast_ortho_rfm_dot_product_attention.dot_product_attention(
        query, key, value)

    max_error = 0.33
    unstruct_error = jnp.abs(
        (standard_attention_result - unstruct_rfm_attention_result) /
        standard_attention_result)
    ortho_error = jnp.abs(
        (standard_attention_result - ortho_rfm_attention_result) /
        standard_attention_result)
    self.assertLess(jnp.max(jnp.abs(unstruct_error)), max_error)

    max_ortho_error = 2.0
    self.assertLess(jnp.max(jnp.abs(ortho_error)), max_ortho_error)

  def test_small_example_evaluate_unidirectional(self):

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]

    qk_dim = 1
    batch_size = 1
    dim = 3
    num_heads = 1
    nb_random_features = 10000

    shape_query = (batch_size, dim, num_heads, qk_dim)
    shape_key = (batch_size, dim, num_heads, qk_dim)

    query = jnp.ones(shape_query)
    key = jnp.ones(shape_key)

    value = onp.zeros((1, 3, 1, 1))
    value[0][0][0][0] = 1.0
    value[0][1][0][0] = 0.0
    value[0][2][0][0] = 0.0
    value = jnp.array(value)

    groundtruth = onp.array([[[[1.0]], [[0.5]], [[1.0 / 3.0]]]])

    renormalize_attention = True
    numerical_stabilizer = 0.0
    redraw_features = False
    unidirectional = True

    unstructured_random_matrix_creator = functools.partial(
        fast_attention.GaussianUnstructuredRandomMatrix, nb_random_features,
        qk_dim)
    ortho_random_matrix_creator = functools.partial(
        fast_attention.GaussianOrthogonalRandomMatrix, nb_random_features,
        qk_dim)
    fast_unstruct_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
        unstructured_random_matrix_creator, kernel_feature_creator,
        renormalize_attention, numerical_stabilizer, redraw_features,
        unidirectional)
    fast_ortho_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
        ortho_random_matrix_creator, kernel_feature_creator,
        renormalize_attention, numerical_stabilizer, redraw_features,
        unidirectional)

    unidirectional_unstruct_rfm_attention_result = fast_unstruct_rfm_dot_product_attention.dot_product_attention(
        query, key, value)

    unidirectional_ortho_rfm_attention_result = fast_ortho_rfm_dot_product_attention.dot_product_attention(
        query, key, value)

    max_error = 0.02
    unstruct_error = jnp.abs(unidirectional_unstruct_rfm_attention_result -
                             groundtruth)
    ortho_error = jnp.abs(unidirectional_ortho_rfm_attention_result -
                          groundtruth)

    self.assertLess(jnp.max(jnp.abs(unstruct_error)), max_error)
    self.assertLess(jnp.max(jnp.abs(ortho_error)), max_error)

  def test_attention_speed(self):

    fast = True
    mode = "backward"  # 'forward', 'backward'
    jit = True
    length = 256
    batch_size = 2
    qk_dim = 64
    sample_number = 10
    num_heads = 1
    renormalize_attention = True
    nb_features = 256
    unidirectional = False

    if fast:
      raw_attention_fn = fast_attention.make_fast_generalized_attention(
          qk_dim // num_heads,
          renormalize_attention=renormalize_attention,
          nb_features=nb_features,
          unidirectional=unidirectional)
    else:
      raw_attention_fn = dot_product_attention

    def sum_attention_fn(*args, **kwargs):
      return jnp.sum(raw_attention_fn(*args, **kwargs))

    if jit:
      attention_fn = jax.jit(sum_attention_fn)

    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, qk_dim)

    query = jnp.array(onp.random.rand(*shape_query)) * 0.001
    key = jnp.array(onp.random.rand(*shape_key)) * 0.001
    value = jnp.array(onp.random.rand(*shape_value)) * 0.001

    raw_grad_fn = jax.grad(lambda q: sum_attention_fn(q, key=key, value=value))
    grad_fn = lambda q: jnp.sum(raw_grad_fn(q))

    if jit:
      grad_fn = jax.jit(grad_fn)

    for s in range(sample_number):
      logging.info("Sample: %d", s)

      if mode == "forward":
        start = time.time()
        attention_fn(query, key, value).block_until_ready()
        end = time.time()
      elif mode == "backward":
        start = time.time()
        grad_fn(query).block_until_ready()
        end = time.time()

      logging.info("Time Taken: %f", end - start)


if __name__ == "__main__":
  absltest.main()
