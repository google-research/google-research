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

"""Utilities for approximate TopK."""

import jax

from jax import numpy as jnp


@jax.vmap
def slice_2d(x, y):
  return x[y]


def top_k_approx(scores, k=100):
  """Returns approximate topk highest scores for each row.

  The api is same as jax.lax.top_k, so this can be used as a drop in replacement
  as long as num dims of scores tensor is 2. For more dimensions, please use one
  or more vmap(s) to be able to use it.

  In essence, we perform jnp.max operation, which can be thought of as
  lossy top 1, on fixed length window of items. We can control the amound of
  approximation by changing the window length. Smaller it gets, the
  approximation gets better but at the cost of performance.

  Once we have the max for all the windows, we apply regular slow but exact
  jax.lax.top_k over reduced set of items.

  Args:
    scores: [num_rows, num_cols] shaped tensor. Will return top K over last dim.
    k: How many top scores to return for each row.

  Returns:
    Topk scores, topk ids. Both shaped [num_rows, k]
  """
  num_queries = scores.shape[0]
  num_items = scores.shape[1]

  # Make this bigger to improve recall. Should be between [1, k].
  num_windows_multiplier = 5
  window_lengths = num_items // k // num_windows_multiplier + 1
  padded_num_items = k * num_windows_multiplier * window_lengths

  print(f"scores shape: {scores.shape}")
  print(f"padded_num_items: {padded_num_items}")
  print(f"num_items: {num_items}")
  scores = jnp.pad(
      scores, ((0, 0), (0, padded_num_items - num_items)),
      mode="constant",
      constant_values=jnp.NINF)
  scores = jnp.reshape(
      scores, (num_queries, k * num_windows_multiplier, window_lengths))
  approx_top_local_scores = jnp.max(scores, axis=2)

  sorted_approx_top_scores_across_local = jnp.flip(
      jnp.sort(approx_top_local_scores, axis=1), axis=1)
  approx_top_ids_across_local = jnp.flip(
      jnp.argsort(approx_top_local_scores, axis=1), axis=1)[:, :k]

  approx_top_local_ids = jnp.argmax(scores, axis=2)
  offsets = jnp.arange(0, padded_num_items, window_lengths)
  approx_top_ids_with_offsets = approx_top_local_ids + offsets
  approx_top_ids = slice_2d(approx_top_ids_with_offsets,
                            approx_top_ids_across_local)

  topk_scores = sorted_approx_top_scores_across_local[:, :k]
  topk_ids = approx_top_ids

  return topk_scores, topk_ids
