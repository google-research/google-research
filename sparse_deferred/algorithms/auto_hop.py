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

"""Implements recursive propagate method to get features of connected nodes."""

from typing import Any
import tensorflow as tf
import sparse_deferred as sd


def recursive_propagate(
    feat_dict,
    adjacency_dict,
    query_key,
    hops = 3,
    bidirectional=False,
):
  """Recurses over all nodes that are connected via the adjacency matrices.

  Args:
    feat_dict: Dictionary of features
    adjacency_dict: Dictionary of adjacencies
    query_key: Start recursion point
    hops: Max number of hops
    bidirectional: Allow one node go back to the previous node

  Returns:
    List of tensors with transformed features from connected nodes and edges.
  """
  if hops == 0:
    return [feat_dict[query_key]]
  all_terms = []
  for (key1, key2), adj_list in sorted(adjacency_dict.items()):
    for adj in adj_list:
      if key1 == query_key:
        terms = recursive_propagate(
            feat_dict, adjacency_dict, key2, hops - 1, bidirectional
        )
        terms = [adj @ term for term in terms]
        all_terms.extend(terms)
      elif bidirectional and key2 == query_key:
        terms = recursive_propagate(
            feat_dict, adjacency_dict, key1, hops - 1, bidirectional
        )
        terms = [adj.T @ term for term in terms]
        all_terms.extend(terms)
  all_terms.append(feat_dict[query_key])
  return all_terms
