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

"""Parent class for contextual path finding datasets."""

from typing import Any, Optional, TypedDict, Union

import tensorflow as tf

from semantic_routing.benchmark.query_engines import query_engine

# Graph types
NodeType = int
EdgeType = int
EdgeListType = Union[tuple[EdgeType], tuple[()]]
TERM: EdgeType = 0


class DatapointType(TypedDict):
  """Represents a (potentially labeled) datapoint in contextual path finding.

  Attributes:
    end: The final edge of the ground-truth path.
    query_text: A user query, presumably in natural language.
    query_data: The structured ground-truth data of the user query.
    parent: Pointer to dataset that generated this datapoint.
    edgelist: The prefix or entirety of a path. In the latter case, this
      sequence ends with TERM. This can either be ground-truth or predicted. We
      use the term edgelist instead of path as this list may not always be
      connected.
    candidates: Edges that can be appended to the predicted path.
    ground_truth: The ground-truth of what the next edge in the path is.
  """

  end: Optional[EdgeType]
  edgelist: EdgeListType
  query_data: query_engine.QueryDataType
  query_text: str
  parent: Any
  candidates: Union[tuple[EdgeType, Ellipsis], tuple[()]]
  ground_truth: Optional[EdgeType]


class FeatDatapointType(TypedDict):
  """A featurized view of a next-edge prediction datapoint."""

  token_ids: tf.Tensor
  type_ids: tf.Tensor
  input_mask: tf.Tensor
  aux_embeddings: tf.Tensor
  position_ids: tf.Tensor
  candidate_token_ids: tf.Tensor
  candidate_type_ids: tf.Tensor
  num_candidates: int
  candidate_aux_embeddings: tf.Tensor
  candidates: tf.Tensor  # If known, the ground-truth next edge should be first.


class EdgeListPredictionDataset(object):
  """Parent that all edge-list prediction dataset classes inherit from.

  A EdgeListPredictionDataset (ELPD) allows one to interact with and generate
  next-edge prediction datapoints.
  """

  aux_embeddings_dim: tuple[int, Ellipsis]
  total_vocab_size: int
  min_prefix_len: int = 0

  def sample_datapoint(
      self,
      labeled,
      split,
      prefix_len = -1,
      use_fresh = True,
  ):
    """Sample a next-edge prediction datapoint.

    The `edgelist` of datapoints sampled this way are a ground-truth prefix.

    Args:
      labeled: Return a datapoint with `ground_truth` labeled if True and None
        if False.
      split: Data split to sample from. 0 is the train set, 1 the validation, 2
        the test set.
      prefix_len: Length of the edgelist prefix. If 0, prefix_len is set to the
        min_prefix_len. If -1, prefix_len is randomly chosen uniformly from
        [min_prefix_len, len(ground-truth edgelist) - 1].
      use_fresh: If use_fresh is False, sample datapoints from the same task at
        different prefix_len until the task is exhausted.
    """
    raise NotImplementedError("")

  def add_edge_to_datapoint(
      self, datapoint, edge_to_add
  ):
    """Add a predicted next-edge to a datapoint.

    Add `edge_to_add` to the datapoint's `edgelist`, replaces `candidates` with
    edges that can follow `edge_to_add`, and clears the `ground_truth` field.

    Args:
      datapoint: Datapoint to be mutated.
      edge_to_add: New edge.
    """

    if datapoint["edgelist"] and datapoint["edgelist"][-1] == TERM:
      raise ValueError("Cannot add edge to datapoint with complete edge-list.")
    if edge_to_add not in datapoint["candidates"]:
      raise ValueError("Cannot add non-candidate edge to datapoint.")

    datapoint["ground_truth"] = None
    datapoint["edgelist"] = datapoint["edgelist"] + (edge_to_add,)  # type: ignore

    datapoint["candidates"] = self.get_candidates(datapoint["edgelist"])
    if edge_to_add == datapoint["end"]:
      datapoint["candidates"] += (TERM,)

  def featurize_datapoint(
      self, datapoint, pad = False
  ):
    """Produces a featurized copy of a readable datapoint.

    Args:
      datapoint: Datapoint to be featurized.
      pad: If True, datapoint is padded to universally consistent length.

    Returns:
      Featurized datapoint.
    """
    raise NotImplementedError("")

  def get_candidates(self, prefix):
    """Return candidate edges that could follow a certain prefix."""
    raise NotImplementedError("")

  def evaluate_datapoint(
      self, datapoint
  ):
    """Evaluate the edge-list prediction in a datapoint."""
    raise NotImplementedError("")

  def fast_score(
      self, datapoint
  ):
    """Evaluate the edge-list prediction in a datapoint."""
    raise NotImplementedError()
