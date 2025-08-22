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

"""Implements a dataset class for customized routing tasks."""

import random
from typing import Any, Union

import numpy as np
import tensorflow as tf

from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.graphs import road_graph
from semantic_routing.benchmark.query_engines import query_engine


CAND_PADDING = -1


class RoutingDataset(dataset.EdgeListPredictionDataset):
  """Dataset class for customized routing tasks."""

  poi_specs: utils.POISpecType
  vocab_offset: int

  # Datapoint sampling parameters
  min_prefix_len: int = 1
  max_sample_attempts: int = 10
  min_segments: int = 1
  max_segments: int = 30

  # Datapoint featurization length cutoffs
  cand_len: int = 8
  input_len: int = 256

  receptive_field_size: int = 512

  # Datapoint featurization type_id parameters
  cand_type: int = 1
  query_type: int = 2
  end_type: int = 3
  prefix_type: int = 4
  receptive_type: int = 5

  # Datapoint featurization token parameters
  term_token: int = dataset.TERM
  sep_token: int = 1
  mid_sep_token: int = 2
  placeholder_token: int = 3
  no_cand_token: int = 4
  cand_tokens: tuple[int, Ellipsis] = tuple(range(5, 5 + cand_len))

  def __init__(
      self,
      tokenizer,
      graph,
      engine,
      poi_specs,
      seed,
      receptive_field_size,
      poi_receptive_field_size,
      auto_simplify_datapoint = False,
      max_segments = 30,
      cand_len = 8,
  ):
    assert self.term_token == 0

    self.max_segments = max_segments
    self.auto_simplify_datapoint = auto_simplify_datapoint
    self.tokenizer = tokenizer
    self.road_graph = graph
    self.query_engine = engine
    self.cand_len = cand_len
    self.cand_tokens = tuple(range(5, 5 + cand_len))

    self.poi_specs = poi_specs
    self.seed = seed
    self.receptive_field_size = receptive_field_size
    self.poi_receptive_field_size = poi_receptive_field_size

    if self.input_len <= self.receptive_field_size:
      print(
          "Increasing input length from {} to {}.".format(
              self.input_len, self.receptive_field_size
          )
      )
      self.input_len = self.receptive_field_size

    self.rng = random.Random(
        self.seed
    )  # Internal RNG for datapoint randomness.

    self.vocab_offset = 6 + self.cand_len
    self._current_params = None
    self._current_datapoint = None

  def get_candidates(
      self, prefix
  ):
    """Return list of candidates for nodes that follow a prefix."""
    if not prefix:
      raise ValueError("prefix must not be empty.")
    if prefix[-1] == self.term_token:
      return ()
    return self.road_graph.get_reachable(prefix[-1])

  @property
  def datapoint_cached(self):
    return self._current_datapoint is not None

  def simplify_datapoint(
      self, datapoint
  ):
    assert len(datapoint["edgelist"]) == 1
    new_graph = self.road_graph.contract_graph(
        [datapoint["edgelist"][0], datapoint["end"]]
    )
    new_path_info = new_graph.get_shortest_path_len(
        datapoint["edgelist"][0],
        datapoint["end"],
        datapoint["query_data"],
        return_path=True,
    )
    assert new_path_info is not None
    new_edgelist = new_path_info[1]
    new_parent = RoutingDataset(
        tokenizer=self.tokenizer,
        graph=new_graph,
        engine=self.query_engine,
        poi_specs=self.poi_specs,
        seed=self.seed,
        receptive_field_size=self.receptive_field_size,
        poi_receptive_field_size=self.poi_receptive_field_size,
        auto_simplify_datapoint=False,
        max_segments=self.max_segments,
        cand_len=self.cand_len,
    )
    if datapoint["ground_truth"] is None:
      new_ground_truth = None
    else:
      new_ground_truth = new_edgelist[1]
    new_candidates = new_parent.get_candidates(datapoint["edgelist"])
    return dataset.DatapointType(
        end=datapoint["end"],
        query_text=datapoint["query_text"],
        query_data=datapoint["query_data"],
        edgelist=datapoint["edgelist"],
        ground_truth=new_ground_truth,
        candidates=new_candidates,
        parent=new_parent,
    )

  def sample_datapoint(
      self,
      labeled,
      split,
      prefix_len = -1,
      use_fresh = True,
  ):
    """Sample a next-edge prediction datapoint."""

    assert (
        prefix_len == -1 or prefix_len == 0 or prefix_len >= self.min_prefix_len
    )
    assert prefix_len <= self.max_segments - 1
    assert self.max_sample_attempts >= 1
    assert use_fresh or labeled

    if prefix_len == 0:
      prefix_len = self.min_prefix_len

    if (
        self._current_params != (labeled, split, self.min_prefix_len)
        or use_fresh
    ):  # Cache miss
      self._current_params = None
      end = None
      edgelist = None
      query_text = None
      query_data = None
      found = False
      parent = self
      for _ in range(self.max_sample_attempts):
        query_data, query_text = self.query_engine.sample_query(split, self.rng)
        start = self.rng.choice(self.road_graph.central_edges)
        end = self.road_graph.sample_noncentral_edge(split, self.rng)
        path_info = self.road_graph.get_shortest_path_len(
            start, end, query_data, return_path=True
        )
        if path_info is None:
          continue
        assert isinstance(path_info, tuple)
        _, edgelist = path_info
        if len(edgelist) < self.min_prefix_len:
          continue
        if len(edgelist) <= self.min_segments:
          continue
        if prefix_len != -1 and len(edgelist) < prefix_len:
          continue
        if len(edgelist) > self.max_segments:
          continue
        if self.auto_simplify_datapoint:
          new_graph = self.road_graph.contract_graph([start, end])
          new_path_info = new_graph.get_shortest_path_len(
              start, end, query_data, return_path=True
          )
          assert new_path_info is not None
          edgelist = new_path_info[1]
          parent = RoutingDataset(
              tokenizer=self.tokenizer,
              graph=new_graph,
              engine=self.query_engine,
              poi_specs=self.poi_specs,
              seed=self.seed,
              receptive_field_size=self.receptive_field_size,
              poi_receptive_field_size=self.poi_receptive_field_size,
              auto_simplify_datapoint=False,
              max_segments=self.max_segments,
              cand_len=self.cand_len,
          )
        found = True
        break
      if not found:
        raise TimeoutError("")
      edgelist += (self.term_token,)
    else:  # Cache hit, re-use datapoint from last time
      assert self._current_datapoint is not None
      end = self._current_datapoint["end"]
      query_text = self._current_datapoint["query_text"]
      query_data = self._current_datapoint["query_data"]
      edgelist = self._current_datapoint["edgelist"]
      parent = self._current_datapoint["parent"]

    if not use_fresh:  # Cache this datapoint
      prefix_len = len(edgelist) - 1
      ground_truth = edgelist[prefix_len]  # Ground truth for CURRENT datapoint
      edgelist = edgelist[:prefix_len]  # Edge list for CURRENT datapoint
      if prefix_len >= 2:
        self._current_datapoint = dataset.DatapointType(
            end=end,
            query_text=query_text,
            query_data=query_data,
            edgelist=edgelist,
            ground_truth=ground_truth,
            candidates=(),
            parent=parent,
        )
        self._current_params = (labeled, split, self.min_prefix_len)
      else:
        self._current_datapoint = None
        self._current_params = None
    else:  # Do not cache this datapoint; pick a random position
      if prefix_len == -1:
        prefix_len = self.rng.randint(self.min_prefix_len, len(edgelist) - 1)
      if labeled:
        ground_truth = edgelist[prefix_len]
      else:
        ground_truth = None
      edgelist = edgelist[:prefix_len]
      self._current_datapoint = None
      self._current_params = None

    candidates = parent.get_candidates(edgelist)
    if edgelist[-1] == end:
      candidates += (dataset.TERM,)

    return dataset.DatapointType(
        end=end,
        query_text=query_text,
        query_data=query_data,
        edgelist=edgelist,
        ground_truth=ground_truth,
        candidates=candidates,
        parent=parent,
    )

  def evaluate_datapoint(
      self, datapoint
  ):
    if dataset.TERM in datapoint["edgelist"]:
      assert datapoint["edgelist"].count(dataset.TERM) == 1
      assert datapoint["edgelist"][-1] == dataset.TERM
      edgelist = datapoint["edgelist"][:-1]
    else:
      edgelist = datapoint["edgelist"]

    metrics = self.road_graph.route_metrics(
        datapoint["query_data"], datapoint["end"], edgelist
    )
    score = (
        10000 * int(metrics["reaches_destination"])
        + 10000 * int(metrics["frac_pois_achieved"] > 0.99)
        - metrics["excess_penalty"]
    )
    metrics["score"] = score
    metrics["mask"] = int(
        metrics["reaches_destination"] and metrics["frac_pois_achieved"] > 0.99
    )
    return metrics

  def featurize_datapoint(
      self, datapoint, pad = False
  ):
    """Default featurization of datapoints."""
    assert datapoint["edgelist"][-1] != self.term_token
    empty_embedding = np.zeros((self.road_graph.embedding_dim,), dtype=np.int32)

    # Move ground-truth to the first position in `candidates`
    if datapoint["ground_truth"] is not None:
      candidates = [datapoint["ground_truth"]]
    else:
      candidates = []
    candidates += [
        cand
        for cand in datapoint["candidates"]
        if cand != datapoint["ground_truth"]
    ]
    # Drop candidates if there are too many
    candidates = candidates[: self.cand_len]
    num_candidates = len(candidates)
    # Assign each candidate a random ID
    candidate_ordering = {}
    for i, cand in enumerate(
        list(self.rng.sample(candidates, len(candidates)))
    ):
      candidate_ordering[cand] = i
    # Create candidate features
    candidate_tokens = []
    candidate_type_ids = []
    candidate_embeddings = []
    for cand in candidates:
      if cand == dataset.TERM:  # Termination candidate has no embedding.
        candidate_tokens.append(self.term_token)
        candidate_embeddings.append(empty_embedding)
      else:
        candidate_tokens.append(self.cand_tokens[candidate_ordering[cand]])
        candidate_embeddings.append(
            self.road_graph.get_edge_embedding(cand, datapoint["edgelist"][0])
        )
        # Use candidate number as token.
      candidate_type_ids.append(self.cand_type)

    # Generate main content
    tokens = []
    position_ids = []
    type_ids = []
    embeddings = []

    # Initialize the sequence with an empty separation token.
    tokens.append(self.sep_token)
    position_ids.append(0)
    type_ids.append(self.query_type)
    embeddings.append(empty_embedding)

    # Add queries to tokens
    query_tokens = [
        v + self.vocab_offset
        for v in self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(datapoint["query_text"])
        )
    ]
    query_tokens.append(self.sep_token)
    tokens += query_tokens
    position_ids += list(range(1, len(query_tokens) + 1))
    type_ids += [self.query_type] * len(query_tokens)
    embeddings += [empty_embedding] * len(query_tokens)

    # Add end to tokens
    end_tokens = [self.placeholder_token]
    end_embeddings = [
        self.road_graph.get_edge_embedding(
            datapoint["end"], datapoint["edgelist"][0]
        )
    ]  # End embeddings are w.r.t. start.
    end_tokens.append(self.sep_token)
    end_embeddings.append(empty_embedding)
    tokens += end_tokens
    position_ids += [i + self.input_len for i in range(len(end_tokens))]
    type_ids += [self.end_type] * len(end_tokens)
    embeddings += end_embeddings

    # Add prefix to tokens
    prefix_tokens = []
    prefix_embeddings = []
    for edge in datapoint["edgelist"]:
      prefix_tokens.append(self.placeholder_token)
      prefix_embeddings.append(
          self.road_graph.get_edge_embedding(edge, datapoint["edgelist"][0])
      )  # Prefix embeddings are w.r.t. start.
    prefix_tokens.append(self.sep_token)
    prefix_embeddings.append(empty_embedding)
    tokens += prefix_tokens
    position_ids += [i + self.input_len * 2 for i in range(len(prefix_tokens))]
    type_ids += [self.prefix_type] * len(prefix_tokens)
    embeddings += prefix_embeddings

    # Add candidates to tokens
    tokens += candidate_tokens + [self.sep_token]
    position_ids += [0 + self.input_len * 3] * (len(candidate_tokens) + 1)
    type_ids += candidate_type_ids + [self.cand_type]
    embeddings += candidate_embeddings + [empty_embedding]

    # Add receptive field
    receptive_tokens = []
    receptive_embeddings = []
    current_edge = datapoint["edgelist"][-1]
    for _, (edge, cand_edge) in enumerate(
        self.road_graph.get_receptive_field(
            current_edge,
            self.receptive_field_size,
            includes=[datapoint["end"]],
        )  # Receptive embeddings are w.r.t. HEAD.
    ):
      if edge in candidate_ordering:
        continue
      elif cand_edge is None or cand_edge not in candidate_ordering:
        receptive_tokens.append(self.no_cand_token)
      else:
        receptive_tokens.append(self.cand_tokens[candidate_ordering[cand_edge]])
      receptive_embeddings.append(
          self.road_graph.get_edge_embedding(edge, datapoint["edgelist"][0])
      )
    tokens += receptive_tokens + [self.sep_token]
    position_ids += [
        i + self.input_len * 4 for i in range(len(receptive_tokens) + 1)
    ]
    type_ids += [self.receptive_type] * (len(receptive_tokens) + 1)
    embeddings += receptive_embeddings + [empty_embedding]

    # Pad things out
    tokens = np.array(tokens, dtype=np.int32)
    position_ids = np.array(position_ids, dtype=np.int32)
    type_ids = np.array(type_ids, dtype=np.int32)
    embeddings = np.array(embeddings, dtype=np.float32)
    if pad:
      tokens, input_mask = utils.pad_seq(
          tokens, self.input_len, pad_from="right"
      )
      position_ids, _ = utils.pad_seq(
          position_ids, self.input_len, pad_from="right"
      )
      type_ids, _ = utils.pad_seq(type_ids, self.input_len, pad_from="right")
      embeddings, _ = utils.pad_seq(
          embeddings,
          self.input_len,
          pad_from="right",
          fill_value=0,
      )
    else:
      input_mask = np.ones_like(tokens)

    candidate_tokens = np.array(candidate_tokens, dtype=np.int32)
    candidate_type_ids = np.array(candidate_type_ids, dtype=np.int32)
    candidate_embeddings = np.array(candidate_embeddings, dtype=np.float32)
    if pad:
      candidate_tokens, _ = utils.pad_seq(
          candidate_tokens,
          self.cand_len,
          pad_from="right",
      )
      candidate_type_ids, _ = utils.pad_seq(
          candidate_type_ids,
          self.cand_len,
          pad_from="right",
      )
      candidate_embeddings, _ = utils.pad_seq(
          candidate_embeddings,
          self.cand_len,
          pad_from="right",
          fill_value=0,
      )
      if not candidates:
        candidate_embeddings = np.zeros(
            (
                self.cand_len,
                self.road_graph.embedding_dim,
            ),
            dtype=np.float32,
        )
      candidates = np.array(
          candidates + [CAND_PADDING] * (self.cand_len - len(candidates)),
          dtype=np.int32,
      )
      assert candidates.shape == candidate_tokens.shape
    else:
      candidates = np.array(candidates, dtype=np.int32)

    return dataset.FeatDatapointType(
        token_ids=tf.convert_to_tensor(tokens),
        input_mask=tf.convert_to_tensor(input_mask),
        type_ids=tf.convert_to_tensor(type_ids),
        aux_embeddings=tf.convert_to_tensor(embeddings),
        position_ids=tf.convert_to_tensor(position_ids),
        candidate_token_ids=tf.convert_to_tensor(candidate_tokens),
        candidate_type_ids=tf.convert_to_tensor(candidate_type_ids),
        candidate_aux_embeddings=tf.convert_to_tensor(candidate_embeddings),
        candidates=tf.convert_to_tensor(candidates),
        num_candidates=num_candidates,
    )
