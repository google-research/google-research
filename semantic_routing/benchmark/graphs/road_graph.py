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

"""Contains classes for representing road network graphs."""

from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np

from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.query_engines import query_engine


RoadType = str


class RoadGraph(object):
  """Parent that all road network graph classes inherit from."""

  embedding_dim: int

  road_types: tuple[RoadType, Ellipsis]
  poi_specs: utils.POISpecType = ([], {})

  def contract_graph(
      self,
      preserve_edges,
  ):
    raise NotImplementedError()

  def get_edge_embedding(
      self, edge, ego_edge
  ):
    """Return the embedding of an edge."""
    raise NotImplementedError()

  def get_poi_embedding(
      self, poi, ego_edge
  ):
    """Return the embedding of a POI."""
    raise NotImplementedError()

  def get_edge_pois(
      self, edge
  ):
    """Returns POIs associated with an edge."""
    raise NotImplementedError()

  def get_road_type(self, edge):
    """Returns the road type of an edge."""
    raise NotImplementedError()

  def get_receptive_field(
      self,
      ego_edge,
      receptive_field_size,
      includes = None,
  ):
    """Returns receptive field around an ego edge.

    Args:
      ego_edge: Ego edge to return the neighborhood of.
      receptive_field_size: Number of edges to return total.
      includes: Edges to include in the receptive field.

    Returns:
      List of pairs of edges. The first edge is an edge in the receptive field.
      The second is the candidate that usually leads to the first fastest.
    """
    raise NotImplementedError()

  def get_reachable(
      self, edge
  ):
    """Returns the immediate reachable edges from the end of `edge`."""
    raise NotImplementedError()

  def get_shortest_path_len(
      self,
      start,
      end,
      query_data = None,
      return_path = False,
  ):
    """Returns best path between two edges given a query."""
    raise NotImplementedError()

  @property
  def central_edges(self):
    """Returns the central edges."""
    raise NotImplementedError()

  def sample_noncentral_edge(
      self, split, rng
  ):
    """Sample a noncentral edge."""
    raise NotImplementedError()

  def route_metrics(
      self,
      query_data,
      end,
      route,
  ):
    """Return a dictionary of evaluation metrics for a given route."""
    raise NotImplementedError()
