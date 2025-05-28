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

import random
from typing import Any, Optional

import networkx as nx
import numpy as np

from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.graphs import networkx_graph


DEFAULT_LENGTH = 10
DEFAULT_LANES = 1
HIGHWAY_TRAVEL_TIME = 1
COORD_SCALE = 0.01
TRAFFIC_BINOMIAL_DENOM = 10
TRAFFIC_BINOMIAL_P = 0.1


class GridGraph(networkx_graph.NetworkXRoadGraph):
  """Implements a synthetic 2d grid road graph.

  Each node is connected to its 2-4 cardinal neighbors. Each edge is assigned a
  random positive duration value sampled from the Poisson distribution. With
  small probability, an edge is assigned a POI attribute. Highway edges
  are randomly added, sampled uniformly from all pairs of nodes that are at
  least an L1 distance of 3 apart (by grid topology, not duration). All edges
  are bidirectional, as are highways and POI designations. Edge IDs are
  directional.
  """

  # Graph construction parameters
  seed: int = 0
  duration_poisson_mu: int = 5  # Poisson parameter for duration distribution.
  poi_prob: float = 0.002  # Prob that a given edge is assigned a given POI.
  highway_density_node: int = 50  # Num nodes / this = num highways.
  min_node_gap: float = 3  # Distance between highways

  # Data contamination parameters
  splits: tuple[float, Ellipsis] = (1, 0.0, 0.0)

  def __init__(
      self,
      poi_specs,
      num_nodes,
      seed = None,
      duration_poisson_mu = None,
      poi_prob = None,
      highway_density_node = None,
      min_node_gap = None,
      splits = None,
  ):
    """Initialize road graph."""
    self.num_nodes = num_nodes
    self.poi_specs = poi_specs

    if seed is not None:
      self.seed = seed
    if duration_poisson_mu is not None:
      self.duration_poisson_mu = duration_poisson_mu
    if poi_prob is not None:
      self.poi_prob = poi_prob
    if highway_density_node is not None:
      self.highway_density_node = highway_density_node
    if min_node_gap is not None:
      self.min_node_gap = min_node_gap
    if splits is not None:
      self.splits = splits

    self.rng = random.Random()
    self.rng.seed(self.seed)
    self.np_rng = np.random.RandomState(self.seed)

    (
        self.nx_graph,
        self.edge_to_internal,
        self.poi_type_id_to_edge,
        self.central_node,
    ) = self._get_graph()
    self.edge_from_internal: dict[
        networkx_graph.InternalEdgeType, dataset.EdgeType
    ] = {}
    for edge, internal_edge in self.edge_to_internal.items():
      self.edge_from_internal[internal_edge] = edge

    self.embedding_dim = (
        2
        + 2
        + 4
        + 1
        + 2
        + (max(self.poi_type_id_to_edge) + 1)
        + (len(utils.ROAD_VALUES) + 1)
    )

    # Find ground-truth shortest paths without POI constraints
    self.shortest_path_lens = {}
    self.query_shortest_path_lens = {}
    self.query_shortest_paths = {}
    self._divide_dataset()

  def _add_highways(
      self,
      grid_graph,
      num_highways,
      attrs,
      edge_to_internal,
  ):
    """Given a 2d-grid NetworkX graph, randomly add highway edges."""
    if num_highways == 0:
      return

    candidate_edges = []
    for u in grid_graph.nodes:
      for v in grid_graph.nodes:
        if u == v:
          continue
        l1_dist = np.linalg.norm((u[0] - v[0], u[1] - v[1]), ord=1)
        if l1_dist >= self.min_node_gap:
          candidate_edges.append((u, v))  # Highways are unidirectional
    highway_id_start = len(nx.edges(grid_graph)) + 1
    assert highway_id_start not in edge_to_internal
    self.np_rng.shuffle(candidate_edges)  # Uniformly sample without replacement
    new_edges = candidate_edges[:num_highways]

    new_edges_both_dirs = []
    for edge in new_edges:
      if edge not in new_edges_both_dirs:  # Add highways in both directions
        new_edges_both_dirs.append(edge)
      rev_edge = (edge[1], edge[0])
      if rev_edge not in new_edges_both_dirs:
        new_edges_both_dirs.append(rev_edge)

    grid_graph.add_edges_from(new_edges_both_dirs)
    for i, (u, v) in enumerate(new_edges_both_dirs):
      edge = (u, v)
      new_id = highway_id_start + i
      assert new_id not in edge_to_internal
      edge_to_internal[new_id] = (u, v)
      attrs[edge] = {
          "id": new_id,
          "highway": "motorway",
          "travel_time": HIGHWAY_TRAVEL_TIME,
          "current_travel_time": HIGHWAY_TRAVEL_TIME * (
              1
              + 2
              * self.np_rng.binomial(TRAFFIC_BINOMIAL_DENOM, TRAFFIC_BINOMIAL_P)
              / TRAFFIC_BINOMIAL_DENOM
          ),
          "level": 1,
          "lanes": DEFAULT_LANES,
          "length": DEFAULT_LENGTH,
          "u_lat": edge[0][0] * COORD_SCALE,
          "u_lon": edge[0][1] * COORD_SCALE,
          "v_lat": edge[1][0] * COORD_SCALE,
          "v_lon": edge[1][1] * COORD_SCALE,
          "poi_node_names": [],
          "poi_type_ids": [],
          "poi_node_ids": [],
      }
      attrs[edge]["edges"] = [edge]

  def _get_graph(self):
    """Generate a 2d grid graph."""

    # Initialize grid graph
    num_rows = int(np.sqrt(self.num_nodes))
    num_cols = int(np.sqrt(self.num_nodes))
    num_highways = self.num_nodes // self.highway_density_node

    num_nodes = num_cols * num_rows
    nx_graph = nx.grid_2d_graph(num_cols, num_rows)
    nodes_list = list(nx_graph.nodes)
    assert num_nodes == len(nodes_list)
    nx_graph = nx_graph.to_directed()

    # Assign POIs to edges
    general_pois, specialized_pois = self.poi_specs
    num_pois = 0
    edge_to_poi = {}
    poi_type_id_to_edge = {}
    for edge in nx.edges(nx_graph):
      if edge in edge_to_poi:
        continue
      edge_to_poi[edge] = []
      edge_to_poi[(edge[1], edge[0])] = []
      for poi_info in general_pois + sum(specialized_pois.values(), []):
        if poi_info["poi_type_id"] not in poi_type_id_to_edge:
          poi_type_id_to_edge[poi_info["poi_type_id"]] = []
        if self.rng.random() < self.poi_prob:
          num_pois += 1
          edge_to_poi[edge].append((num_pois, (poi_info["poi_type_id"],)))
          edge_to_poi[(edge[1], edge[0])].append(
              (num_pois, (poi_info["poi_type_id"],))
          )
          poi_type_id_to_edge[poi_info["poi_type_id"]].append(edge)
          poi_type_id_to_edge[poi_info["poi_type_id"]].append(
              (edge[1], edge[0])
          )
    for poi_info in general_pois + sum(specialized_pois.values(), []):
      if not poi_type_id_to_edge[poi_info["poi_type_id"]]:
        edge = self.rng.choice(list(nx.edges(nx_graph)))
        num_pois += 1
        edge_to_poi[edge].append((num_pois, (poi_info["poi_type_id"],)))
        edge_to_poi[(edge[1], edge[0])].append(
            (num_pois, (poi_info["poi_type_id"],))
        )
        poi_type_id_to_edge[poi_info["poi_type_id"]].append(edge)
        poi_type_id_to_edge[poi_info["poi_type_id"]].append((edge[1], edge[0]))

    # Add edge attributes
    edge_to_internal = {}
    edge_attrs = {}
    for i, edge in enumerate(nx.edges(nx_graph)):
      edge_to_internal[i + 1] = edge
      travel_time = float(
          self.np_rng.poisson(lam=self.duration_poisson_mu, size=1)[0]
      )
      current_travel_time = travel_time * (
          1
          + 2
          * self.np_rng.binomial(TRAFFIC_BINOMIAL_DENOM, TRAFFIC_BINOMIAL_P)
          / TRAFFIC_BINOMIAL_DENOM
      )
      edge_attrs[edge] = {
          "id": i + 1,
          "travel_time": travel_time,
          "current_travel_time": current_travel_time,
          "level": 1,
          "lanes": DEFAULT_LANES,
          "length": DEFAULT_LENGTH,
          "highway": "residential",
          "u_lat": edge[0][0] * COORD_SCALE,
          "u_lon": edge[0][1] * COORD_SCALE,
          "v_lat": edge[1][0] * COORD_SCALE,
          "v_lon": edge[1][1] * COORD_SCALE,
          "poi_node_names": ["" for _ in edge_to_poi[edge]],
          "poi_type_ids": [type_ids for (_, type_ids) in edge_to_poi[edge]],
          "poi_node_ids": [node_id for (node_id, _) in edge_to_poi[edge]],
      }
      edge_attrs[edge]["edges"] = [edge]

    self._add_highways(
        nx_graph,
        num_highways,
        edge_attrs,
        edge_to_internal,
    )
    nx.set_edge_attributes(nx_graph, edge_attrs)

    central_node = (
        int(np.sqrt(self.num_nodes) / 2),
        int(np.sqrt(self.num_nodes) / 2),
    )

    return (
        nx_graph,
        edge_to_internal,
        poi_type_id_to_edge,
        central_node,
    )
