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

"""Contains classes for the road network graphs of major US cities."""

import ast
import random
from typing import Optional

import networkx as nx
import numpy as np

from semantic_routing.benchmark import config
from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.graphs import networkx_graph


COORD_RESCALE = 10
TRAFFIC_BINOMIAL_DENOM = 10
TRAFFIC_BINOMIAL_P = 0.1
CITY_SUBGRAPH_MAX_ATTEMPTS = 10
CITY_EDGES_CACHE_MAX_SIZE = 3
TRAIN_CITIES = [
    (("NewYork", 22), ("SantaBarbara", 2), ("Miami", 3)),
    (("CambridgeMa", 18), ("Providence", 3), ("SanFrancisco", 4)),
    (("Orlando", 17), ("Boulder", 4), ("Albuquerque", 4)),
    (("Portland", 17), ("Madison", 4), ("NewOrleans", 6)),
    (("Chicago", 15), ("PaloAlto", 6), ("Philadelphia", 7)),
    (("Austin", 15), ("SanJose", 8), ("Memphis", 13)),
    (("Denver", 14), ("WashingtonDC", 13)),
    (("Seattle", 14), ("Sacramento", 10), ("SantaCruz", 8)),
]
TEST_CITY = ("Berkeley", 8)

CITY_FILE_SEP = "|||"
CITY_FILE_KEYS = (
    "highway",
    "travel_time",
    "lanes",
    "length",
    "poi_node_names",
    "poi_type_ids",
    "poi_node_ids",
    "osmid",
    "level",
    "u_lat",
    "u_lon",
    "v_lat",
    "v_lon",
    "pen_v",
)

CITY_EDGES_CACHE = {}


class CityGraph(networkx_graph.NetworkXRoadGraph):
  """Implements urban city road graphs using data from OpenStreetMap."""

  # Graph construction parameters
  seed: int = 0

  # Data contamination parameter
  splits: tuple[float, Ellipsis] = (1, 0.0, 0.0)
  path: str = config.OSM_PATH
  city_group: tuple[tuple[str, int], Ellipsis]

  def __init__(
      self,
      poi_specs,
      num_nodes,
      seed = None,
      city_group_seed = None,
      splits = None,
      use_test_city = False,
  ):
    """Initialize road graph."""
    self.num_nodes = num_nodes
    self.poi_specs = poi_specs

    if use_test_city:
      self.city_group = (TEST_CITY,)
    else:
      assert city_group_seed is not None
      rng = random.Random(city_group_seed)
      self.city_group = rng.choice(TRAIN_CITIES)

    if seed is not None:
      self.seed = seed
    if splits is not None:
      self.splits = splits

    self.rng = random.Random(self.seed)
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

    # Find ground-truth shortest paths without POI constraints
    self.shortest_path_lens = {}
    self.query_shortest_path_lens = {}
    self.query_shortest_paths = {}
    self._divide_dataset()

    self.embedding_dim = (
        2
        + 2
        + 4
        + 1
        + 2
        + (max(self.poi_type_id_to_edge) + 1)
        + (len(utils.ROAD_VALUES) + 1)
    )

  def _get_graph(self):
    """Generate a 2d grid graph."""

    # Select a city
    city, total = self.rng.choices(
        self.city_group, [w for _, w in self.city_group]
    )[0]

    type_id_whitelist = [spec["poi_type_id"] for spec in self.poi_specs[0]]

    # Grab edges from dumped edgelists
    if city in CITY_EDGES_CACHE:
      edges = CITY_EDGES_CACHE[city]
    else:
      edges = []
      for i in range(total):
        with open(self.path + "{}_edgelist_{}.list".format(city, i), "r") as f:
          for line in f:
            items = line.strip().split(CITY_FILE_SEP)
            assert len(items) == len(CITY_FILE_KEYS) + 2
            source = (int(items[0]), 0)
            target = (int(items[1]), 0)
            data = {}
            for i, key in enumerate(CITY_FILE_KEYS):
              item = items[i + 2]
              if item == "None":
                item = None
                if key in ("poi_node_names", "poi_type_ids", "poi_node_ids"):
                  item = []
                else:
                  assert key in ("lanes", "pen_v")
              elif key == "lanes":
                if "---" in str(item):
                  item = item.replace("---", "")
                try:
                  item = int(float(item))
                except ValueError:
                  item = None
              elif key in ("lanes", "osmid", "level"):
                item = int(item)
              elif key == "highway":
                item = str(item)
              elif key in (
                  "travel_time",
                  "length",
              ):
                item = float(item)
              elif key in (
                  "u_lat",
                  "u_lon",
                  "v_lat",
                  "v_lon",
              ):
                item = COORD_RESCALE * float(item)
              elif key == "pen_v":
                item = int(item)
                assert item == -1
                item = None
              elif key in ("poi_node_names", "poi_type_ids", "poi_node_ids"):
                item = ast.literal_eval(item)
              data[key] = item

            new_poi_node_names = []
            new_poi_type_ids = []
            new_poi_node_ids = []
            for node_name, type_ids, node_id in zip(
                data["poi_node_names"],
                data["poi_type_ids"],
                data["poi_node_ids"],
            ):
              type_ids = tuple(
                  type_id
                  for type_id in type_ids
                  if type_id in type_id_whitelist
              )
              if not type_ids:
                continue
              new_poi_node_names.append(node_name)
              new_poi_type_ids.append(type_ids)
              new_poi_node_ids.append(node_id)
            data["poi_node_names"] = new_poi_node_names
            data["poi_type_ids"] = new_poi_type_ids
            data["poi_node_ids"] = new_poi_node_ids
            edges.append((source, target, data))
      if len(CITY_EDGES_CACHE) >= CITY_EDGES_CACHE_MAX_SIZE:
        CITY_EDGES_CACHE.pop(list(CITY_EDGES_CACHE.keys())[0])
      CITY_EDGES_CACHE[city] = edges

    # Grab a subgraph
    nx_graph = nx.DiGraph(edges)
    central_node = None
    nodes = []
    for _ in range(CITY_SUBGRAPH_MAX_ATTEMPTS):
      central_node = self.rng.choice(list(nx_graph.nodes))
      nodes = [central_node]
      for (_, v), _ in zip(
          nx.bfs_edges(nx_graph, central_node), range(self.num_nodes - 1)
      ):
        nodes.append(v)
      if len(nodes) == self.num_nodes:
        break
    assert len(nodes) == self.num_nodes

    nx_graph = nx.DiGraph(nx_graph.subgraph(nodes))

    # Keep record of POIs
    poi_type_id_to_edge = {}
    general_pois, specialized_pois = self.poi_specs
    poi_type_info = {}
    for poi_info in general_pois + sum(specialized_pois.values(), []):
      poi_type_id_to_edge[poi_info["poi_type_id"]] = []
      poi_type_info[poi_info["poi_type_id"]] = poi_info
    for u, v, data in nx_graph.edges(data=True):
      for type_ids in data["poi_type_ids"]:
        for type_id in type_ids:
          if (u, v) not in poi_type_id_to_edge:
            poi_type_id_to_edge[type_id].append((u, v))

    for u, v, data in nx_graph.edges(data=True):
      nx_graph[u][v]["current_travel_time"] = data["travel_time"] * (
          1
          + 2
          * self.np_rng.binomial(TRAFFIC_BINOMIAL_DENOM, TRAFFIC_BINOMIAL_P)
          / TRAFFIC_BINOMIAL_DENOM
      )
      nx_graph[u][v]["edges"] = [(u, v)]

    # Track internal edges
    edge_to_internal = {}
    for i, (u, v) in enumerate(nx_graph.edges()):
      # Add "live" edge attributes
      edge_to_internal[i + 1] = (u, v)
      nx_graph[u][v]["id"] = i + 1

    return (
        nx_graph,
        edge_to_internal,
        poi_type_id_to_edge,
        central_node,
    )
