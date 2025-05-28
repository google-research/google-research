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

"""Implements a road graph for networkx graphs."""

import copy
import itertools
import random
from typing import Optional, Union
import networkx as nx
import numpy as np
from semantic_routing.benchmark import utils
from semantic_routing.benchmark.datasets import dataset
from semantic_routing.benchmark.graphs import road_graph
from semantic_routing.benchmark.query_engines import query_engine

InternalNodeType = tuple[int, int]
InternalEdgeType = tuple[InternalNodeType, InternalNodeType]

COORD_RESCALE = 40
LENGTH_RESCALE = 0.01
LANE_RESCALE = 0.1
TRAVEL_TIME_RESCALE = 0.1
MAX_SEQS = 30
BUDGET_SCALE = 1.1
TOO_MANY_EDGES = 10000


def combine_edges(u_data, v_data, method="left"):
  """Return the metadata that results from merging two road segments."""
  if method == "left":
    main_data = u_data
  elif method == "right":
    main_data = v_data
  else:
    raise ValueError("Unknown method: %s" % method)
  u_osmid = u_data["osmid"]
  v_osmid = v_data["osmid"]
  if not isinstance(u_osmid, list):
    u_osmid = [u_osmid]
  if not isinstance(v_osmid, list):
    v_osmid = [v_osmid]
  return {
      "highway": main_data["highway"],
      "travel_time": u_data["travel_time"] + v_data["travel_time"],
      "current_travel_time": (
          u_data["current_travel_time"] + v_data["current_travel_time"]
      ),
      "length": u_data["length"] + v_data["length"],
      "lanes": main_data["lanes"],
      "pen_v": None,
      "v_lat": v_data["v_lat"],
      "v_lon": v_data["v_lon"],
      "u_lat": u_data["u_lat"],
      "u_lon": u_data["u_lon"],
      "level": main_data["level"],
      "osmid": u_osmid + v_osmid,
      "poi_type_ids": main_data["poi_type_ids"],
      "poi_node_names": main_data["poi_node_names"],
      "poi_node_ids": main_data["poi_node_ids"],
      "edges": u_data["edges"] + v_data["edges"],
  }


def contract_nodes(nx_graph, must_keep=None):
  """Return a copy of a NetworkX graph with uninformative nodes contracted."""
  removed = 0
  must_keep = must_keep or []
  for node, _ in sorted(nx_graph.degree(), key=lambda x: x[1]):
    # Block if required.
    if node in must_keep:
      continue

    # Block if node too informative.
    road_types = []
    pois = []
    lanes = []
    for _, _, data in list(nx_graph.in_edges(node, data=True)) + list(
        nx_graph.out_edges(node, data=True)
    ):
      road_types.append(
          "small"
          if data["highway"] in utils.SMALL_ROAD_VALUES
          else data["highway"]
      )
      lanes.append(data["lanes"] or 1)
      pois += data["poi_type_ids"]
    if len(set(road_types)) > 1:
      continue
    if len(set(lanes)) > 1:
      continue
    if pois:
      continue

    # Block if too many edges would be created.
    lens_u = []
    lens_v = []
    for u, _ in nx_graph.in_edges(node):
      lens_u.append(nx_graph.degree(u))
    for _, v in nx_graph.out_edges(node):
      lens_v.append(nx_graph.degree(v))
    if lens_u and lens_v:
      if len(lens_u) + max(lens_v) - 1 > MAX_SEQS:
        continue
      if len(lens_v) + max(lens_u) - 1 > MAX_SEQS:
        continue

    #
    edges = {}
    for u, _, u_data in nx_graph.in_edges(node, data=True):
      for _, v, v_data in nx_graph.out_edges(node, data=True):
        if u == v:
          continue
        data = combine_edges(u_data, v_data)
        if (u, v) not in edges:
          edges[(u, v)] = None
        if (
            edges[(u, v)] is not None
            and edges[(u, v)]["current_travel_time"]
            <= data["current_travel_time"]
        ):
          continue
        existing_data = nx_graph.get_edge_data(u, v, default=None)
        if existing_data is not None:
          if existing_data["poi_type_ids"]:
            continue
          existing_highway = (
              "small"
              if existing_data["highway"] in utils.SMALL_ROAD_VALUES
              else existing_data["highway"]
          )
          existing_lanes = existing_data["lanes"] or 1
          if existing_lanes != lanes[0]:
            continue
          if existing_highway != road_types[0]:
            continue
          if (
              existing_data["current_travel_time"]
              <= data["current_travel_time"]
          ):
            continue
        edges[(u, v)] = data
    if any([data is None for data in edges.values()]):
      continue
    for (u, v), data in edges.items():
      nx_graph.add_edge(u, v, **data)
    nx_graph.remove_node(node)
    removed += 1
  return removed


class NetworkXRoadGraph(road_graph.RoadGraph):
  """A road graph implemented with a NetworkX graph.

  It is assumed that the NetworkX graph is bidirectional and has the following
  data for each edge: travel_time (float), current_travel_time (float),
  level (int), length (float), lanes (int), u_lat (float),
  u_lon (float), v_lat (float), v_lon (float), edges (list[edgetypes]),
  poi_node_names
  (list[str]) poi_type_ids (list[int]), poi_node_ids (list[int]).
  """

  nx_graph: nx.DiGraph
  central_node: InternalNodeType
  edge_to_internal: dict[dataset.EdgeType, InternalEdgeType]
  edge_from_internal: dict[InternalEdgeType, dataset.EdgeType]
  splits: tuple[int, Ellipsis]
  embedding_dim: int
  rng: random.Random
  shortest_path_lens: dict[
      tuple[str, InternalNodeType, InternalNodeType], float
  ]

  poi_type_id_to_edge: dict[query_engine.POIType, list[InternalEdgeType]]
  query_shortest_path_lens: dict[
      tuple[
          str,
          Union[tuple[()], tuple[tuple[query_engine.POIType, Ellipsis], Ellipsis]],
          InternalNodeType,
          InternalNodeType,
      ],
      Optional[float],
  ]
  query_shortest_paths: dict[
      tuple[
          str,
          Union[tuple[()], tuple[tuple[query_engine.POIType, Ellipsis], Ellipsis]],
          InternalNodeType,
          InternalNodeType,
      ],
      tuple[dataset.EdgeType, Ellipsis],
  ]

  def get_poi_embedding(
      self, poi, ego_edge
  ):
    return np.zeros((self.embedding_dim,), dtype=np.float32)

  def get_edge_embedding(
      self, edge, ego_edge
  ):
    u, v = self.edge_to_internal[edge]
    u_ego, v_ego = self.edge_to_internal[ego_edge]
    edge_data = self.nx_graph.get_edge_data(u, v)
    ego_edge_data = self.nx_graph.get_edge_data(u_ego, v_ego)
    if edge == ego_edge:
      dist = 0
      hop_dist = 0
    else:
      dist = float(self.get_node_shortest_path_len("", v_ego, u))
      hop_dist = float(self.get_node_shortest_path_len("hop", v_ego, u))

    lanes = edge_data["lanes"]
    length = edge_data["length"]

    pois = [0] * (max(self.poi_type_id_to_edge) + 1)
    for _, poi_type_ids in self.get_edge_pois(edge):
      for poi_type_id in poi_type_ids:
        pois[poi_type_id] = 1

    road_values = [0] * (len(utils.ROAD_VALUES) + 1)
    road_values[utils.ROAD_VALUES.index(self.get_road_type(edge))] = 1

    embedding = [
        0.01 * dist,
        0.1 * hop_dist,
        TRAVEL_TIME_RESCALE
        * edge_data["travel_time"],  # Usual travel time, usually 0.3-10
        TRAVEL_TIME_RESCALE
        * edge_data["current_travel_time"],  # Current travel time
        LENGTH_RESCALE * length,  # Length, usually 1-100
        LANE_RESCALE * lanes if lanes else -1,  # Number of lanes, usual 1 - 10
        COORD_RESCALE
        * (edge_data["u_lat"] - ego_edge_data["v_lat"]),  # 1 mi ~= 0.015 deg
        COORD_RESCALE * (edge_data["u_lon"] - ego_edge_data["v_lon"]),
        COORD_RESCALE * (edge_data["v_lat"] - ego_edge_data["v_lat"]),
        COORD_RESCALE * (edge_data["v_lon"] - ego_edge_data["v_lon"]),
        *pois,
        *road_values,
        1,  # Flag
    ]
    return np.array(embedding, dtype=np.float32)

  def contract_graph(
      self,
      preserve_edges,
  ):
    new_graph = copy.copy(self)
    new_graph.query_shortest_paths = {}
    new_graph.query_shortest_path_lens = {}
    new_graph.shortest_path_lens = {}
    new_graph.nx_graph = copy.deepcopy(self.nx_graph)
    new_graph.edge_to_internal = copy.deepcopy(self.edge_to_internal)
    new_graph.edge_from_internal = copy.deepcopy(self.edge_from_internal)

    # Grab a subgraph
    must_keep = []
    for edge in preserve_edges:
      must_keep.append(self.edge_to_internal[edge][0])
      must_keep.append(self.edge_to_internal[edge][1])
    removed = 1
    while removed:
      removed = contract_nodes(new_graph.nx_graph, must_keep=must_keep)
    idx_start = max(new_graph.edge_from_internal.values()) + 1
    for edge in new_graph.nx_graph.edges:
      if edge not in new_graph.edge_from_internal:
        new_graph.edge_from_internal[edge] = idx_start
        new_graph.edge_to_internal[idx_start] = edge
        idx_start += 1
    return new_graph

  def get_edge_pois(
      self, edge
  ):
    """Returns POIs associated with an edge."""
    u, v = self.edge_to_internal[edge]
    edge_data = self.nx_graph.get_edge_data(u, v)
    return tuple(zip(edge_data["poi_node_ids"], edge_data["poi_type_ids"]))

  def get_road_type(self, edge):
    """Returns the road type of an edge."""
    u, v = self.edge_to_internal[edge]
    edge_data = self.nx_graph.get_edge_data(u, v)
    if edge_data["highway"] not in utils.ROAD_VALUES:
      return "unknown"
    return edge_data["highway"]

  def get_reachable(
      self,
      edge,
  ):
    _, v = self.edge_to_internal[edge]
    edges = self.nx_graph.out_edges(v)
    return tuple(self.edge_from_internal[e] for e in edges)

  def get_receptive_field(
      self,
      ego_edge,
      receptive_field_size,
      includes = None,
  ):
    _, ego_v = self.edge_to_internal[ego_edge]

    edges = []
    visited = set()
    includes = includes or []

    for edge in includes:
      if edge == ego_edge:
        continue
      edges.append(self.edge_to_internal[edge])

    queue = [ego_v]
    while queue and len(edges) < receptive_field_size:
      current_node = queue.pop(0)
      if current_node in visited:
        continue
      visited.add(current_node)
      for u, v in self.nx_graph.out_edges(current_node):
        if len(edges) >= receptive_field_size:
          break
        if v in visited:
          continue
        queue.append(v)
        if (u, v) not in edges:
          edges.append((u, v))

    edge_pairs = []
    for u, v in edges:
      edge = self.edge_from_internal[(u, v)]
      if ego_v != u:
        if nx.has_path(self.nx_graph, ego_v, u):
          path_info = self.get_shortest_path_len(
              ego_edge,
              edge,
              {"linear": "estimate", "pois": ()},
              return_path=True,
          )
          assert isinstance(path_info, tuple)
          _, path = path_info
          cand_edge = path[1]
        else:
          cand_edge = None
      else:
        cand_edge = None
      edge_pairs.append((
          edge,
          cand_edge,
      ))
    return tuple(edge_pairs)

  def sample_noncentral_edge(
      self, split, rng
  ):
    if split == 0:
      internal_edge = rng.choice(self.pairs_train)
    elif split == 1:
      internal_edge = rng.choice(self.pairs_val)
    elif split == 2:
      internal_edge = rng.choice(self.pairs_test)
    else:
      raise ValueError("no dataset split specified.")
    return self.edge_from_internal[internal_edge]

  @property
  def central_edges(self):
    edges = []
    for edge in self.nx_graph.out_edges(self.central_node):
      edges.append(self.edge_from_internal[edge])
    return tuple(edges)

  def _divide_dataset(self):
    """Divide dataset of nav problems into train, val, and test splits."""

    edges = []
    for edge in self.nx_graph.edges():
      if edge not in self.nx_graph.out_edges(self.central_node):
        edges.append(edge)

    if self.splits[0] == 1:
      self.pairs_train = edges
      self.pairs_val = []
      self.pairs_test = []

    num_train = int(self.splits[0] * len(edges))
    num_val = int(self.splits[1] * len(edges))

    self.rng.shuffle(edges)
    self.pairs_train = []
    for pair in edges[:num_train]:
      self.pairs_train.append(pair)
    self.pairs_val = []
    for pair in edges[num_train : num_train + num_val]:
      self.pairs_val.append(pair)
    self.pairs_test = []
    for pair in edges[num_train + num_val :]:
      self.pairs_test.append(pair)

  def get_node_shortest_path_len(
      self,
      linear_pref,
      start,
      end,
  ):
    if (linear_pref, start, end) in self.shortest_path_lens:
      return self.shortest_path_lens[(linear_pref, start, end)]
    if linear_pref == "hop":
      cost_fn = "hop"
    else:
      cost_fn = utils.get_modified_cost_fn(linear_pref)
    dist = nx.shortest_path_length(self.nx_graph, start, end, weight=cost_fn)
    self.shortest_path_lens[(linear_pref, start, end)] = dist
    return dist

  def get_shortest_path_len(
      self,
      start,
      end,
      query_data = None,
      return_path = False,
  ):
    if query_data is None:
      query_data = {"linear": "", "pois": ()}

    # Convert to internal types
    start = self.edge_to_internal[start]
    end = self.edge_to_internal[end]
    start_u, start_v = start
    end_u, end_v = end

    # Remove parts of query already satisfied by start and end edges
    query_pois = query_data["pois"]
    new_query_pois = []
    for poi_pool in query_pois:
      sat = False
      for poi in poi_pool:
        if (
            start in self.poi_type_id_to_edge[poi]
            or end in self.poi_type_id_to_edge[poi]
        ):
          sat = True
          break
      if not sat:
        new_query_pois.append(poi_pool)
    query_pois = tuple(new_query_pois)

    # Check cache for results
    if (
        query_data["linear"],
        query_data["pois"],
        start_v,
        end_u,
    ) in self.query_shortest_paths:
      shortest_path_len = self.query_shortest_path_lens[(
          query_data["linear"],
          query_data["pois"],
          start_v,
          end_u,
      )]
      if shortest_path_len is None:
        return None
      if return_path:
        shortest_path = self.query_shortest_paths[(
            query_data["linear"],
            query_data["pois"],
            start_v,
            end_u,
        )]
        return shortest_path_len, shortest_path
      else:
        return shortest_path_len

    cost_fn = utils.get_modified_cost_fn(query_data["linear"])
    graph = self.nx_graph

    if not query_pois:  # Easy shortest-path if no POI specified
      try:
        path = list(nx.shortest_path(graph, start_v, end_u, weight=cost_fn))
      except nx.NetworkXNoPath:
        return None
    else:
      sat_edges = []  # Satisfactory edges for each query req.
      for pool in query_pois:
        sat = set(sum([self.poi_type_id_to_edge[p] for p in pool], []))
        if not sat:
          return None
        sat_edges.append(sat)
      edge_combos = {
          tuple(sorted(set(edges))) for edges in itertools.product(*sat_edges)
      }
      if len(edge_combos) > TOO_MANY_EDGES:
        print("Too many routes to consider: {}".format(len(edge_combos)))
        raise TimeoutError(
            "Too many routes to consider: {}".format(len(edge_combos))
        )
      best_sat_edge_order = ()
      for sat_edge_combo in edge_combos:
        for sat_edge_order in itertools.permutations(sat_edge_combo):
          dist = 0
          current_node = start_v
          try:
            for u, v in sat_edge_order:
              dist += self.get_node_shortest_path_len(
                  query_data["linear"],
                  current_node,
                  u,
              )
              dist += cost_fn(self.nx_graph.get_edge_data(u, v))
              current_node = v
            dist += self.get_node_shortest_path_len(
                query_data["linear"],
                current_node,
                end_u,
            )
            if not best_sat_edge_order or best_sat_edge_order[1] > dist:
              best_sat_edge_order = (sat_edge_order, dist)
          except nx.NetworkXNoPath:
            continue
      if not best_sat_edge_order:
        self.query_shortest_path_lens[
            (query_data["linear"], query_pois, start_v, end_u)
        ] = None
        return

      path = []
      current_node = start_v
      for u, v in best_sat_edge_order[0] + (end,):
        path += list(nx.shortest_path(graph, current_node, u, weight=cost_fn))
        current_node = v

    dist = 0.0
    edges = []
    path = [start_u] + path + [end_v]
    for i in range(len(path) - 1):
      edge = (path[i], path[i + 1])
      edges.append(self.edge_from_internal[edge])
      dist += self.nx_graph.get_edge_data(*edge)["current_travel_time"]
    edges = tuple(edges)
    self.query_shortest_path_lens[(
        query_data["linear"],
        query_data["pois"],
        start_v,
        end_u,
    )] = dist
    self.query_shortest_paths[(
        query_data["linear"],
        query_data["pois"],
        start_v,
        end_u,
    )] = edges

    if return_path:
      return dist, edges
    else:
      return dist

  def route_metrics(
      self,
      query_data,
      end,
      edgelist,
  ):
    internal_edgelist = [self.edge_to_internal[edge] for edge in edgelist]

    # Get our travel time
    our_time = 0
    for internal_edge in internal_edgelist:
      our_time += self.nx_graph.get_edge_data(*internal_edge)[
          "current_travel_time"
      ]

    # Check end vertex
    internal_end = self.edge_to_internal[end]
    reaches_destination = int(internal_edgelist[-1] == internal_end)

    # Check POIs visited
    meets_poi = {poi: False for poi in self.poi_type_id_to_edge.keys()}
    for internal_edge in internal_edgelist + [internal_end]:
      for _, poi_types in self.get_edge_pois(
          self.edge_from_internal[internal_edge]
      ):
        for poi_type in poi_types:
          meets_poi[poi_type] = True
    achieved = 0
    for poi_pool in query_data["pois"]:
      if any([meets_poi[poi] for poi in poi_pool]):
        achieved += 1 / len(query_data["pois"])
    if query_data["pois"]:
      frac_achieved = achieved / len(query_data["pois"])
    else:
      frac_achieved = 1

    our_penalty = 0
    for internal_edge in internal_edgelist:
      our_penalty += utils.get_modified_cost_fn(query_data["linear"])(
          self.nx_graph.get_edge_data(*internal_edge),
      )

    metrics = {
        "num_pois_achieved": achieved,
        "frac_pois_achieved": frac_achieved,
        "reaches_destination": reaches_destination,
        "travel_time": our_time,
        "penalty": our_penalty,
    }

    if "time_budget" in query_data:
      metrics["budget_met"] = (
          query_data["time_budget"] * BUDGET_SCALE * 60 >= our_time
      )
    else:
      best_info = self.get_shortest_path_len(
          edgelist[0], end, query_data, return_path=True
      )
      if best_info is None:
        raise ValueError("No satisfactory route exists.")
      _, best_path = best_info  # type: ignore
      best_edgelist = [self.edge_to_internal[edge] for edge in best_path]
      best_penalty = 0
      for internal_edge in best_edgelist:
        best_penalty += utils.get_modified_cost_fn(query_data["linear"])(
            None,
            None,
            self.nx_graph.get_edge_data(*internal_edge),
        )

      best_time = self.get_shortest_path_len(
          edgelist[0],
          end,
          {"pois": query_data["pois"], "linear": ""},
      )

      metrics["excess_travel_time"] = our_time - best_time
      metrics["excess_penalty"] = our_penalty - best_penalty

    return metrics
