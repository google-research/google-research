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

"""Naive implementation of the contractive hierarchies algorithm."""

import heapq

from semantic_routing.benchmark import utils

# Contraction parameters.
INHERIT_FIELDS = [
    "lanes",
    "highway",
]
COMBINE_FIELDS = [
    "length",
    "poi_node_names",
    "poi_type_ids",
    "poi_node_ids",
    "edges",
    "current_travel_times",
    "highways",
]
CONTRACTION_REPORT_LEVELS = list(range(8, 40))
POI_COUNT_MAX = 10
LOG_PROGRESS_FREQ = 50000
ROAD_LEVEL_DEFAULT = 10
CONTRACTED_NEIGHBORS_PRIORITY_WEIGHT = 2
ROAD_LEVEL_PRIORITY_WEIGHT = 4


def calculate_priority(orig_graph, cont_graph, node):
  """Compute the priority of a node.

  Args:
    orig_graph: Graph without shortcuts.
    cont_graph: Graph with shortcuts and some nodes contracted out.
    node: Node to compute the priority of. Lower priority means contract first.

  Returns:
    Number of contracted neighbors, the edge difference, and the highway status.
  """
  orig_in_neighbors = set(orig_graph.predecessors(node))
  orig_out_neighbors = set(orig_graph.successors(node))
  orig_neighbors = orig_in_neighbors.union(orig_out_neighbors)

  road_level = []
  for _, _, data in orig_graph.out_edges(node, data=True):
    if data["highway"] not in utils.ROAD_VALUES:
      road_level.append(ROAD_LEVEL_DEFAULT)
      continue
    road_level.append(utils.ROAD_VALUES.index(data["highway"]))
  road_level = sum(road_level) / len(road_level) if road_level else 0

  in_neighbors = set(cont_graph.predecessors(node))
  out_neighbors = set(cont_graph.successors(node))
  neighbors = in_neighbors.union(out_neighbors)

  contracted_neighbors = len(orig_neighbors - neighbors)

  shortcuts = 0
  for u in in_neighbors:
    for v in out_neighbors:
      if u != v and not cont_graph.has_edge(u, v):
        shortcuts += 1

  edge_difference = shortcuts - len(neighbors)
  return (
      CONTRACTED_NEIGHBORS_PRIORITY_WEIGHT * contracted_neighbors,
      edge_difference,
      ROAD_LEVEL_PRIORITY_WEIGHT * road_level,
  )


def contract_node(record_graph, cont_graph, node):
  """Contract a node."""

  for pred in cont_graph.predecessors(node):
    for succ in cont_graph.successors(node):
      if pred == succ:
        continue

      # Compute travel time, level and penultimate node of the shortcut
      new_current_travel_time = (
          cont_graph[pred][node]["current_travel_time"]
          + cont_graph[node][succ]["current_travel_time"]
      )
      new_travel_time = (
          cont_graph[pred][node]["travel_time"]
          + cont_graph[node][succ]["travel_time"]
      )
      new_level = (
          cont_graph[pred][node]["level"] + cont_graph[node][succ]["level"]
      )
      combined_data = {
          k: cont_graph[pred][node][k] + cont_graph[node][succ][k]
          for k in COMBINE_FIELDS
      }

      # Terminate contraction if level is too high
      if new_level > max(CONTRACTION_REPORT_LEVELS):
        continue

      # Terminate contraction if too many POIs accumulated
      if len(combined_data["poi_node_names"]) > POI_COUNT_MAX:
        continue

      # Only replace edge with shortcut if shortcut is faster
      if cont_graph.has_edge(pred, succ):
        if cont_graph[pred][succ]["travel_time"] <= new_travel_time:
          continue
        cont_graph.remove_edge(pred, succ)

      # Add shortcut
      edge_data = {k: cont_graph[node][succ][k] for k in INHERIT_FIELDS}
      edge_data.update(combined_data)
      edge_data["osmid"] = -1
      edge_data["current_travel_time"] = new_current_travel_time
      edge_data["travel_time"] = new_travel_time
      edge_data["level"] = new_level
      edge_data["u_lat"] = cont_graph[pred][node]["u_lat"]
      edge_data["u_lon"] = cont_graph[pred][node]["u_lon"]
      edge_data["v_lat"] = cont_graph[node][succ]["v_lat"]
      edge_data["v_lon"] = cont_graph[node][succ]["v_lon"]
      cont_graph.add_edge(pred, succ, **edge_data)

      if new_level in CONTRACTION_REPORT_LEVELS:
        if (
            record_graph.has_edge(pred, succ)
            and record_graph[pred][succ]["level"] > 1
            and cont_graph[pred][succ]["level"] >= 2
        ):
          record_graph.remove_edge(pred, succ)
        record_graph.add_edge(pred, succ, **edge_data)


def contraction_hierarchies_preprocessing(record_graph):
  """Add contraction hierarchy shortcuts to a graph."""

  orig_graph = record_graph.copy()
  cont_graph = record_graph.copy()

  # Initial priority calculation
  priority = {
      node: calculate_priority(orig_graph, cont_graph, node)
      for node in record_graph.nodes()
  }
  priority_queue = []
  for node in cont_graph.nodes():
    heapq.heappush(priority_queue, (sum(priority[node]), node))

  # Main contraction loop
  contracted = []
  j = len(cont_graph.nodes())
  while cont_graph.nodes:
    orig_priority, node = heapq.heappop(priority_queue)
    if node not in cont_graph:
      continue
    if orig_priority != sum(priority[node]):
      continue

    priority.pop(node)
    contracted.append(node)

    if len(contracted) % LOG_PROGRESS_FREQ == 0:
      print("Contraction progress:", len(contracted), "/", j)
    contract_node(record_graph, cont_graph, node)

    neighs = set(cont_graph.predecessors(node)).union(
        cont_graph.successors(node)
    )
    cont_graph.remove_node(node)

    # Update priorities of neighbors
    for neighbor in neighs:
      if neighbor == node:
        continue
      new_p = calculate_priority(orig_graph, cont_graph, neighbor)
      if new_p != priority[neighbor]:
        priority[neighbor] = new_p
        heapq.heappush(priority_queue, (sum(priority[neighbor]), neighbor))
