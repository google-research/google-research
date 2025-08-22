// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ONLINE_CORRELATION_CLUSTERING_GRAPH_HANDLER_H_
#define ONLINE_CORRELATION_CLUSTERING_GRAPH_HANDLER_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

class GraphHandler {
 public:
  // Reads the input. For each edge, the expected format is the endpoint of the
  // edge seperate with a white space. For instance, if we have three edges
  // (1, 5), (2, 3), (1, 3), an input can be:
  // 1 5
  // 2 3
  // 1 3
  // Ignores parallel edges and only keeps one copy. Notice that the name of the
  // endpoints can be strings or integers.
  void ReadGraph();

  // Same as ReadGraph, but the arrival sequence is consider to be the
  // increasing order of node ids.
  void ReadGraphAndOrderByNodeId();

  // If for a node there is no self-loop, this function adds it.
  void AddMissingSelfLoops();

  // Sets up a new arrival sequence.
  void StartMaintainingOnlineGraphInstance(bool shuffle_order);

  // Cleans up the online maintained graph.
  void RemoveAllOnlineNodes();

  // Adds to the maintained graph and returns the next node, w.r.t. the arrival
  // order that was decided.
  std::vector<int> AddNextOnlineNode();

  // Returns whether there exists another node in the online arrival sequence to
  // be added to the graph.
  bool NextOnlineNodeExists();

  // Represents the edges of the graph. The id of the nodes belongs to [0, n).
  // the i-th vector contains the neighbors of the node "i".
  std::vector<std::vector<int>> neighbors_;

  // Represents the edges of the graph. The id of the nodes belongs to [0, n).
  // the i-th vector contains the neighbors of the node "i".
  std::vector<std::vector<int>> online_neighbors_;

  int64_t online_num_edges_;

 private:
  int num_nodes_;
  // The name of a vertex to its id.
  std::map<std::string, int> id_map_;
  // Set of all the edges.
  std::set<std::pair<int, int>> edges_;

  std::vector<int> order_to_node_;
  std::vector<int> node_to_order_;
};

#endif  // ONLINE_CORRELATION_CLUSTERING_GRAPH_HANDLER_H_
