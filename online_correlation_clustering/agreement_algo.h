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

#ifndef ONLINE_CORRELATION_CLUSTERING_AGREEMENT_ALGO_H_
#define ONLINE_CORRELATION_CLUSTERING_AGREEMENT_ALGO_H_

#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

class AgreementCorrelationClustering {
 public:
  AgreementCorrelationClustering(const double beta, const double lambda)
      : lambda_(lambda), beta_(beta) {}

  // Given a new node, with its neighbors, returns a clustering of the updated
  // instance where the new node has been added.
  std::vector<int> Cluster(const int new_node_id,
                           const std::vector<int> neighbors_of_new_node);

  // Cancels all updates, and starts maintaining a solution from scratch.
  void RestartUpdateSequence();

 private:
  // Given two sorted sets, computes the size of the intersection.
  int SortedVectorIntersection(const std::vector<int>& x,
                               const std::vector<int>& y);

  // Inserts a new node to the maintained graph.
  void AddNodeToMaintainedGraph(const int node_id,
                                const std::vector<int>& neighbors);

  // Updates the number of common neighbors of the endpoint of each edge.
  void UpdateCommonNeighborsOfEdgeEndpoints(const int new_node_id);

  // Computes the light nodes, and stores in `light_nodes_` whether each node is
  // heavy.
  void ComputeLightNodes();

  // Stores in `maintained_filtered_graph_` the subset of the edges of the edges
  // between nodes that are in agreement and at least on of the endpoints is
  // heavy.
  void RemoveLightEdges();

  // Computes the connected components of the `maintained_filtered_graph_`
  // graph.
  void ComputeConnectedComponents();

  // Variables defined in the paper.
  double lambda_, beta_;

  // The number of edges in the maintained graph.
  int num_edges_ = 0;
  // The number of nodes in the maintained graph.
  int num_nodes_ = 0;

  // The maintained graph.
  std::vector<std::vector<int>> neighbors_;

  // Stores the maintained graph of edges that are in agreement.
  std::vector<std::set<int>> maintained_agreement_neighbors_;

  // Stores the graph of the edges that are in agreement and at least one of
  // them is heavy.
  std::vector<std::set<int>> maintained_filtered_graph_;

  // Indicates whether each node is light.
  std::vector<bool> light_nodes_;

  // The id of the connected component that this node belongs to.
  std::vector<int> connected_components_;

  // Stores for each existing edge, the number of common neighbor of the two
  // endpoints.
  std::map<std::pair<int, int>, int> common_neighbors_of_pairs_;
};

#endif  // ONLINE_CORRELATION_CLUSTERING_AGREEMENT_ALGO_H_
