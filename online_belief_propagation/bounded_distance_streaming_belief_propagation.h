// Copyright 2022 The Google Research Authors.
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

#ifndef ONLINE_BELIEF_PROPAGATION_BOUNDED_DISTANCE_STREAMING_BELIEF_PROPAGATION_H_
#define ONLINE_BELIEF_PROPAGATION_BOUNDED_DISTANCE_STREAMING_BELIEF_PROPAGATION_H_

#include <map>
#include <set>

#include "streaming_community_detection_algorithm.h"

class EdgeLevel {
 public:
  EdgeLevel(int x_, int y_, int level_) {
    x = x_;
    y = y_;
    level = level_;
  }

  bool operator<(const EdgeLevel& edge) const {
    if (x != edge.x) return x < edge.x;
    if (y != edge.y) return y < edge.y;
    return level < edge.level;
  }

 private:
  // Endpoints of the edge.
  int x, y;
  // Level of the edge, in the range [0, R].
  int level;
};

// This class implements Bounded-Distance Streaming Belief Propagation
// (Agorithm 2 in the paper).
class BoundedDistanceStreamingBeliefPropagation
    : public StreamingCommunityDetectionAlgorithm<int, int> {
 public:
  BoundedDistanceStreamingBeliefPropagation(double a, double b,
                                            double label_confidence,
                                            int num_clusters, int radius,
                                            int num_vertices);
  std::vector<int> GenerateClusters() const override;

 private:
  void Initialize() override;
  void UpdateLabels(int source, const GraphStream* input,
                    const int* side_information = NULL) override;

  // BFS function on the graph.
  // multi_neighbors[i] stores the nodes at distance 'r-i' from 'source'.
  void BFS(const GraphStream* input,
           std::vector<std::vector<double>>* multi_neighbors,
           std::map<int, int>* parents, int vertex, int radius, int parent);

  // The probability of edge between two vertices inside each cluster
  // and between two clusters times 'num_vertices_'.
  double a_;
  double b_;

  // The probability that the side information is correct. Represents
  // 1-alpha from the paper.
  double label_confidence_;

  // The number of clusters and vertices in the input.
  int num_communities_, num_vertices_;

  // The radius of BFS in our algorithm.
  int radius_;

  // The side information.
  std::vector<int> side_infos_;

  // Represents the label distribution of each edge after applying the
  // 'WeightFunction'.
  std::map<EdgeLevel, std::vector<double>> edge_labels_;

  // The label of each vertex. This vector is populated at the end of last
  // insertion.
  std::vector<int> vertex_labels_;
};

#endif  // ONLINE_BELIEF_PROPAGATION_BOUNDED_DISTANCE_STREAMING_BELIEF_PROPAGATION_H_
