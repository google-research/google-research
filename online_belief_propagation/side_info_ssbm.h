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

#ifndef ONLINE_BELIEF_PROPAGATION_SIDE_INFO_SSBM_H_
#define ONLINE_BELIEF_PROPAGATION_SIDE_INFO_SSBM_H_

#include <map>
#include <set>

#include "streaming_community_detection_algorithm.h"

// Streaming Stochastic Block Belief Propagation algorithm with side
// information. The details are provided in the paper (Algorithm 1).
class SideInfoSSBM : public StreamingCommunityDetectionAlgorithm<int, int> {
 public:
  SideInfoSSBM(double a, double b, double label_confidence, int num_clusters,
               int radius, int num_vertices);
  std::vector<int> GenerateClusters() const override;

 private:
  void Initialize() override;
  void UpdateLabels(int source, const GraphStream* input,
                    const int* side_information = NULL) override;

  // BFS function on the graph.
  // multi_neighbors[i] stores the neighbors of source in distance 'r-i'.
  void BFS(const GraphStream* input,
           std::vector<std::vector<double>>* multi_neighbors,
           std::map<int, int>* parents, int vertex, int radius, int parent);

  // The probability of edge between two vertices inside each cluster
  // and between two clusters times 'n'. Represents the variables
  // 'a' and 'b' from the paper.
  double a_;
  double b_;

  // The probability that the side information is correct. Represents
  // 1-alpha from the paper.
  double label_confidence_;

  // The number of clusters and vertices in the input.
  int num_clusters_, num_vertices_;

  // The radius of BFS in our algorithm.
  int radius_;

  // The side information.
  std::vector<int> side_infos_;

  // Represents the label distribution of each edge after applying the
  // above WeightFunction.
  std::map<std::pair<int, int>, std::vector<double>> edge_labels_;

  // The label of each vertex. It gets value at the end of last insertion.
  std::vector<int> vertex_labels_;
};

#endif  // ONLINE_BELIEF_PROPAGATION_SIDE_INFO_SSBM_H_
