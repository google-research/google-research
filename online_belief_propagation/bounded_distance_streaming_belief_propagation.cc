// Copyright 2021 The Google Research Authors.
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

#include "bounded_distance_streaming_belief_propagation.h"

#include <assert.h>

#include <cmath>
#include <iostream>
#include <random>

#include "belief_propagation_utils.h"

BoundedDistanceStreamingBeliefPropagation::
    BoundedDistanceStreamingBeliefPropagation(double a, double b,
                                              double label_confidence,
                                              int num_clusters, int radius,
                                              int num_vertices)
    : a_(a),
      b_(b),
      label_confidence_(label_confidence),
      num_communities_(num_clusters),
      num_vertices_(num_vertices),
      radius_(radius) {
  assert(a_ >= 0);
  assert(b_ >= 0);
  assert(label_confidence_ > 0.0);
  assert(label_confidence_ <= 1.0);
  assert(num_communities_ > 0);
  assert(radius_ > 0);
  assert(num_vertices_ > 0);
  assert(a <= num_vertices_);
  assert(b <= num_vertices_);
}

std::vector<int> BoundedDistanceStreamingBeliefPropagation::GenerateClusters()
    const {
  return vertex_labels_;
}

void BoundedDistanceStreamingBeliefPropagation::Initialize() {
  side_infos_.clear();
  edge_labels_.clear();
  vertex_labels_.clear();
}

void BoundedDistanceStreamingBeliefPropagation::UpdateLabels(
    int source, const GraphStream* input, const int* side_information) {
  // Update side information
  side_infos_.push_back(*side_information);
  const auto& neighbors = input->GetCurrentNeighborhood(source);
  // First step: compute the label distribution of all the new edges.
  // Mapping of the names here to the names in the write up:
  // source -> v(t)
  // nei -> v
  // lvl2 -> v'
  for (auto& nei : neighbors) {
    // m = 0 case.
    std::vector<double> current_edge_label_0(num_communities_, 0);
    for (int j = 0; j < num_communities_; j++) {
      if (j == side_infos_[nei]) {
        current_edge_label_0[j] += log(label_confidence_);
      } else {
        current_edge_label_0[j] +=
            log((1.0 - label_confidence_) / (num_communities_ - 1.0));
      }
    }
    EdgeLevel edge_level_0(nei, source, 0);
    edge_labels_[edge_level_0] =
        WeightFunction(current_edge_label_0, num_communities_, a_, b_);

    for (int m = 1; m <= radius_; m++) {
      const auto& lvl2 = input->GetCurrentNeighborhood(nei);
      std::vector<double> current_edge_label(num_communities_, 0);
      for (int j = 0; j < num_communities_; j++) {
        for (int i = 0; i < lvl2.size(); i++) {
          if (lvl2[i] != source) {
            EdgeLevel edge_level_1(lvl2[i], nei, m - 1);
            // Notice that the WeightFunction is applyed to edge_label_ already.
            current_edge_label[j] += edge_labels_[edge_level_1][j];
          }
        }
        if (j == side_infos_[nei]) {
          current_edge_label[j] += log(label_confidence_);
        } else {
          current_edge_label[j] +=
              log((1.0 - label_confidence_) / (num_communities_ - 1.0));
        }
      }
      double g_value = 0.0;
      for (int i = 0; i < num_communities_; i++) {
        g_value += exp(current_edge_label[i]);
      }
      double log_g_value = log(g_value);
      for (int j = 0; j < num_communities_; j++) {
        current_edge_label[j] -= log_g_value;
      }
      EdgeLevel edge_level(nei, source, m);
      edge_labels_[edge_level] =
          WeightFunction(current_edge_label, num_communities_, a_, b_);
    }
  }
  // Second step: Compute the tree structure of height 'r' around source.
  // Multi R, R = 0
  // Mapping of the names here to the names in the write up:
  // source -> v(t)
  // nei -> v
  // lvl2 -> v'
  for (auto& nei : neighbors) {
    // m = 0 case.
    std::vector<double> current_edge_label_0(num_communities_, 0);
    for (int j = 0; j < num_communities_; j++) {
      if (j == side_infos_[source]) {
        current_edge_label_0[j] += log(label_confidence_);
      } else {
        current_edge_label_0[j] +=
            log((1.0 - label_confidence_) / (num_communities_ - 1.0));
      }
    }
    EdgeLevel edge_level_0(source, nei, 0);
    edge_labels_[edge_level_0] =
        WeightFunction(current_edge_label_0, num_communities_, a_, b_);
    for (int m = 1; m <= radius_; m++) {
      const auto& lvl2 = input->GetCurrentNeighborhood(source);
      std::vector<double> current_edge_label(num_communities_, 0);
      for (int j = 0; j < num_communities_; j++) {
        for (int i = 0; i < lvl2.size(); i++) {
          if (lvl2[i] != nei) {
            EdgeLevel edge_level_1(lvl2[i], source, m - 1);
            // Notice that the WeightFunction is applyed to edge_label_ already.
            current_edge_label[j] += edge_labels_[edge_level_1][j];
          }
        }
        if (j == side_infos_[source]) {
          current_edge_label[j] += log(label_confidence_);
        } else {
          current_edge_label[j] +=
              log((1.0 - label_confidence_) / (num_communities_ - 1.0));
        }
      }
      double g_value = 0;
      for (int i = 0; i < num_communities_; i++) {
        g_value += exp(current_edge_label[i]);
      }
      double log_g_value = log(g_value);
      for (int j = 0; j < num_communities_; j++) {
        current_edge_label[j] -= log_g_value;
      }
      EdgeLevel edge_level(source, nei, m);
      edge_labels_[edge_level] =
          WeightFunction(current_edge_label, num_communities_, a_, b_);
    }
  }

  // multi_neighbors[i] contains the vertices at distance r-i from 'source'.
  std::vector<std::vector<double>> multi_neighbors(radius_ + 1);

  // The parent to a vertex on the path to source.
  std::map<int, int> parents;

  BFS(input, &multi_neighbors, &parents, source, radius_, -1);
  // Edge weight for a radius of 'r'.
  for (int r = radius_ - 1; r > 0; r--) {
    // Mapping of the names here to the names in the write up:
    // source -> v(t)
    // r_nei -> v
    // nei -> v'
    // lvl2 -> v''
    for (int m = r; m <= radius_; m++) {
      for (const auto& r_nei : multi_neighbors[r]) {
        int nei = parents[r_nei];
        const auto& lvl2 = input->GetCurrentNeighborhood(nei);
        std::vector<double> current_edge_label(num_communities_, 0);
        for (int j = 0; j < num_communities_; j++) {
          for (int i = 0; i < lvl2.size(); i++) {
            if (lvl2[i] != r_nei) {
              // Notice that the WeightFunction is applied to edge_label_
              // already.
              EdgeLevel edge_level(lvl2[i], nei, m - 1);
              current_edge_label[j] += edge_labels_[edge_level][j];
            }
          }
          if (j == side_infos_[nei]) {
            current_edge_label[j] += log(label_confidence_);
          } else {
            current_edge_label[j] +=
                log((1.0 - label_confidence_) / (num_communities_ - 1.0));
          }
        }
        double g_value = 0.0;
        for (int i = 0; i < num_communities_; i++) {
          g_value += exp(current_edge_label[i]);
        }
        double log_g_value = log(g_value);
        for (int j = 0; j < num_communities_; j++) {
          current_edge_label[j] -= log_g_value;
        }
        EdgeLevel edge_level(nei, r_nei, m);
        edge_labels_[edge_level] =
            WeightFunction(current_edge_label, num_communities_, a_, b_);
      }
    }
  }
  // We compute the weights of the node after all the nodes are inserted.
  if (source == num_vertices_ - 1) {
    // Mapping of the names here to the names in the write up:
    // i -> u
    // nei -> u'
    for (int i = 0; i < num_vertices_; i++) {
      std::vector<double> label(num_communities_, 0);
      const auto& neighbors = input->GetCurrentNeighborhood(i);
      for (int j = 0; j < num_communities_; j++) {
        // Notice that the WeightFunction is applied to edge_label_ already.
        for (const auto& nei : neighbors) {
          EdgeLevel edge_level(nei, i, radius_ - 1);
          label[j] += edge_labels_[edge_level][j];
        }
        if (j == side_infos_[i]) {
          label[j] += log(label_confidence_);
        } else {
          label[j] += log((1.0 - label_confidence_) / (num_communities_ - 1.0));
        }
      }
      // We do not need to compute G(q_u) as it is a double value which all
      // the coordinates are shifted by. Therefore, it does not
      // affect the maximum value.
      int max_index = 0;
      for (int j = 0; j < num_communities_; j++) {
        if (label[j] > label[max_index]) {
          max_index = j;
        }
      }
      vertex_labels_.push_back(max_index);
    }
  }
}

void BoundedDistanceStreamingBeliefPropagation::BFS(
    const GraphStream* input, std::vector<std::vector<double>>* multi_neighbors,
    std::map<int, int>* parents, int vertex, int radius, int parent) {
  std::set<int> seen;
  seen.insert(vertex);
  // first is the name, second is the radius
  std::vector<std::pair<int, int>> queue;
  queue.push_back(std::pair<int, int>(vertex, radius));
  int pointer = 0;
  while (pointer < queue.size()) {
    if (queue[pointer].second == 0) {
      break;
    }
    const auto& neighbors = input->GetCurrentNeighborhood(queue[pointer].first);
    for (const auto& nei : neighbors) {
      if (seen.find(nei) == seen.end()) {
        seen.insert(nei);
        (*parents)[nei] = queue[pointer].first;
        (*multi_neighbors)[queue[pointer].second - 1].push_back(nei);
        queue.push_back(std::pair<int, int>(nei, queue[pointer].second - 1));
      }
    }
    pointer++;
  }
}
