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

#include "side_info_ssbm.h"

#include <assert.h>

#include <cmath>
#include <iostream>

#include "belief_propagation_utils.h"

SideInfoSSBM::SideInfoSSBM(double a, double b, double label_confidence,
                           int num_clusters, int radius, int num_vertices)
    : a_(a),
      b_(b),
      label_confidence_(label_confidence),
      num_clusters_(num_clusters),
      num_vertices_(num_vertices),
      radius_(radius) {
  assert(a_ > 0);
  assert(b_ >= 0);
  assert(label_confidence_ >= 0.0);
  assert(label_confidence_ <= 1.0);
  assert(num_clusters_ > 0);
  assert(radius_ > 0);
  assert(num_vertices_ > 0);
  assert(a_ <= num_vertices_);
  assert(b_ <= num_vertices_);
}

std::vector<int> SideInfoSSBM::GenerateClusters() const {
  return vertex_labels_;
}

void SideInfoSSBM::Initialize() {
  side_infos_.clear();
  edge_labels_.clear();
  vertex_labels_.clear();
}

void SideInfoSSBM::UpdateLabels(int source, const GraphStream* input,
                                const int* side_information) {
  // Update side information
  side_infos_.push_back(*side_information);
  const auto& neighbors = input->GetCurrentNeighborhood(source);
  // First step: compute the label distribution of all the new edges..
  // Mapping of the names here to the names in the paper:
  // source -> v(t)
  // nei -> v
  // lvl2 -> v'
  for (auto& nei : neighbors) {
    const auto& lvl2 = input->GetCurrentNeighborhood(nei);
    std::vector<double> current_edge_label(num_clusters_, 0);
    for (int j = 0; j < num_clusters_; j++) {
      for (int i = 0; i < lvl2.size(); i++) {
        if (lvl2[i] != source) {
          // Notice that the WeightFunction is applyed to edge_label_ already.
          current_edge_label[j] +=
              edge_labels_[std::pair<int, int>(lvl2[i], nei)][j];
        }
      }
      if (j == side_infos_[nei]) {
        current_edge_label[j] += log(label_confidence_);
      } else {
        current_edge_label[j] +=
            log((1 - label_confidence_) / (num_clusters_ - 1));
      }
    }
    double g_value = 0;
    for (int i = 0; i < num_clusters_; i++) {
      g_value += exp(current_edge_label[i]);
    }
    double log_g_value = log(g_value);
    for (int j = 0; j < num_clusters_; j++) {
      current_edge_label[j] -= log_g_value;
    }
    edge_labels_[std::pair<int, int>(nei, source)] =
        WeightFunction(current_edge_label, num_clusters_, a_, b_);
  }

  // Second step: Compute the tree structure of height 'r' around source.

  // multi_neighbors[i] contains the neighbors of source in distance 'r-i'.
  std::vector<std::vector<double>> multi_neighbors(radius_ + 1);

  // The parent to a vertex on the path to source.
  std::map<int, int> parents;

  BFS(input, &multi_neighbors, &parents, source, radius_, -1);
  // Edge weight for a radius of 'r'.
  for (int r = radius_ - 1; r >= 0; r--) {
    // Mapping of the names here to the names in the paper:
    // source -> v(t)
    // r_nei -> v
    // nei -> v'
    // lvl2 -> v''
    for (const auto& r_nei : multi_neighbors[r]) {
      int nei = parents[r_nei];
      const auto& lvl2 = input->GetCurrentNeighborhood(nei);
      std::vector<double> current_edge_label(num_clusters_, 0);
      for (int j = 0; j < num_clusters_; j++) {
        for (int i = 0; i < lvl2.size(); i++) {
          if (lvl2[i] != r_nei) {
            // Notice that the WeightFunction is applied to edge_label_ already.
            current_edge_label[j] +=
                edge_labels_[std::pair<int, int>(lvl2[i], nei)][j];
          }
        }
        if (j == side_infos_[nei]) {
          current_edge_label[j] += log(label_confidence_);
        } else {
          current_edge_label[j] +=
              log((1 - label_confidence_) / (num_clusters_ - 1));
        }
      }
      double g_value = 0;
      for (int i = 0; i < num_clusters_; i++) {
        g_value += exp(current_edge_label[i]);
      }
      double log_g_value = log(g_value);
      for (int j = 0; j < num_clusters_; j++) {
        current_edge_label[j] -= log_g_value;
      }
      edge_labels_[std::pair<int, int>(nei, r_nei)] =
          WeightFunction(current_edge_label, num_clusters_, a_, b_);
    }
  }
  // We compute the weights of the node after all the nodes are inserted.
  if (source == num_vertices_ - 1) {
    // Mapping of the names here to the names in the paper:
    // i -> u
    // nei -> v
    for (int i = 0; i < num_vertices_; i++) {
      std::vector<double> label(num_clusters_, 0);
      const auto& neighbors = input->GetCurrentNeighborhood(i);
      for (int j = 0; j < num_clusters_; j++) {
        // Notice that the WeightFunction is applied to edge_label_ already.
        for (const auto& nei : neighbors) {
          if (edge_labels_.find(std::pair<int, int>(nei, i)) !=
              edge_labels_.end())
            label[j] += edge_labels_[std::pair<int, int>(nei, i)][j];
        }
        if (j == side_infos_[i]) {
          label[j] += log(label_confidence_);
        } else {
          label[j] += log((1 - label_confidence_) / (num_clusters_ - 1));
        }
      }
      int max_index = 0;
      for (int j = 0; j < num_clusters_; j++) {
        if (label[j] > label[max_index]) {
          max_index = j;
        }
      }
      vertex_labels_.push_back(max_index);
    }
  }
}

void SideInfoSSBM::BFS(const GraphStream* input,
                       std::vector<std::vector<double>>* multi_neighbors,
                       std::map<int, int>* parents, int vertex, int radius,
                       int parent) {
  std::set<int> seen;
  seen.insert(vertex);
  // the pari represents (vertex id, radius)
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
