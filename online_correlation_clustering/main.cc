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

#include <time.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "agreement_algo.h"
#include "graph_handler.h"
#include "pivot_algo.h"
#include "utils.h"

double beta = 0.2, lambda = 0.2;
bool timestamped_input = true;

int main(int argc, char* argv[]) {
  GraphHandler graph_handler;
  if (timestamped_input) {
    graph_handler.ReadGraphAndOrderByNodeId();
  } else {
    graph_handler.ReadGraph();
  }
  graph_handler.AddMissingSelfLoops();
  graph_handler.StartMaintainingOnlineGraphInstance(
      /*shuffle_order=*/!timestamped_input);

  std::cout << "Singletons" << std::endl;
  graph_handler.RemoveAllOnlineNodes();
  while (graph_handler.NextOnlineNodeExists()) {
    (void)graph_handler.AddNextOnlineNode();
    std::cout << graph_handler.online_num_edges_ -
                     graph_handler.online_neighbors_.size()
              << "\t" << 0 << "\t" << 0 << std::endl;
  }

  std::vector<int> old_clustering;
  clock_t t;

  std::cout << "AgreementAlgo" << std::endl;
  int64_t total_recourse = 0;
  double total_time = 0;
  old_clustering.clear();
  graph_handler.RemoveAllOnlineNodes();
  AgreementCorrelationClustering agreement_correlation_clustering(beta, lambda);
  RecourseCalculator recourse_calculator;
  while (graph_handler.NextOnlineNodeExists()) {
    const std::vector<int>& neighbors_of_new_node =
        graph_handler.AddNextOnlineNode();
    t = clock();
    std::vector<int> clustering = agreement_correlation_clustering.Cluster(
        graph_handler.online_neighbors_.size() - 1, neighbors_of_new_node);
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;

    int64_t score = ComputeCost::ComputeClusteringCost(
        graph_handler.online_neighbors_, clustering);

    total_recourse += recourse_calculator.RecourseCostUsingMaxOverlap(
        old_clustering, clustering);

    old_clustering = clustering;
    std::cout << score << "\t" << total_recourse << "\t" << total_time
              << std::endl;
  }

  std::cout << "PivotAlgo" << std::endl;
  PivotAlgorithm pivot_algorithm({});
  int64_t total_recourse_pivot = 0;
  double total_time_pivot = 0;
  old_clustering.clear();

  RecourseCalculator recourse_calculator_pivot;
  for (int i = 0; i < graph_handler.online_neighbors_.size(); i++) {
    std::vector<int> online_neighborhood;
    for (int neighbor_id : graph_handler.online_neighbors_[i]) {
      if (neighbor_id <= i) {
        online_neighborhood.push_back(neighbor_id);
      }
    }
    t = clock();
    std::vector<int> clustering =
        pivot_algorithm.InsertNodeToClustering(online_neighborhood);

    total_time_pivot += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    int score = ComputeCost::ComputeClusteringCost(pivot_algorithm.neighbors_,
                                                   clustering);
    // declare the new node as a singleton cluster so that the two clusterings
    // have the same number of entries.
    total_recourse_pivot +=
        recourse_calculator_pivot.RecourseCostUsingMaxOverlap(old_clustering,
                                                              clustering);

    old_clustering = clustering;
    std::cout << score << "\t" << total_recourse_pivot << "\t"
              << total_time_pivot << std::endl;
  }

  std::cout << "OnlineAgreementAlgo" << std::endl;
  int64_t total_overlap_recourse_online_agreement = 0;
  double total_time_online_agreement = 0;
  std::vector<int> old_maintained_clustering;

  AgreementReconcileClustering reconsile_clustering;
  RecourseCalculator exact_recourse_calculator_online_agreement;
  RecourseCalculator overlap_recourse_calculator_online_agreement;
  agreement_correlation_clustering.RestartUpdateSequence();
  old_clustering.clear();
  graph_handler.RemoveAllOnlineNodes();
  while (graph_handler.NextOnlineNodeExists()) {
    const std::vector<int>& neighbors_of_new_node =
        graph_handler.AddNextOnlineNode();
    t = clock();
    std::vector<int> clustering = agreement_correlation_clustering.Cluster(
        graph_handler.online_neighbors_.size() - 1, neighbors_of_new_node);
    reconsile_clustering.AgreeementClusteringTransformCost(old_clustering,
                                                           clustering);

    total_time_online_agreement +=
        static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    int score = ComputeCost::ComputeClusteringCost(
        graph_handler.online_neighbors_,
        reconsile_clustering.maintained_clustering_);

    total_overlap_recourse_online_agreement +=
        overlap_recourse_calculator_online_agreement
            .RecourseCostUsingMaxOverlap(
                old_maintained_clustering,
                reconsile_clustering.maintained_clustering_);

    old_clustering = clustering;
    old_maintained_clustering = reconsile_clustering.maintained_clustering_;
    std::cout << score << "\t" << total_overlap_recourse_online_agreement
              << "\t" << total_time_online_agreement << std::endl;
  }

  return 0;
}
