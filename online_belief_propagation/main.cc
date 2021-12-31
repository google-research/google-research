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

// The main file for running experiments.

// This file contains several simple experiments for testing the quality of
// belief propagation algorithms for community detection.
// The nodes of the graph have integer ground truth, we use a noisy version of
// this ground truth as side information. Due to this the
// StreamingCommunityDetectionAlgorithm being tested must have
// SideInfoType = int.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bounded_distance_streaming_belief_propagation.h"
#include "overlap_evaluate.h"
#include "sbm_graph.h"
#include "side_info_ssbm.h"

std::vector<int> GenerateSideInfoVector(
    const int k, const std::vector<int>& ground_truth_communities,
    const unsigned int seed, const double noise) {
  std::default_random_engine generator(seed);
  std::bernoulli_distribution ber(noise / (1 - 1.0 / k));
  std::uniform_int_distribution<int> unif(0, k - 1);

  // Generate the side information vector by copying 'ground_truth_communities'.
  // For ease of implementation, we set the side information to be completely
  // uniform (including the correct label) with slightly higher than 'noise'
  // probability. This produces the desired distribution.
  std::vector<int> side_info_vector;
  for (int i = 0; i < ground_truth_communities.size(); i++) {
    side_info_vector.push_back(ground_truth_communities[i]);
    if (ber(generator)) side_info_vector[i] = unif(generator);
  }
  return side_info_vector;
}

void RunExperiment(int n, int k, double a, double b, double radius,
                   double alpha, int seed) {
  // Constructs a graph.
  std::default_random_engine generator(seed);
  // One can use any graph instead of creating a syntethic one. In order to do
  // that one can implement a sub-class of Graph. Please refer to "graph.h" and
  // "sbm_graph.h" for more details.
  // In case any of the `n`, `k`, `a`, `b` variables are not known for the
  // graph, we provide code to estimate them (please see
  // estimate_stsbm_parameters.h).
  SBMGraph sbm_graph(n, k, a, b, generator());
  sbm_graph.Sort();
  GraphStream graph_stream(&sbm_graph);
  const std::vector<int> ground_truth_communities =
      sbm_graph.GetGroundTruthCommunities();
  const std::vector<int> side_info_vector =
      GenerateSideInfoVector(k, ground_truth_communities, seed, alpha);
  // Initiates and runs the algorithm. One can use
  // BoundedDistanceStreamingBeliefPropagation instead of SideInfoSSBM in the
  // following line to use bounded distance algorithm.
  SideInfoSSBM side_info_ssbm(a, b, 1 - alpha, k, radius, n);

  side_info_ssbm.Run(&graph_stream, &side_info_vector);
  std::vector<int> generated_clusters = side_info_ssbm.GenerateClusters();

  // Evaluates the solution.
  OverlapEvaluate evaluator;
  const double quality =
      evaluator(generated_clusters, ground_truth_communities);

  std::cout << "Fraction of correct labels: " << quality << std::endl;
  std::cout << "The computed labels: " << std::endl;
  for (const auto& label : generated_clusters) {
    std::cout << label << " ";
  }
  std::cout << std::endl;
}

int main() {
  // Set the following variables based on the experiment you are interested in.
  // The number of nodes of the graph.
  int n = 1000;

  // The number of communities.
  int k = 3;

  // Same as 'a', 'b' variables in the paper.
  double a = 4.0, b = 0.1;

  // The radius that we run the algorithm on.
  double radius = 2;

  // Same as 'alpha' variable in the paper.
  double alpha = 0.4;

  // A number for random seed.
  uint64_t seed = 123456789;

  // Run experiments.
  RunExperiment(n, k, a, b, radius, alpha, seed);
  return 0;
}
