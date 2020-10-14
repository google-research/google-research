// Copyright 2020 The Google Research Authors.
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

#include <math.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "compute_cost.h"
#include "fast_k_means_algo.h"
#include "kmeanspp_seeding.h"
#include "rejection_sampling_lsh.h"

using std::cout;
using std::endl;
using std::vector;

namespace fast_k_means {

vector<vector<double> > ReadInput() {
  int n, d;
  vector<vector<double> > input_point;
  FILE* fp;
  // Add the input file here.  We expect the number of points and number of
  // dimension in the first line (space separated). Followed by one line for
  // each input point, describing the dimensions in double format
  // (space separated). Input example with three points each have 2 dimensions:
  // 3 2
  // 1.00 2.50
  // 3.30 4.12
  // 0.0 -10.0
  fp = fopen(
      "/usr/local/google/home/ashkannorouzi/Documents/fast-kmedian/dataset/"
      "cloud.data",
      "r");
  fscanf(fp, "%d%d", &n, &d);
    cout << n << " " << d << endl;
  for (int i = 0; i < n; i++) {
    input_point.push_back(vector<double>(d));
    for (int j = 0; j < d; j++) fscanf(fp, "%lf", &input_point[i][j]);
  }
  return input_point;
}

}  // namespace fast_k_means

// The variables of the algorithm.

// Number of centers
int k = 0;
// Number of trees that we embed.
int number_of_trees = 4;
// The scalling factor of the input before casted to integer.
double scaling_factor = 1.0;
// Number of greedy samples in case one is runnig the greedy version of the
// algorithms.
int number_greedy_samples = 0;
// Multiples the probability of accepting a center in rejection sampling by this
// factor. We recommend to use sqrt(# of dimensions) or # of dimensions.
double boosting_prob_factor = 1.0;

// Fast k-means algorithm. The rest of the function are the same experiments for
// different algorithms.
void FastKmeansExperiment(const vector<vector<double>>& input_point) {
  fast_k_means::FastKMeansAlgo fast_algo;

  // Running the algotihm, the result is stored in fast_algo.centers_ vector.
  clock_t start_time = clock();
  fast_algo.RunAlgorithm(input_point, k, number_of_trees, scaling_factor,
                         number_greedy_samples + 1);
  cout << "Running time for the Fast Algorithm in seconds: "
       << static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC << endl;

  // One can compute the cost of the solution as follows
  cout << "The cost of the solution is: "
       << fast_k_means::ComputeCost::GetCost(input_point, fast_algo.centers)
       << endl;
}

void RejectionSamplingLSHExperiment(const vector<vector<double>>& input_point) {
  clock_t start_time = clock();
  fast_k_means::RejectionSamplingLSH rejsamplelsh_algo;
  rejsamplelsh_algo.RunAlgorithm(input_point, k, number_of_trees,
                                 scaling_factor, number_greedy_samples + 1,
                                 boosting_prob_factor);
  cout << "Running time for the Fast Algorithm in seconds: "
       << static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC << endl;

  // One can compute the cost of the solution as follows
  cout << "The cost of the solution is: "
       << fast_k_means::ComputeCost::GetCost(input_point,
                                             rejsamplelsh_algo.centers)
       << endl;
}

void KMeansPPSeedingExperiment(const vector<vector<double>>& input_point) {
  clock_t start_time = clock();
  fast_k_means::KMeansPPSeeding kmeanspp_algo;
  kmeanspp_algo.RunAlgorithm(input_point, k, number_greedy_samples + 1);
  cout << "Running time for the Fast Algorithm in seconds: "
       << static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC << endl;

  // One can compute the cost of the solution as follows
  cout << "The cost of the solution is: "
       << fast_k_means::ComputeCost::GetCost(input_point,
                                             kmeanspp_algo.centers_)
       << endl;
}

int main() {
  vector<vector<double> > input_point = fast_k_means::ReadInput();
  // Fixing the variables explained above.
  k = 10;
  boosting_prob_factor = sqrt(input_point[0].size());

  // Call the functions here.
  // This part is supposed to be very short.
  cout << "Starting FastKmeans Experiment" << endl;
  FastKmeansExperiment(input_point);
  cout << "Starting Rejection Sampling LSH Experiment" << endl;
  RejectionSamplingLSHExperiment(input_point);
  cout << "Starting KMeans++ Seeding Experiment" << endl;
  KMeansPPSeedingExperiment(input_point);
}
