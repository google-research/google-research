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

#ifndef MAIN_CANN_RG_H_
#define MAIN_CANN_RG_H_

#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"

namespace cann_rg {
typedef std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>>
    DescriptorData;

struct ScoredPair {
  std::string query;
  std::string target;
  double score;

  std::string ToString() const;
};

// Reads a list of descriptor files names.
std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>> ReadList(
    const std::vector<std::string> &file_list, int dim);

// Computes the image retrieval and their score for a set of descriptos (index
// and query). The top 50 images per query are retrieved.
std::vector<ScoredPair> ComputeQueryScores(const DescriptorData &query_set,
                                           const DescriptorData &index_set,
                                           int num_features, int dim,
                                           double c_approx, int num_grids_init,
                                           double eps, double p0, double p1,
                                           double min_distance);
// A wrapper for the above function that first reads the names of the files and
// then calls it. It is needed for the GCP parallel (scaled) implementation
// where several workers are applied together after the data is loaded.
std::vector<ScoredPair> MatchImages(
    const std::vector<std::string> &query_file_names,
    const std::vector<std::string> &mapping_file_names, int num_features,
    int dim, double c_approx, int num_grids_init, double eps, double p0,
    double p1, double min_distance);

}  // namespace cann_rg

#endif  // MAIN_CANN_RG_H_
