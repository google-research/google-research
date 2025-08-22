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

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace exposure_design {

using DiversionId = std::string;
using OutcomeIndex = std::size_t;
using SparseVector = std::vector<std::pair<OutcomeIndex, double>>;

struct Instance {
  std::unordered_map<DiversionId, SparseVector> diversion_units;
};

struct Parameters {
  // Controls the trade-off between higher individual exposure variance (lesser
  // values of phi) and lower exposure correlation (greater values of phi).
  double phi{0.001};
  // Each cluster is limited to a 1/k unweighted fraction of the edges.
  double k{1000};
  // Number of iterations.
  int T{1000};
  // If true, log intermediate objective values to standard error.
  bool verbose{false};
};

using DiversionCluster = std::vector<DiversionId>;

struct DiversionClustering {
  double objective{0};
  std::vector<DiversionCluster> clusters;
};

DiversionClustering ComputeClustering(const Instance& instance,
                                      const Parameters& parameters);

}  // namespace exposure_design
