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

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exposure_design.h"

namespace exposure_design {
namespace {

using OutcomeId = std::string;

class Indexer {
 public:
  OutcomeIndex IndexOf(const OutcomeId& id) {
    return indexes_.emplace(id, indexes_.size()).first->second;
  }

 private:
  std::unordered_map<OutcomeId, OutcomeIndex> indexes_;
};

Instance ReadInstance(std::istream& stream) {
  std::unordered_map<DiversionId, SparseVector> diversion_units;
  Indexer indexer;
  DiversionId diversion_id;
  OutcomeId outcome_id;
  double weight;
  while (stream >> diversion_id >> outcome_id >> weight)
    diversion_units[diversion_id].push_back(
        {indexer.IndexOf(outcome_id), weight});
  return Instance{std::move(diversion_units)};
}

void WriteClustering(std::ostream& stream,
                     const DiversionClustering& clustering) {
  stream << clustering.objective << "\n";
  for (const DiversionCluster& cluster : clustering.clusters) {
    bool space{false};
    for (const DiversionId& diversion_id : cluster) {
      if (space) stream << " ";
      stream << diversion_id;
      space = true;
    }
    stream << "\n";
  }
}

void Main() {
  Parameters parameters;
  if (const char* phi{std::getenv("phi")}) parameters.phi = std::atof(phi);
  if (const char* k{std::getenv("k")}) parameters.k = std::atof(k);
  if (const char* T{std::getenv("T")}) parameters.T = std::atoi(T);
  if (const char* verbose{std::getenv("verbose")})
    parameters.verbose = static_cast<bool>(std::atoi(verbose));
  WriteClustering(std::cout,
                  ComputeClustering(ReadInstance(std::cin), parameters));
}

}  // namespace
}  // namespace exposure_design

int main() { exposure_design::Main(); }
