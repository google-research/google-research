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

#include "clustering/in-memory-clusterer.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

absl::Status InMemoryClusterer::Graph::FinishImport() {
  return absl::OkStatus();
}

absl::StatusOr<std::vector<InMemoryClusterer::Clustering>>
InMemoryClusterer::HierarchicalCluster(const ClustererConfig& config) const {
  InMemoryClusterer::Clustering clusters;
  ASSIGN_OR_RETURN(clusters, Cluster(config));
  return std::vector<Clustering>{clusters};
}

std::string InMemoryClusterer::StringId(NodeId id) const {
  if (node_id_map_ == nullptr) {
    return absl::StrCat(id);
  } else if (id >= 0 && id < node_id_map_->size()) {
    return (*node_id_map_)[id];
  } else {
    return absl::StrCat("missing-id-", id);
  }
}

}  // namespace in_memory
}  // namespace research_graph
