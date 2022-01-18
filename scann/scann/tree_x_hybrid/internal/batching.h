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



#ifndef SCANN_TREE_X_HYBRID_INTERNAL_BATCHING_H_
#define SCANN_TREE_X_HYBRID_INTERNAL_BATCHING_H_

#include "scann/base/search_parameters.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace tree_x_internal {

template <typename T>
bool SupportsLowLevelBatching(const TypedDataset<T>& queries,
                              ConstSpan<SearchParameters> params) {
  if (!queries.IsDense()) return false;
  for (const SearchParameters& p : params) {
    if (p.pre_reordering_crowding_enabled() || p.restricts_enabled()) {
      return false;
    }
  }
  return true;
}

template <typename Mutator>
void AddLeafResultsToTopN(ConstSpan<DatapointIndex> local_to_global_index,
                          const float distance_to_center_adjustment,
                          const float cluster_stdev_adjustment,
                          ConstSpan<pair<DatapointIndex, float>> leaf_results,
                          Mutator* mutator) {
  float epsilon = mutator->epsilon();
  for (const auto& result : leaf_results) {
    float dist = result.second * cluster_stdev_adjustment +
                 distance_to_center_adjustment;
    if (dist <= epsilon) {
      if (ABSL_PREDICT_FALSE(
              mutator->Push(local_to_global_index[result.first], dist))) {
        mutator->GarbageCollect();
        epsilon = mutator->epsilon();
      }
    }
  }
}

struct QueryForResidualLeaf {
  QueryForResidualLeaf() {}
  QueryForResidualLeaf(DatapointIndex query_index, float distance_to_center)
      : query_index(query_index), distance_to_center(distance_to_center) {}

  DatapointIndex query_index = 0;
  float distance_to_center = NAN;
};

inline DatapointIndex QueryIndex(DatapointIndex x) { return x; }
inline DatapointIndex QueryIndex(const QueryForResidualLeaf& q) {
  return q.query_index;
}

inline float DistanceToCenterAdjustment(DatapointIndex query_index) {
  return 0.0f;
}

inline float DistanceToCenterAdjustment(const QueryForResidualLeaf& q) {
  return q.distance_to_center;
}

template <typename QueryForLeaf>
vector<SearchParameters> CreateParamsSubsetForLeaf(
    ConstSpan<SearchParameters> params,
    ConstSpan<FastTopNeighbors<float>::Mutator> mutators,
    ConstSpan<shared_ptr<const SearcherSpecificOptionalParameters>>
        leaf_optional_params,
    ConstSpan<QueryForLeaf> queries_for_leaf) {
  vector<SearchParameters> result;
  result.reserve(queries_for_leaf.size());
  for (const QueryForLeaf& q : queries_for_leaf) {
    const DatapointIndex query_index = QueryIndex(q);
    SearchParameters leaf_params;
    leaf_params.set_pre_reordering_num_neighbors(
        params[query_index].pre_reordering_num_neighbors());
    leaf_params.set_pre_reordering_epsilon(mutators[query_index].epsilon() -
                                           DistanceToCenterAdjustment(q));
    leaf_params.set_searcher_specific_optional_parameters(
        leaf_optional_params[query_index]);
    result.emplace_back(std::move(leaf_params));
  }
  return result;
}

template <typename T>
size_t RecursiveSize(const T& vec) {
  size_t result = 0;
  for (auto& inner : vec) {
    result += inner.size();
  }
  return result;
}

}  // namespace tree_x_internal
}  // namespace research_scann

#endif
