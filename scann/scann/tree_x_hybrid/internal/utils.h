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

#ifndef SCANN_TREE_X_HYBRID_INTERNAL_UTILS_H_
#define SCANN_TREE_X_HYBRID_INTERNAL_UTILS_H_

#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/restricts/restrict_allowlist.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

#ifdef __x86_64__
SCANN_AVX2_OUTLINE size_t Avx2GatherCreateLeafLocalAllowlist(
    RestrictAllowlistConstView global_allowlist,
    RestrictAllowlistMutableView leaf_view,
    ConstSpan<DatapointIndex> leaf_local_to_global);
#endif

template <typename T, bool kSoar = false, typename GetDatasetFunctor>
std::conditional_t<kSoar, StatusOr<pair<vector<T>, vector<T>>>,
                   StatusOr<vector<T>>>
CombineLeafDatasets(size_t expected_size, const string_view name,
                    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
                    GetDatasetFunctor F) {
  ssize_t count = 0, total_size = 0, dimensionality = -1;
  for (int leaf : Seq(datapoints_by_token.size())) {
    const DenseDataset<T>* dataset_ptr = F(leaf);
    if (dataset_ptr == nullptr) continue;
    count++;
    total_size += dataset_ptr->size();
    if (!dataset_ptr->empty()) {
      if (dimensionality == -1)
        dimensionality = dataset_ptr->dimensionality();
      else if (dimensionality != dataset_ptr->dimensionality())
        return FailedPreconditionError(
            "Dimensionality mismatch among leaf %s datasets: %d vs %d", name,
            dimensionality, dataset_ptr->dimensionality());
    }
  }

  if (count == 0) {
    if constexpr (kSoar) {
      return std::make_pair(vector<T>(), vector<T>());
    } else {
      return vector<T>();
    }
  }
  if (count != datapoints_by_token.size())
    return FailedPreconditionError("Leaf %s dataset count mismatch: %d vs %d",
                                   name, count, datapoints_by_token.size());
  if (total_size < expected_size || total_size > expected_size * 2)
    return FailedPreconditionError(
        "Unexpected total leaf size of %d (dataset size = %d)", total_size,
        expected_size);

  vector<T> combined(dimensionality * expected_size);
  vector<T> combined_soar(kSoar ? combined.size() : 0);
  vector<bool> seen(kSoar ? expected_size : 0);
  for (int leaf : Seq(datapoints_by_token.size())) {
    const DenseDataset<T>* dataset_ptr = F(leaf);
    for (const auto [inner_idx, global_idx] :
         Enumerate(datapoints_by_token[leaf])) {
      auto dest_iter = combined.begin() + dimensionality * global_idx;
      if constexpr (kSoar) {
        if (seen[global_idx])
          dest_iter = combined_soar.begin() + dimensionality * global_idx;
        seen[global_idx] = true;
      }
      std::copy(dataset_ptr->data(inner_idx).begin(),
                dataset_ptr->data(inner_idx).end(), dest_iter);
    }
  }
  if constexpr (kSoar) {
    return std::make_pair(std::move(combined), std::move(combined_soar));
  } else {
    return combined;
  }
}

namespace tree_ah_utils_internal {

StatusOr<SingleMachineFactoryOptions> FinishMergeAHLeafOptions(
    MutableSpan<SingleMachineFactoryOptions> leaf_opts,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    const int expected_size, const float spilling_mult);

}

template <template <class> class V, typename T>
StatusOr<SingleMachineFactoryOptions> MergeAHLeafOptions(
    const vector<unique_ptr<V<T>>>& leaf_searchers,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    const int expected_size, const float spilling_mult) {
  const int n_leaves = leaf_searchers.size();
  DCHECK_EQ(datapoints_by_token.size(), n_leaves);
  std::vector<SingleMachineFactoryOptions> leaf_opts(n_leaves);
  for (int i = 0; i < n_leaves; i++) {
    SCANN_ASSIGN_OR_RETURN(
        leaf_opts[i], leaf_searchers[i]->ExtractSingleMachineFactoryOptions());
  }
  return tree_ah_utils_internal::FinishMergeAHLeafOptions(
      MakeMutableSpan(leaf_opts), datapoints_by_token, expected_size,
      spilling_mult);
}

StatusOr<bool> ValidateDatapointsByToken(
    absl::Span<const std::vector<DatapointIndex>> datapoints_by_token,
    DatapointIndex num_datapoints);

vector<uint32_t> SizeByPartition(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

void DeduplicateDatabaseSpilledResults(NNResultsVector* results,
                                       size_t final_size);

inline int32_t SafeIntFloatMul(int32_t x, float f) {
  const double result = static_cast<double>(x) * static_cast<double>(f);
  if (ABSL_PREDICT_FALSE(result >
                         static_cast<double>(numeric_limits<int32_t>::max()))) {
    return numeric_limits<int32_t>::max();
  } else if (ABSL_PREDICT_FALSE(result < static_cast<double>(
                                             numeric_limits<int32_t>::min()))) {
    return numeric_limits<int32_t>::min();
  } else {
    return static_cast<int32_t>(result);
  }
}

}  // namespace research_scann

#endif
