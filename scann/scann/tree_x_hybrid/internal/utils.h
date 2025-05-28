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
  SingleMachineFactoryOptions opts;

  const auto get_ah = [&](int leaf_idx) {
    return leaf_opts[leaf_idx].hashed_dataset.get();
  };
  if (spilling_mult > 1) {
    using PairOfVectors = pair<vector<uint8_t>, vector<uint8_t>>;
    SCANN_ASSIGN_OR_RETURN(
        PairOfVectors ah_datasets,
        (CombineLeafDatasets<uint8_t, true>(expected_size, "AH",
                                            datapoints_by_token, get_ah)));
    if (!ah_datasets.first.empty()) {
      opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
          std::move(ah_datasets.first), expected_size);
      opts.soar_hashed_dataset = make_shared<DenseDataset<uint8_t>>(
          std::move(ah_datasets.second), expected_size);
    }
  } else {
    SCANN_ASSIGN_OR_RETURN(
        vector<uint8_t> ah_dataset,
        (CombineLeafDatasets<uint8_t>(expected_size, "AH", datapoints_by_token,
                                      get_ah)));
    if (!ah_dataset.empty()) {
      opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
          std::move(ah_dataset), expected_size);
    }
  }
  if (n_leaves >= 1) {
    opts.ah_codebook = leaf_opts[0].ah_codebook;
  }
  if (opts.hashed_dataset != nullptr && !opts.hashed_dataset->empty()) {
    std::string codebook_proto_str;
    leaf_opts[0].ah_codebook->SerializeToString(&codebook_proto_str);

    for (int i = 1; i < n_leaves; i++) {
      std::string codebook_to_compare;
      leaf_opts[i].ah_codebook->SerializeToString(&codebook_to_compare);
      if (codebook_proto_str != codebook_to_compare)
        return FailedPreconditionError("Inconsistent codebooks among leaves");
    }
  }

  const auto get_int8 = [&](int leaf_idx) -> DenseDataset<int8_t>* {
    auto fp = leaf_opts[leaf_idx].pre_quantized_fixed_point;
    if (fp == nullptr) return nullptr;
    return fp->fixed_point_dataset.get();
  };
  SCANN_ASSIGN_OR_RETURN(
      vector<int8_t> int8_dataset,
      (CombineLeafDatasets<int8_t>(expected_size, "INT8", datapoints_by_token,
                                   get_int8)));
  if (!int8_dataset.empty()) {
    opts.pre_quantized_fixed_point = make_shared<PreQuantizedFixedPoint>();
    opts.pre_quantized_fixed_point->fixed_point_dataset =
        make_shared<DenseDataset<int8_t>>(std::move(int8_dataset),
                                          expected_size);

    bool int8_has_norms = false;
    for (int i = 0; i < n_leaves; i++) {
      auto int8 = leaf_opts[i].pre_quantized_fixed_point;
      if (int8 && int8->squared_l2_norm_by_datapoint &&
          !int8->squared_l2_norm_by_datapoint->empty())
        int8_has_norms = true;
    }
    if (int8_has_norms) {
      opts.pre_quantized_fixed_point->squared_l2_norm_by_datapoint =
          make_shared<vector<float>>(expected_size);
      for (int i = 0; i < n_leaves; i++) {
        auto int8 = leaf_opts[i].pre_quantized_fixed_point;
        for (const auto [inner_idx, global_idx] :
             Enumerate(datapoints_by_token[i])) {
          opts.pre_quantized_fixed_point->squared_l2_norm_by_datapoint->at(
              global_idx) = int8->squared_l2_norm_by_datapoint->at(inner_idx);
        }
      }
    }
  }
  return opts;
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
