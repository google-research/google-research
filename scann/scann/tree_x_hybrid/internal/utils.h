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

#ifndef SCANN_TREE_X_HYBRID_INTERNAL_UTILS_H_
#define SCANN_TREE_X_HYBRID_INTERNAL_UTILS_H_

#include <cstdint>

#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/logging.h"

namespace research_scann {

inline void TranslateGlobalToLeafLocalWhitelist(
    const SearchParameters& params,
    ConstSpan<DatapointIndex> leaf_local_to_global,
    SearchParameters* leaf_params) {}

template <typename T, typename GetDatasetFunctor>
StatusOr<vector<T>> CombineLeafDatasets(
    size_t expected_size, const string_view name,
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

  if (count == 0) return vector<T>();
  if (count != datapoints_by_token.size())
    return FailedPreconditionError("Leaf %s dataset count mismatch: %d vs %d",
                                   name, count, datapoints_by_token.size());
  if (expected_size != total_size)
    return FailedPreconditionError("Leaf %s dataset size mismatch: %d vs %d",
                                   name, expected_size, total_size);

  vector<T> combined(dimensionality * expected_size);
  for (int leaf : Seq(datapoints_by_token.size())) {
    const DenseDataset<T>* dataset_ptr = F(leaf);
    for (const auto [inner_idx, global_idx] :
         Enumerate(datapoints_by_token[leaf])) {
      std::copy(dataset_ptr->data(inner_idx).begin(),
                dataset_ptr->data(inner_idx).end(),
                combined.begin() + dimensionality * global_idx);
    }
  }
  return combined;
}

template <template <class> class V, typename T>
StatusOr<SingleMachineFactoryOptions> MergeAHLeafOptions(
    const vector<unique_ptr<V<T>>>& leaf_searchers,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    const int expected_size) {
  const int n_leaves = leaf_searchers.size();
  DCHECK_EQ(datapoints_by_token.size(), n_leaves);
  std::vector<SingleMachineFactoryOptions> leaf_opts(n_leaves);
  for (int i = 0; i < n_leaves; i++) {
    TF_ASSIGN_OR_RETURN(
        leaf_opts[i], leaf_searchers[i]->ExtractSingleMachineFactoryOptions());
  }
  SingleMachineFactoryOptions opts;

  const auto get_ah = [&](int leaf_idx) {
    return leaf_opts[leaf_idx].hashed_dataset.get();
  };
  TF_ASSIGN_OR_RETURN(vector<uint8_t> ah_dataset,
                      (CombineLeafDatasets<uint8_t>(
                          expected_size, "AH", datapoints_by_token, get_ah)));
  if (!ah_dataset.empty()) {
    opts.hashed_dataset =
        make_shared<DenseDataset<uint8_t>>(ah_dataset, expected_size);

    opts.ah_codebook = leaf_opts[0].ah_codebook;
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
  TF_ASSIGN_OR_RETURN(
      vector<int8_t> int8_dataset,
      (CombineLeafDatasets<int8_t>(expected_size, "INT8", datapoints_by_token,
                                   get_int8)));
  if (!int8_dataset.empty()) {
    opts.pre_quantized_fixed_point = make_shared<PreQuantizedFixedPoint>();
    opts.pre_quantized_fixed_point->fixed_point_dataset =
        make_shared<DenseDataset<int8_t>>(int8_dataset, expected_size);

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

}  // namespace research_scann

#endif
