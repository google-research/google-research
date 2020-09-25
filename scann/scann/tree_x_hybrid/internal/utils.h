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

#ifndef SCANN__TREE_X_HYBRID_INTERNAL_UTILS_H_
#define SCANN__TREE_X_HYBRID_INTERNAL_UTILS_H_

#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace scann_ops {

inline void TranslateGlobalToLeafLocalWhitelist(
    const SearchParameters& params,
    ConstSpan<DatapointIndex> leaf_local_to_global,
    SearchParameters* leaf_params) {}

template <template <class> class V, typename T>
StatusOr<SingleMachineFactoryOptions> MergeAHLeafOptions(
    const vector<unique_ptr<V<T>>>& leaf_searchers,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    const int expected_size) {
  const int n_leaves = leaf_searchers.size();
  auto leaf_opts = std::vector<SingleMachineFactoryOptions>(n_leaves);

  int hash_ct = 0, codebook_ct = 0, total_hashed = 0, hash_dim = -1;

  ssize_t int8_ct = 0, total_int8 = 0, int8_dim = -1;
  bool int8_has_norms = false;
  for (int i = 0; i < n_leaves; i++) {
    TF_ASSIGN_OR_RETURN(
        leaf_opts[i], leaf_searchers[i]->ExtractSingleMachineFactoryOptions());
    if (leaf_opts[i].hashed_dataset != nullptr) {
      hash_ct++;
      size_t cur_size = leaf_opts[i].hashed_dataset->size();
      total_hashed += cur_size;
      const int cur_dims = leaf_opts[i].hashed_dataset->dimensionality();

      if (cur_size > 0) {
        if (hash_dim == -1)
          hash_dim = cur_dims;
        else if (hash_dim != cur_dims)
          return FailedPreconditionError(absl::StrFormat(
              "Dimensionality mismatch among hashed leaf datasets: %d vs %d",
              hash_dim, cur_dims));
      }
    }
    if (leaf_opts[i].ah_codebook != nullptr) codebook_ct++;
    auto int8_t = leaf_opts[i].pre_quantized_fixed_point;
    if (int8_t != nullptr) {
      auto dataset = int8_t->fixed_point_dataset;
      if (dataset) {
        int8_ct++;
        total_int8 += dataset->size();
        int8_dim = dataset->dimensionality();
      }
      auto l2_norms = int8_t->squared_l2_norm_by_datapoint;
      if (l2_norms && !l2_norms->empty()) {
        if (!dataset || dataset->size() != l2_norms->size())
          return FailedPreconditionError(
              "Int8-quantized dataset: number of squared L2 norms inconsistent "
              "with dataset size: %d vs %d",
              dataset->size(), l2_norms->size());
        int8_has_norms = true;
      }
    }
  }

  SingleMachineFactoryOptions opts;
  if (hash_ct != 0 || codebook_ct != 0 || total_hashed != 0) {
    if (hash_ct != n_leaves)
      return FailedPreconditionError(
          absl::StrFormat("Detected tree-AH hybrid but not all (%d/%d) leaf "
                          "searchers have hashed datasets",
                          hash_ct, n_leaves));
    if (codebook_ct != n_leaves)
      return FailedPreconditionError(
          "Detected tree-AH hybrid but not all leaf searchers have AH "
          "codebooks");
    if (total_hashed != expected_size)
      return FailedPreconditionError(
          "Detected tree-AH hybrid but sum of leaf searcher hashed datasets "
          "doesn't equal expected dataset size");

    opts.ah_codebook = leaf_opts[0].ah_codebook;
    std::string codebook_proto_str;
    leaf_opts[0].ah_codebook->SerializeToString(&codebook_proto_str);

    for (int i = 1; i < n_leaves; i++) {
      std::string codebook_to_compare;
      leaf_opts[i].ah_codebook->SerializeToString(&codebook_to_compare);
      if (codebook_proto_str != codebook_to_compare)
        return FailedPreconditionError("Inconsistent codebooks among leaves");
    }

    vector<uint8_t> storage(hash_dim * expected_size);
    for (int i = 0; i < n_leaves; i++) {
      int inner_idx = 0;
      for (const auto dptr : *leaf_opts[i].hashed_dataset) {
        const uint64_t res_idx = datapoints_by_token[i][inner_idx++];
        std::copy(dptr.values(), dptr.values() + hash_dim,
                  storage.begin() + res_idx * hash_dim);
      }
    }
    opts.hashed_dataset =
        make_shared<DenseDataset<uint8_t>>(storage, expected_size);
  }
  if (int8_ct != 0) {
    if (int8_ct != n_leaves)
      return FailedPreconditionError(absl::StrFormat(
          "Detected tree-scalar quantization hybrid but not all (%d/%d) leaf "
          "searchers have int8_t-quantized datasets",
          hash_ct, n_leaves));
    if (total_int8 != expected_size)
      return FailedPreconditionError(
          "Detected tree-scalar quantization hybrid but sum of leaf searcher "
          "datasets doesn't equal expected dataset size");
    opts.pre_quantized_fixed_point = make_shared<PreQuantizedFixedPoint>();

    vector<int8_t> storage(int8_dim * expected_size);
    if (int8_has_norms)
      opts.pre_quantized_fixed_point->squared_l2_norm_by_datapoint =
          make_shared<vector<float>>(expected_size);
    for (int i = 0; i < n_leaves; i++) {
      auto int8_t = leaf_opts[i].pre_quantized_fixed_point;
      for (size_t inner_idx : Seq(int8_t->fixed_point_dataset->size())) {
        const uint64_t global_idx = datapoints_by_token[i][inner_idx];

        auto dptr = (*int8_t->fixed_point_dataset)[inner_idx];
        std::copy(dptr.values(), dptr.values() + int8_dim,
                  storage.begin() + global_idx * int8_dim);

        if (int8_has_norms)
          opts.pre_quantized_fixed_point->squared_l2_norm_by_datapoint->at(
              global_idx) = int8_t->squared_l2_norm_by_datapoint->at(inner_idx);
      }
    }
    opts.pre_quantized_fixed_point->fixed_point_dataset =
        make_shared<DenseDataset<int8_t>>(storage, expected_size);
  }
  return opts;
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
