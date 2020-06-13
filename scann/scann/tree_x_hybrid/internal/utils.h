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
  for (int i = 0; i < n_leaves; i++) {
    TF_ASSIGN_OR_RETURN(
        leaf_opts[i], leaf_searchers[i]->ExtractSingleMachineFactoryOptions());
    if (leaf_opts[i].hashed_dataset != nullptr) {
      hash_ct++;
      total_hashed += leaf_opts[i].hashed_dataset->size();
      const int cur_dims = leaf_opts[i].hashed_dataset->dimensionality();
      if (hash_dim == -1)
        hash_dim = cur_dims;
      else if (hash_dim != cur_dims)
        return FailedPreconditionError(
            "Dimensionality mismatch among hashed leaf datasets");
    }
    if (leaf_opts[i].ah_codebook != nullptr) codebook_ct++;
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
  return opts;
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
