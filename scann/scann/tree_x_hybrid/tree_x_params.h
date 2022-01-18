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



#ifndef SCANN_TREE_X_HYBRID_TREE_X_PARAMS_H_
#define SCANN_TREE_X_HYBRID_TREE_X_PARAMS_H_

#include <cstdint>

#include "scann/base/search_parameters.h"
#include "scann/utils/types.h"

namespace research_scann {

class TreeXOptionalParameters final
    : public SearcherSpecificOptionalParameters {
 public:
  TreeXOptionalParameters();
  ~TreeXOptionalParameters() override;

  Status EnablePreTokenization(vector<int32_t> leaf_tokens_to_search);

  void DisableTreeXPreTokenization() { leaf_tokens_to_search_.clear(); }

  bool pre_tokenization_enabled() const {
    return !leaf_tokens_to_search_.empty();
  }

  ConstSpan<int32_t> leaf_tokens_to_search() const {
    return leaf_tokens_to_search_;
  }

  shared_ptr<const SearcherSpecificOptionalParameters>
  all_leaf_optional_params() const {
    return all_leaf_optional_params_;
  }

  void set_all_leaf_optional_params(
      shared_ptr<const SearcherSpecificOptionalParameters> val) {
    all_leaf_optional_params_ = std::move(val);
  }

  int32_t num_partitions_to_search_override() const {
    return num_partitions_to_search_override_;
  }

  void set_num_partitions_to_search_override(
      int32_t num_partitions_to_search_override) {
    num_partitions_to_search_override_ = num_partitions_to_search_override;
  }

 private:
  vector<int32_t> leaf_tokens_to_search_ = {};

  int32_t num_partitions_to_search_override_ = 0;

  shared_ptr<const SearcherSpecificOptionalParameters>
      all_leaf_optional_params_;
};

}  // namespace research_scann

#endif
