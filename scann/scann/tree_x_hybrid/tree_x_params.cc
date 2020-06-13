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



#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/base/search_parameters.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

TreeXOptionalParameters::TreeXOptionalParameters() {}
TreeXOptionalParameters::~TreeXOptionalParameters() {}

Status TreeXOptionalParameters::EnablePreTokenization(
    vector<int32_t> leaf_tokens_to_search) {
  if (leaf_tokens_to_search.empty()) {
    return InvalidArgumentError(
        "leaf_tokens_to_search cannot be empty on calls to "
        "EnablePreTokenization.");
  }

  if (pre_tokenization_enabled()) {
    return FailedPreconditionError(
        "Pre-tokenization cannot be enabled if it is already enabled.");
  }

  leaf_tokens_to_search_ = std::move(leaf_tokens_to_search);
  return OkStatus();
}

Status TreeXOptionalParameters::EnablePreTokenization(
    vector<int32_t> leaf_tokens_to_search,
    vector<shared_ptr<const SearcherSpecificOptionalParameters>>
        leaf_params_by_token) {
  if (leaf_tokens_to_search.empty()) {
    return InvalidArgumentError(
        "leaf_tokens_to_search cannot be empty on calls to "
        "EnablePreTokenization.");
  }

  if (pre_tokenization_enabled()) {
    return FailedPreconditionError(
        "Pre-tokenization cannot be enabled if it is already enabled.");
  }

  if (leaf_tokens_to_search.size() != leaf_params_by_token.size()) {
    return InvalidArgumentError(
        absl::StrCat("Size mismatch between leaf_tokens_to_search and "
                     "leaf_params_by_token (",
                     leaf_tokens_to_search.size(), " vs. ",
                     leaf_params_by_token.size(), ")."));
  }

  leaf_tokens_to_search_ = std::move(leaf_tokens_to_search);
  leaf_params_by_token_ = std::move(leaf_params_by_token);
  return OkStatus();
}

}  // namespace scann_ops
}  // namespace tensorflow
