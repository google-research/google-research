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



#include "scann/base/search_parameters.h"
#include "scann/data_format/datapoint.h"
#include "scann/utils/types.h"

#ifndef SCANN_TREE_X_HYBRID_LEAF_SEARCHER_OPTIONAL_PARAMETER_CREATOR_H_
#define SCANN_TREE_X_HYBRID_LEAF_SEARCHER_OPTIONAL_PARAMETER_CREATOR_H_

namespace research_scann {

template <typename T>
class LeafSearcherOptionalParameterCreator {
 public:
  virtual ~LeafSearcherOptionalParameterCreator() {}

  virtual StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
  CreateLeafSearcherOptionalParameters(const DatapointPtr<T>& query) const = 0;
};

}  // namespace research_scann

#endif
