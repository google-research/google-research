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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/utils/top_n_amortized_constant.h"

#include "absl/flags/flag.h"
#include "scann/utils/zip_sort.h"

ABSL_RETIRED_FLAG(bool, use_branch_optimized_top_n, , );

namespace tensorflow {
namespace scann_ops {

template <typename Distance>
void TopNeighbors<Distance>::PartitionElements(vector<Neighbor>* elements,
                                               const DistanceComparator& cmp) {
  ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                               this->limit() - 1, elements->begin(),
                               elements->end());
}

SCANN_INSTANTIATE_TYPED_CLASS(, TopNeighbors);

}  // namespace scann_ops
}  // namespace tensorflow
