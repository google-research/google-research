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

#include "scann/projection/truncate_projection.h"

#include "scann/utils/datapoint_utils.h"

namespace research_scann {

template <typename T>
Status TruncateProjection<T>::ProjectInput(const DatapointPtr<T>& input,
                                           Datapoint<double>* projected) const {
  SCANN_RET_CHECK_EQ(input.dimensionality(), input_dims_);
  SCANN_RET_CHECK(input.IsDense())
      << "TruncateProjection only works with dense data.";
  DatapointPtr<T> truncated(nullptr, input.values(), projected_dims_,
                            projected_dims_);
  CopyToDatapoint(truncated, projected);
  return OkStatus();
}

template <typename T>
Status TruncateProjection<T>::ProjectInput(const DatapointPtr<T>& input,
                                           Datapoint<float>* projected) const {
  SCANN_RET_CHECK(input.IsDense())
      << "TruncateProjection only works with dense data.";
  SCANN_RET_CHECK_EQ(input.dimensionality(), input_dims_);
  DatapointPtr<T> truncated(nullptr, input.values(), projected_dims_,
                            projected_dims_);
  CopyToDatapoint(truncated, projected);
  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, TruncateProjection);

}  // namespace research_scann
