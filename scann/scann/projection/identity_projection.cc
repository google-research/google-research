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



#include "scann/projection/identity_projection.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
template <typename FloatT>
Status IdentityProjection<T>::ProjectInputImpl(
    const DatapointPtr<T>& input, Datapoint<FloatT>* projected) const {
  CHECK(projected != nullptr);
  projected->clear();

  DimensionIndex dims = input.dimensionality();
  projected->mutable_values()->resize(dims);

  if (input.IsDense()) {
    for (size_t i = 0; i < dims; ++i) {
      projected->mutable_values()->at(i) =
          static_cast<FloatT>(input.values()[i]);
    }
  } else {
    for (size_t i = 0; i < input.nonzero_entries(); ++i) {
      const auto dim = input.indices()[i];
      projected->mutable_values()->at(dim) =
          static_cast<FloatT>(input.values()[i]);
    }
  }

  return OkStatus();
}

DEFINE_PROJECT_INPUT_OVERRIDES(IdentityProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, IdentityProjection);

}  // namespace scann_ops
}  // namespace tensorflow
