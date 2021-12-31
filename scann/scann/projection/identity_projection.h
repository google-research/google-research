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



#ifndef SCANN_PROJECTION_IDENTITY_PROJECTION_H_
#define SCANN_PROJECTION_IDENTITY_PROJECTION_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/projection/projection_base.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class IdentityProjection : public Projection<T> {
 public:
  IdentityProjection() {}

  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<float>* projected) const override;
  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<double>* projected) const override;

  int32_t projected_dimensionality() const override { return -1; }

 private:
  template <typename FloatT>
  Status ProjectInputImpl(const DatapointPtr<T>& input,
                          Datapoint<FloatT>* projected) const;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, IdentityProjection);

}  // namespace research_scann

#endif
