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



#ifndef SCANN_PROJECTION_TRUNCATE_PROJECTION_H_
#define SCANN_PROJECTION_TRUNCATE_PROJECTION_H_

#include <algorithm>
#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/projection/projection_base.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class TruncateProjection : public Projection<T> {
 public:
  TruncateProjection(const int32_t input_dims, const int32_t projected_dims)
      : input_dims_(input_dims),
        projected_dims_(std::min(projected_dims, input_dims)) {}

  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<double>* projected) const final;
  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<float>* projected) const final;

 private:
  int32_t input_dims_ = 0;
  int32_t projected_dims_ = 0;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TruncateProjection);

}  // namespace research_scann

#endif
