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



#ifndef SCANN__PROJECTION_PROJECTION_BASE_H_
#define SCANN__PROJECTION_PROJECTION_BASE_H_

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

class UntypedProjection : public VirtualDestructor {
 public:
  virtual int32_t num_blocks() const { return 1; }

  virtual int32_t projected_dimensionality() const { return -1; }

  virtual StatusOr<shared_ptr<const TypedDataset<float>>> GetDirections() const;
};

template <typename T>
class Projection : public UntypedProjection {
 public:
  virtual Status ProjectInput(const DatapointPtr<T>& input,
                              Datapoint<double>* projected) const = 0;
  virtual Status ProjectInput(const DatapointPtr<T>& input,
                              Datapoint<float>* projected) const = 0;
};

#define DEFINE_PROJECT_INPUT_OVERRIDES(Class)                         \
  template <typename T>                                               \
  Status Class<T>::ProjectInput(const DatapointPtr<T>& input,         \
                                Datapoint<double>* projected) const { \
    return ProjectInputImpl<double>(input, projected);                \
  }                                                                   \
  template <typename T>                                               \
  Status Class<T>::ProjectInput(const DatapointPtr<T>& input,         \
                                Datapoint<float>* projected) const {  \
    return ProjectInputImpl<float>(input, projected);                 \
  }

SCANN_INSTANTIATE_TYPED_CLASS(extern, Projection);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
