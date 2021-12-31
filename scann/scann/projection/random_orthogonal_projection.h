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



#ifndef SCANN_PROJECTION_RANDOM_ORTHOGONAL_PROJECTION_H_
#define SCANN_PROJECTION_RANDOM_ORTHOGONAL_PROJECTION_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/projection/projection_base.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class RandomOrthogonalProjection : public Projection<T> {
 public:
  RandomOrthogonalProjection(const int32_t input_dims,
                             const int32_t projected_dims, const int32_t seed);

  void Create();

  StatusOr<shared_ptr<const TypedDataset<float>>> GetDirections() const final {
    return std::dynamic_pointer_cast<const TypedDataset<float>>(
        random_rotation_matrix_);
  }

  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<float>* projected) const override;
  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<double>* projected) const override;

  int32_t projected_dimensionality() const override { return projected_dims_; }

 private:
  template <typename FloatT>
  Status ProjectInputImpl(const DatapointPtr<T>& input,
                          Datapoint<FloatT>* projected) const;

  int32_t input_dims_;
  int32_t projected_dims_;
  unique_ptr<MTRandom> random_;
  int32_t seed_;
  shared_ptr<const DenseDataset<float>> random_rotation_matrix_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, RandomOrthogonalProjection);

}  // namespace research_scann

#endif
