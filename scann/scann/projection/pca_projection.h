// Copyright 2024 The Google Research Authors.
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



#ifndef SCANN_PROJECTION_PCA_PROJECTION_H_
#define SCANN_PROJECTION_PCA_PROJECTION_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/projection/projection_base.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class PcaProjection : public Projection<T> {
 public:
  PcaProjection(const int32_t input_dims, const int32_t projected_dims);

  void Create(const Dataset& data, const bool build_covariance);

  void Create(const Dataset& data, const bool build_covariance,
              const bool use_propack);

  void Create(const Dataset& data, float pca_significance_threshold,
              float pca_truncation_threshold, bool build_covariance = true,
              bool use_propack = false);

  void Create(DenseDataset<float> eigenvectors);

  StatusOr<shared_ptr<const TypedDataset<float>>> GetDirections()
      const override;

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
  shared_ptr<const DenseDataset<float>> pca_vecs_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, PcaProjection);

}  // namespace research_scann

#endif
