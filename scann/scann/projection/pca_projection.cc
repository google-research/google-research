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



#include "scann/projection/pca_projection.h"

#include <cstdint>
#include <memory>

#include "scann/utils/datapoint_utils.h"
#include "scann/utils/pca_utils.h"

namespace research_scann {

template <typename T>
PcaProjection<T>::PcaProjection(const int32_t input_dims,
                                const int32_t projected_dims)
    : input_dims_(input_dims), projected_dims_(projected_dims) {
  CHECK_GT(input_dims_, 0) << "Input dimensionality must be > 0";
  CHECK_GT(projected_dims_, 0) << "Projected dimensionality must be > 0";

  CHECK_GE(input_dims_, projected_dims_)
      << "The projected dimensions cannot be larger than input dimensions";
}

template <typename T>
void PcaProjection<T>::Create(const Dataset& data,
                              const bool build_covariance) {
  Create(data, build_covariance, false);
}

template <typename T>
void PcaProjection<T>::Create(const Dataset& data, const bool build_covariance,
                              const bool use_propack) {
  vector<float> eigen_vals;

  vector<Datapoint<float>> pca_vecs;
  PcaUtils::ComputePca(use_propack, data, projected_dims_, build_covariance,
                       &pca_vecs, &eigen_vals);

  auto pca_vec_dataset = std::make_shared<DenseDataset<float>>();
  for (auto& vec : pca_vecs) {
    pca_vec_dataset->AppendOrDie(vec.ToPtr(), "");
    FreeBackingStorage(&vec);
  }

  pca_vecs_ = pca_vec_dataset;
}

template <typename T>
void PcaProjection<T>::Create(const Dataset& data,
                              const float pca_significance_threshold,
                              const float pca_truncation_threshold,
                              const bool build_covariance,
                              const bool use_propack) {
  vector<float> eigen_vals;
  vector<Datapoint<float>> pca_vecs;
  PcaUtils::ComputePcaWithSignificanceThreshold(
      use_propack, data, pca_significance_threshold, pca_truncation_threshold,
      build_covariance, &pca_vecs, &eigen_vals);

  auto pca_vec_dataset = std::make_shared<DenseDataset<float>>();
  for (auto& vec : pca_vecs) {
    pca_vec_dataset->AppendOrDie(vec.ToPtr(), "");
    FreeBackingStorage(&vec);
  }
  pca_vecs_ = pca_vec_dataset;
  projected_dims_ = pca_vecs.size();
}

template <typename T>
void PcaProjection<T>::Create(DenseDataset<float> eigenvectors) {
  pca_vecs_ = std::make_shared<DenseDataset<float>>(std::move(eigenvectors));
}

template <typename T>
template <typename FloatT>
Status PcaProjection<T>::ProjectInputImpl(const DatapointPtr<T>& input,
                                          Datapoint<FloatT>* projected) const {
  CHECK(projected != nullptr);
  projected->clear();
  projected->mutable_values()->resize(projected_dims_);

  if (!pca_vecs_) {
    return FailedPreconditionError("First compute the pca directions.");
  }

  const auto& pca_vecs = *pca_vecs_;
  for (size_t i = 0; i < projected_dims_; ++i) {
    projected->mutable_values()->at(i) = DotProduct(input, pca_vecs[i]);
  }

  return OkStatus();
}

template <typename T>
StatusOr<shared_ptr<const TypedDataset<float>>>
PcaProjection<T>::GetDirections() const {
  return std::dynamic_pointer_cast<const TypedDataset<float>>(pca_vecs_);
}

DEFINE_PROJECT_INPUT_OVERRIDES(PcaProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, PcaProjection);

}  // namespace research_scann
