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



#include "scann/projection/pca_projection.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric.h"
#include "scann/distance_measures/one_to_many/one_to_many_symmetric.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/projection/random_orthogonal_projection.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
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
void PcaProjection<T>::Create(const Dataset& data, const bool build_covariance,
                              ThreadPool* parallelization_pool) {
  vector<float> eigen_vals;

  vector<Datapoint<float>> pca_vecs;
  PcaUtils::ComputePca(false, data, projected_dims_, build_covariance,
                       &pca_vecs, &eigen_vals, parallelization_pool);

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
                              ThreadPool* parallelization_pool) {
  vector<float> eigen_vals;
  vector<Datapoint<float>> pca_vecs;
  PcaUtils::ComputePcaWithSignificanceThreshold(
      false, data, pca_significance_threshold, pca_truncation_threshold,
      build_covariance, &pca_vecs, &eigen_vals, parallelization_pool);

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
void PcaProjection<T>::Create(
    std::shared_ptr<DenseDataset<float>> eigenvectors) {
  pca_vecs_ = eigenvectors;
}

template <typename T>
Status PcaProjection<T>::Create(
    const SerializedProjection& serialized_projection) {
  if (serialized_projection.rotation_vec_size() == 0) {
    return InvalidArgumentError(
        "Serialized projection rotation matrix is empty in "
        "PcaProjection::Create.");
  }
  auto pca_vecs = std::make_unique<DenseDataset<float>>();
  pca_vecs->set_dimensionality(
      serialized_projection.rotation_vec(0).feature_value_float_size());
  pca_vecs->Reserve(serialized_projection.rotation_vec_size());
  for (const auto& gfv : serialized_projection.rotation_vec()) {
    SCANN_RETURN_IF_ERROR(pca_vecs->Append(gfv, ""));
  }
  pca_vecs_ = std::move(pca_vecs);
  return OkStatus();
}

template <typename T>
void PcaProjection<T>::RandomRotateProjectionMatrix() {
  if (pca_vecs_ == nullptr) {
    LOG(WARNING) << "No PCA vectors to rotate.";
    return;
  }
  DCHECK_EQ(pca_vecs_->size(), projected_dims_);
  DCHECK_EQ(pca_vecs_->dimensionality(), input_dims_);
  RandomOrthogonalProjection<float> ortho(projected_dims_, projected_dims_, 42);
  ortho.Create();
  const shared_ptr<const TypedDataset<float>> ortho_vecs =
      ortho.GetDirections().value();
  DCHECK(ortho_vecs != nullptr);
  DCHECK_EQ(ortho_vecs->size(), projected_dims_);
  DCHECK_EQ(ortho_vecs->dimensionality(), projected_dims_);
  vector<float> rotated_matrix(static_cast<size_t>(input_dims_) *
                               static_cast<size_t>(projected_dims_));
  vector<float> col_vec(projected_dims_);

  for (size_t col_idx : Seq(input_dims_)) {
    for (size_t row_idx : Seq(projected_dims_)) {
      col_vec[row_idx] = (*pca_vecs_)[row_idx].values()[col_idx];
    }
    for (size_t row_idx : Seq(projected_dims_)) {
      rotated_matrix[row_idx * input_dims_ + col_idx] =
          DotProduct(MakeDatapointPtr(col_vec), (*ortho_vecs)[row_idx]);
    }
  }
  pca_vecs_ = std::make_shared<DenseDataset<float>>(std::move(rotated_matrix),
                                                    projected_dims_);
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
  if constexpr (std::is_same_v<T, float>) {
    auto float32_path = [&] {
      DenseDotProductDistanceOneToMany<T, FloatT>(
          input, pca_vecs, MakeMutableSpan(*projected->mutable_values()));
    };

    if constexpr (std::is_same_v<FloatT, float>) {
      if (fixed8_pca_vecs_ != nullptr) {
        vector<float> inv_mult_input(input.dimensionality());
        for (size_t i : Seq(input_dims_)) {
          inv_mult_input[i] = input.values()[i] * inv_fixed8_multipliers_[i];
        }
        DenseDotProductDistanceOneToManyInt8Float(
            MakeDatapointPtr(inv_mult_input), *fixed8_pca_vecs_,
            MakeMutableSpan(*projected->mutable_values()));
      } else if (bfloat16_pca_vecs_ != nullptr) {
        DenseDotProductDistanceOneToManyBf16Float(
            input, *bfloat16_pca_vecs_,
            MakeMutableSpan(*projected->mutable_values()));
      } else {
        float32_path();
      }
    } else {
      float32_path();
    }

    for (FloatT& val : *projected->mutable_values()) {
      val = -val;
    }
  } else {
    for (size_t i = 0; i < projected_dims_; ++i) {
      projected->mutable_values()->at(i) = DotProduct(input, pca_vecs[i]);
    }
  }

  return OkStatus();
}

template <typename T>
StatusOr<shared_ptr<const TypedDataset<float>>>
PcaProjection<T>::GetDirections() const {
  return std::dynamic_pointer_cast<const TypedDataset<float>>(pca_vecs_);
}

template <typename T>
Status PcaProjection<T>::CompressToBFloat16() {
  if (pca_vecs_ == nullptr) {
    return FailedPreconditionError(
        "Can't compress PCA vectors to bfloat16 because PCA vectors are not "
        "initialized.");
  }
  if (bfloat16_pca_vecs_ != nullptr) {
    return OkStatus();
  }

  bfloat16_pca_vecs_ = std::make_unique<DenseDataset<int16_t>>(
      Bfloat16QuantizeFloatDataset(*pca_vecs_));
  return OkStatus();
}

template <typename T>
Status PcaProjection<T>::CompressToFixed8() {
  if (pca_vecs_ == nullptr) {
    return FailedPreconditionError(
        "Can't compress PCA vectors to bfloat16 because PCA vectors are not "
        "initialized.");
  }
  if (fixed8_pca_vecs_ != nullptr) {
    return OkStatus();
  }

  auto sq = ScalarQuantizeFloatDataset(*pca_vecs_);
  fixed8_pca_vecs_ =
      std::make_unique<DenseDataset<int8_t>>(std::move(sq.quantized_dataset));
  inv_fixed8_multipliers_ = std::move(sq.inverse_multiplier_by_dimension);
  return OkStatus();
}

template <typename T>
std::optional<SerializedProjection> PcaProjection<T>::SerializeToProto() const {
  SerializedProjection result;
  result.mutable_rotation_vec()->Reserve(pca_vecs_->size());
  for (DatapointPtr<float> eigenvector : *pca_vecs_) {
    *result.add_rotation_vec() = eigenvector.ToGfv();
  }
  return result;
}

DEFINE_PROJECT_INPUT_OVERRIDES(PcaProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, PcaProjection);

}  // namespace research_scann
