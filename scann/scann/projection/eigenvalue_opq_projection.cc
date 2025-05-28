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

#include "scann/projection/eigenvalue_opq_projection.h"

#include <cstdint>
#include <memory>

#include "scann/utils/datapoint_utils.h"
#include "scann/utils/pca_utils.h"

namespace research_scann {

template <typename T>
EigenvalueOpqProjection<T>::EigenvalueOpqProjection(const int32_t input_dims)
    : input_dims_(input_dims) {
  CHECK_GT(input_dims_, 0) << "Input dimensionality must be > 0";
}

namespace {

struct EigenvalueGroup {
  std::vector<uint32_t> eigen_indices;
  float total_eigen_value = 0.0;

  bool operator<(const EigenvalueGroup& other) const {
    return total_eigen_value < other.total_eigen_value;
  }

  bool operator>(const EigenvalueGroup& other) const {
    return total_eigen_value > other.total_eigen_value;
  }
};

vector<EigenvalueGroup> GroupEigenvalues(ConstSpan<float> eigen_vals,
                                         const int32_t num_groups) {
  vector<EigenvalueGroup> groups(num_groups);
  for (uint32_t eigenvalue_idx : IndicesOf(eigen_vals)) {
    const float eigenvalue = eigen_vals[eigenvalue_idx];
    std::pop_heap(groups.begin(), groups.end(),
                  std::greater<EigenvalueGroup>());
    groups.back().eigen_indices.push_back(eigenvalue_idx);
    groups.back().total_eigen_value += eigenvalue;
    std::push_heap(groups.begin(), groups.end(),
                   std::greater<EigenvalueGroup>());
  }

  std::sort(groups.begin(), groups.end(), std::greater<EigenvalueGroup>());
  return groups;
}

}  // namespace

template <typename T>
void EigenvalueOpqProjection<T>::Create(const Dataset& data,
                                        const uint32_t num_blocks,
                                        const bool build_covariance,
                                        ThreadPool* pool) {
  vector<Datapoint<float>> rotation_matrix;
  vector<float> eigen_vals;
  PcaUtils::ComputePca(false, data, input_dims_, build_covariance,
                       &rotation_matrix, &eigen_vals, pool);
  vector<EigenvalueGroup> eigenvalue_groups =
      GroupEigenvalues(eigen_vals, num_blocks);

  auto rotation_matrix_ds = std::make_unique<DenseDataset<float>>();
  rotation_matrix_ds->set_dimensionality(input_dims_);
  rotation_matrix_ds->Reserve(rotation_matrix.size());
  chunk_sizes_.resize(eigenvalue_groups.size());
  eigenvalue_sums_.resize(eigenvalue_groups.size());

  for (auto [i, group] : Enumerate(eigenvalue_groups)) {
    VLOG(1) << "Eigenvalue group " << i << " of " << num_blocks << ": "
            << group.eigen_indices.size() << ", " << group.total_eigen_value;
    for (uint32_t eigenvalue_idx : group.eigen_indices) {
      rotation_matrix_ds->AppendOrDie(rotation_matrix[eigenvalue_idx].ToPtr(),
                                      "");
    }
    chunk_sizes_[i] = group.eigen_indices.size();
    eigenvalue_sums_[i] = group.total_eigen_value;
  }

  rotation_matrix_ = std::move(rotation_matrix_ds);
}

template <typename T>
Status EigenvalueOpqProjection<T>::Create(
    const SerializedProjection& serialized_projection) {
  if (serialized_projection.rotation_vec_size() == 0) {
    return InvalidArgumentError(
        "Serialized projection rotation matrix is empty in "
        "EigenvalueOpqProjection::Create.");
  }
  auto rotation_matrix = std::make_unique<DenseDataset<float>>();
  rotation_matrix->set_dimensionality(
      serialized_projection.rotation_vec(0).feature_value_float_size());
  rotation_matrix->Reserve(serialized_projection.rotation_vec_size());
  for (const auto& gfv : serialized_projection.rotation_vec()) {
    SCANN_RETURN_IF_ERROR(rotation_matrix->Append(gfv, ""));
  }
  rotation_matrix_ = std::move(rotation_matrix);
  chunk_sizes_ =
      vector<int32_t>(serialized_projection.variable_dims_per_block().begin(),
                      serialized_projection.variable_dims_per_block().end());
  eigenvalue_sums_ =
      vector<float>(serialized_projection.per_block_eigenvalue_sums().begin(),
                    serialized_projection.per_block_eigenvalue_sums().end());
  return OkStatus();
}

template <typename T>
std::optional<SerializedProjection>
EigenvalueOpqProjection<T>::SerializeToProto() const {
  SerializedProjection result;
  result.mutable_variable_dims_per_block()->Assign(chunk_sizes_.begin(),
                                                   chunk_sizes_.end());
  result.mutable_per_block_eigenvalue_sums()->Assign(eigenvalue_sums_.begin(),
                                                     eigenvalue_sums_.end());
  result.mutable_rotation_vec()->Reserve(rotation_matrix_->size());
  for (DatapointPtr<float> vec : *rotation_matrix_) {
    *result.add_rotation_vec() = vec.ToGfv();
  }
  return result;
}

template <typename T>
template <typename FloatT>
Status EigenvalueOpqProjection<T>::ProjectInputImpl(
    const DatapointPtr<T>& input, Datapoint<FloatT>* projected) const {
  CHECK(projected != nullptr);
  projected->clear();
  projected->mutable_values()->resize(input_dims_);

  if (!rotation_matrix_) {
    return FailedPreconditionError("First compute the rotation matrix.");
  }

  const auto& rotation_matrix = *rotation_matrix_;
  for (size_t i : Seq(input_dims_)) {
    projected->mutable_values()->at(i) = DotProduct(input, rotation_matrix[i]);
  }

  return OkStatus();
}

template <typename T>
StatusOr<shared_ptr<const TypedDataset<float>>>
EigenvalueOpqProjection<T>::GetDirections() const {
  return std::dynamic_pointer_cast<const TypedDataset<float>>(rotation_matrix_);
}

DEFINE_PROJECT_INPUT_OVERRIDES(EigenvalueOpqProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, EigenvalueOpqProjection);

}  // namespace research_scann
