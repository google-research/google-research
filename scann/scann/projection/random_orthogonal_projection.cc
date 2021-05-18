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



#include "scann/projection/random_orthogonal_projection.h"

#include <cstdint>

#include "Eigen/Core"
#include "Eigen/QR"
#include "absl/random/random.h"
#include "scann/utils/datapoint_utils.h"

namespace research_scann {

template <typename T>
RandomOrthogonalProjection<T>::RandomOrthogonalProjection(
    const int32_t input_dims, const int32_t projected_dims, const int32_t seed)
    : input_dims_(input_dims), projected_dims_(projected_dims), seed_(seed) {
  CHECK_GT(input_dims_, 0) << "Input dimensionality must be > 0";
  CHECK_GT(projected_dims_, 0) << "Projected dimensionality must be > 0";

  CHECK_GE(input_dims_, projected_dims_)
      << "The projected dimensions cannot be larger than input dimensions";
}

template <typename T>
void RandomOrthogonalProjection<T>::Create() {
  random_.reset(new MTRandom(seed_));

  Eigen::MatrixXf input_matrix(input_dims_, projected_dims_);

  for (size_t i = 0; i < projected_dims_; ++i) {
    for (size_t j = 0; j < input_dims_; ++j) {
      input_matrix(j, i) = absl::Gaussian<double>(*random_);
    }
  }

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(input_matrix);
  Eigen::MatrixXf Q = qr.householderQ();

  auto random_rotation_matrix = std::make_shared<DenseDataset<float>>();
  for (size_t i = 0; i < projected_dims_; ++i) {
    vector<float> current;
    current.resize(input_dims_);
    for (size_t j = 0; j < input_dims_; j++) {
      current[j] = Q(j, i);
    }
    random_rotation_matrix->AppendOrDie(MakeDatapointPtr(current), "");
  }
  random_rotation_matrix_ = random_rotation_matrix;
}

template <typename T>
template <typename FloatT>
Status RandomOrthogonalProjection<T>::ProjectInputImpl(
    const DatapointPtr<T>& input, Datapoint<FloatT>* projected) const {
  CHECK(projected != nullptr);
  projected->clear();
  projected->mutable_values()->resize(projected_dims_);

  if (!random_rotation_matrix_) {
    return FailedPreconditionError(
        "Create the random orthogonal matrix first.");
  }

  const DenseDataset<float>& random_rotation_matrix = *random_rotation_matrix_;
  CHECK_EQ(random_rotation_matrix.dimensionality(), input.dimensionality());

  for (size_t i = 0; i < projected_dims_; ++i) {
    projected->mutable_values()->at(i) =
        static_cast<FloatT>(DotProduct(input, random_rotation_matrix[i]));
  }

  return OkStatus();
}

DEFINE_PROJECT_INPUT_OVERRIDES(RandomOrthogonalProjection);
SCANN_INSTANTIATE_TYPED_CLASS(, RandomOrthogonalProjection);

}  // namespace research_scann
