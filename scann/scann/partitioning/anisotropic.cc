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

#include "scann/partitioning/anisotropic.h"

#include <utility>

namespace research_scann {

AvqAccumulator::AvqAccumulator(size_t dimensionality, float eta)
    : dimensionality_(dimensionality),
      eta_(eta),
      xtx_matrix_(std::isnan(eta)
                      ? EMatrixXf::Zero(0, 0)
                      : EMatrixXf::Zero(dimensionality, dimensionality)),
      weighted_vector_sum_(EVectorXf::Zero(dimensionality)),
      total_weight_(0) {}

AvqAccumulator& AvqAccumulator::AddVectors(ConstSpan<float> vecs) {
  if (IsEmpty(vecs)) return *this;

  const size_t num_vectors = vecs.size() / dimensionality_;
  QCHECK_EQ(vecs.size(), num_vectors * dimensionality_);

  auto partition_matrix =
      EMatrixXfMap(vecs.data(), num_vectors, dimensionality_);

  if (std::isnan(eta_)) {
    weighted_vector_sum_ += partition_matrix.colwise().sum().matrix();
    total_weight_ += num_vectors;
    return *this;
  }

  EVectorXf norms = partition_matrix.rowwise().stableNorm();

  const float fillzero = (eta_ == 1) ? 1 : 0;

  EVectorXf weighting = norms.array().pow(eta_ - 1);
  weighting = (norms.array() == 0).select(fillzero, weighting);

  EVectorXf sqrt_xxt_weighting = norms.array().pow(0.5 * (eta_ - 3));
  sqrt_xxt_weighting =
      (norms.array() < 1e-20).select(fillzero, sqrt_xxt_weighting);

  auto X = (partition_matrix.array().colwise() * sqrt_xxt_weighting.array())
               .matrix();
  xtx_matrix_.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());

  weighted_vector_sum_ +=
      (partition_matrix.array().colwise() * weighting.array())
          .colwise()
          .sum()
          .matrix();

  total_weight_ += weighting.sum();

  return *this;
}

EVectorXf AvqAccumulator::GetCenter() {
  if (total_weight_ == 0) return EVectorXf::Zero(dimensionality_);

  if (std::isnan(eta_)) {
    return weighted_vector_sum_ * 1.0 / total_weight_;
  }

  EMatrixXf to_invert =
      total_weight_ * EMatrixXf::Identity(dimensionality_, dimensionality_) +
      (eta_ - 1) * xtx_matrix_;
  Eigen::LDLT<Eigen::Ref<EMatrixXf>, Eigen::Lower> inverter(to_invert);

  return eta_ * inverter.solve(weighted_vector_sum_);
}

std::pair<double, double> ComputeRescaleFraction(
    ConstSpan<float> partition_center, ConstSpan<float> partition_data) {
  double centroid_sq_norm = 0;
  for (float f : partition_center) centroid_sq_norm += f * f;

  size_t n_dims = partition_center.size();
  vector<double> dataset_sum(n_dims);
  int cur_dim = 0;
  for (float f : partition_data) {
    dataset_sum[cur_dim] += f;
    if (++cur_dim == n_dims) cur_dim = 0;
  }

  double centroid_sum_dp = 0;
  for (int i = 0; i < dataset_sum.size(); i++)
    centroid_sum_dp += dataset_sum[i] * partition_center[i];

  size_t n_points = partition_data.size() / n_dims;
  return std::make_pair(centroid_sum_dp, n_points * centroid_sq_norm);
}

}  // namespace research_scann
