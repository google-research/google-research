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

#include "scann/partitioning/anisotropic.h"

#include <utility>

namespace research_scann {

AccumulatedAVQData ReducePartition(ConstSpan<float> partition,
                                   size_t dimensionality, float eta) {
  DCHECK_EQ(partition.size() % dimensionality, 0)
      << "Number of floats in partition isn't divisible by dimensionality: "
      << partition.size() << " vs " << dimensionality;

  AccumulatedAVQData result;
  result.eta = eta;
  const size_t num_dps = partition.size() / dimensionality;

  if (num_dps == 0) {
    result.xtx = EMatrixXf::Zero(dimensionality, dimensionality);
    result.norm_weighted_dp_sum = EVectorXf::Zero(dimensionality);
    result.dp_norm_sum = 0;
  } else {
    const float fillzero = (eta == 1) ? 1 : 0;

    auto partition_matrix =
        EMatrixXfMap(partition.data(), num_dps, dimensionality);
    EVectorXf norms = partition_matrix.rowwise().stableNorm();

    EVectorXf norms_pow = norms.array().pow(0.5 * (eta - 3));
    norms_pow = (norms.array() < 1e-20).select(fillzero, norms_pow);

    auto X = (partition_matrix.array().colwise() * norms_pow.array()).matrix();
    result.xtx.setZero(dimensionality, dimensionality);
    result.xtx.selfadjointView<Eigen::Lower>().rankUpdate(X.transpose());

    EVectorXf norms_eta1 = norms.array().pow(eta - 1);
    norms_eta1 = (norms.array() == 0).select(fillzero, norms_eta1);
    result.norm_weighted_dp_sum =
        (partition_matrix.array().colwise() * norms_eta1.array())
            .colwise()
            .sum();
    result.dp_norm_sum = norms_eta1.sum();
  }

  return result;
}

EVectorXf ComputeAVQPartition(const AccumulatedAVQData& avq_data) {
  const int d = avq_data.norm_weighted_dp_sum.size();

  if (avq_data.dp_norm_sum == 0) return EVectorXf::Zero(d);

  EMatrixXf to_invert = avq_data.dp_norm_sum * EMatrixXf::Identity(d, d) +
                        (avq_data.eta - 1) * avq_data.xtx;
  Eigen::LDLT<Eigen::Ref<EMatrixXf>, Eigen::Lower> inverter(to_invert);

  return avq_data.eta * inverter.solve(avq_data.norm_weighted_dp_sum);
}

EVectorXf ComputeAVQPartition(ConstSpan<float> partition, size_t dimensionality,
                              float eta) {
  return ComputeAVQPartition(ReducePartition(partition, dimensionality, eta));
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
