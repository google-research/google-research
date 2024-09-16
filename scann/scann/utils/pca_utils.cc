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

#include "scann/utils/pca_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>

#include "Eigen/Core"
#include "Eigen/SVD"
#include "absl/log/check.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

namespace {

template <typename Matrix>
void BuildCenteredMatrix(const Dataset& data, Matrix* centered_data) {
  Datapoint<double> mean_vec;
  CHECK_OK(data.MeanByDimension(&mean_vec));
  DimensionIndex dims = data.dimensionality();
  centered_data->resize(dims, data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    Datapoint<double> dp1, diff_vec;
    data.GetDatapoint(i, &dp1);
    PointDiff(dp1.ToPtr(), mean_vec.ToPtr(), &diff_vec);
    for (size_t j = 0; j < dims; ++j) {
      (*centered_data)(j, i) = diff_vec.values()[j];
    }
  }
}

}  // namespace

void PcaUtils::ComputePca(bool use_propack_if_available, const Dataset& data,
                          const int32_t num_eigenvectors,
                          const bool build_covariance,
                          vector<Datapoint<float>>* eigenvectors,
                          vector<float>* eigenvalues) {
  ComputePcaWithEigen(data, num_eigenvectors, build_covariance, eigenvectors,
                      eigenvalues);
}

void PcaUtils::ComputePcaWithSignificanceThreshold(
    bool use_propack_if_available, const Dataset& data,
    const float significance_threshold, const float truncation_threshold,
    const bool build_covariance, vector<Datapoint<float>>* eigenvectors,
    vector<float>* eigenvalues) {
  DCHECK_LE(significance_threshold, 1.0);
  const DimensionIndex dim = data.dimensionality();
  PcaUtils::ComputePca(false, data, dim, build_covariance, eigenvectors,
                       eigenvalues);
  ZipSortBranchOptimized(std::greater<float>(), eigenvalues->begin(),
                         eigenvalues->end(), eigenvectors->begin(),
                         eigenvectors->end());

  const float ev_sum =
      std::accumulate(eigenvalues->begin(), eigenvalues->end(), 0.0f);
  float sum = 0.0f;
  DimensionIndex truncate = dim;
  for (DimensionIndex i = 0; i < eigenvalues->size(); ++i) {
    sum += eigenvalues->at(i);
    if (sum > significance_threshold * ev_sum) {
      truncate = i + 1;
      break;
    }
  }

  if (truncate < dim * truncation_threshold) {
    eigenvectors->resize(truncate);
    eigenvalues->resize(truncate);
  }
}

void PcaUtils::ComputePcaWithEigen(const Dataset& data,
                                   const int32_t num_eigenvectors,
                                   bool build_covariance,
                                   vector<Datapoint<float>>* eigenvectors,
                                   vector<float>* eigenvalues) {
  CHECK_GT(data.size(), 0) << "The data set is empty";
  CHECK_GT(num_eigenvectors, 0)
      << "The number of eigenvectors to return should be more than zero";
  CHECK_LE(num_eigenvectors, data.dimensionality())
      << "Cannot return more eigenvectors that data dimensionality";
  CHECK_LE(data.dimensionality(), max_dims_)
      << "Cannot process more than " << max_dims_ << "dimensional data";

  CHECK(eigenvectors != nullptr);
  CHECK(eigenvalues != nullptr);
  eigenvectors->resize(num_eigenvectors);
  eigenvalues->resize(num_eigenvectors);
  DimensionIndex dims = data.dimensionality();

  build_covariance = false;
  if (build_covariance) {
  } else {
    Eigen::MatrixXd centered_data;
    BuildCenteredMatrix(data, &centered_data);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered_data, Eigen::ComputeThinU);

    CHECK_EQ(svd.matrixU().rows(), dims);
    int32_t min_dim = dims < data.size() ? dims : data.size();
    CHECK_EQ(svd.matrixU().cols(), min_dim);
    CHECK_EQ(svd.singularValues().rows(), min_dim);

    for (size_t i = 0; i < num_eigenvectors; ++i) {
      auto evec = (*eigenvectors)[i].mutable_values();
      evec->resize(dims);
      for (size_t j = 0; j < dims; ++j) {
        (*evec)[j] = static_cast<float>(svd.matrixU()(j, i));
      }
      double singular_val = svd.singularValues()(i);
      (*eigenvalues)[i] = static_cast<float>(singular_val * singular_val);
    }
  }
}

}  // namespace research_scann
