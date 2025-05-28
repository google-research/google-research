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

template <typename T>
void ComputePcaDenseWrapper(const Dataset& data, const int32_t num_eigenvectors,
                            vector<Datapoint<float>>* eigenvectors,
                            vector<float>* eigenvalues, ThreadPool* pool) {
  auto view =
      DefaultDenseDatasetView<T>(*dynamic_cast<const DenseDataset<T>*>(&data));
  ComputePcaDense(view, num_eigenvectors, eigenvectors, eigenvalues, pool);
}

}  // namespace

void PostprocessPcaToSignificance(const float significance_threshold,
                                  const float truncation_threshold,
                                  vector<Datapoint<float>>* eigenvectors,
                                  vector<float>* eigenvalues) {
  DCHECK_LE(significance_threshold, 1.0);
  DCHECK_EQ(eigenvectors->size(), eigenvalues->size());
  DCHECK_GT(eigenvectors->size(), 0);
  ZipSortBranchOptimized(std::greater<float>(), eigenvalues->begin(),
                         eigenvalues->end(), eigenvectors->begin(),
                         eigenvectors->end());

  const float ev_sum =
      std::accumulate(eigenvalues->begin(), eigenvalues->end(), 0.0f);
  float sum = 0.0f;
  DimensionIndex truncate = eigenvalues->size();
  for (DimensionIndex i = 0; i < eigenvalues->size(); ++i) {
    sum += eigenvalues->at(i);
    if (sum > significance_threshold * ev_sum) {
      truncate = i + 1;
      break;
    }
  }

  if (truncate < eigenvalues->size() * truncation_threshold) {
    eigenvectors->resize(truncate);
    eigenvalues->resize(truncate);
  }
}

void PcaUtils::ComputePca(bool use_propack_if_available, const Dataset& data,
                          const int32_t num_eigenvectors,
                          const bool build_covariance,
                          vector<Datapoint<float>>* eigenvectors,
                          vector<float>* eigenvalues, ThreadPool* pool) {
  if (!use_propack_if_available && build_covariance && data.IsDense()) {
    SCANN_CALL_FUNCTION_BY_TAG(data.TypeTag(), ComputePcaDenseWrapper, data,
                               num_eigenvectors, eigenvectors, eigenvalues,
                               pool);
    return;
  }
  LOG(FATAL) << "Unsupported.";
}

void PcaUtils::ComputePcaWithSignificanceThreshold(
    bool use_propack_if_available, const Dataset& data,
    const float significance_threshold, const float truncation_threshold,
    const bool build_covariance, vector<Datapoint<float>>* eigenvectors,
    vector<float>* eigenvalues, ThreadPool* pool) {
  PcaUtils::ComputePca(use_propack_if_available, data, data.dimensionality(),
                       build_covariance, eigenvectors, eigenvalues, pool);
  PostprocessPcaToSignificance(significance_threshold, truncation_threshold,
                               eigenvectors, eigenvalues);
}

}  // namespace research_scann
