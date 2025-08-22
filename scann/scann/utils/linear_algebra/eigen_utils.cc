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

#include "scann/utils/linear_algebra/eigen_utils.h"

#include <cstdint>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/status/status.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/linear_algebra/types.h"

namespace research_scann {

using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::Vector;

template <typename T>
Map<const Matrix<T, Dynamic, Dynamic, RowMajor>> ToEigenMapHelper(
    const DenseDataset<T>& dataset) {
  int32_t rows = dataset.size();
  int32_t cols = dataset.dimensionality();
  return {dataset.data().data(), rows, cols};
}

template <typename T>
Map<const Vector<T, Dynamic>> ToEigenMapHelper(const DatapointPtr<T>& dptr) {
  int32_t dim = dptr.nonzero_entries();
  return {dptr.values(), dim};
}

template <typename T>
Map<const Vector<T, Dynamic>> ToEigenMapHelper(ConstSpan<T> span) {
  int32_t dim = span.size();
  return {span.data(), dim};
}

template <typename T>
absl::Status DPFromEigenVectorHelper(const Vector<T, Dynamic>& vec,
                                     Datapoint<T>* dp) {
  if (!dp) {
    return InvalidArgumentError("Datapoint cannot be nullptr");
  }

  Vector<T, Dynamic> v = vec.eval();
  uint32_t dim = v.size();
  dp->clear();
  dp->set_dimensionality(dim);
  auto* storage = dp->mutable_values();
  storage->resize(dim);
  memcpy(storage->data(), v.data(), dim * sizeof(T));
  return OkStatus();
}

template <typename T>
absl::Status FillSpanFromEigenVectorHelper(const Vector<T, Dynamic>& vec,
                                           absl::Span<T> dp) {
  Vector<T, Dynamic> v = vec.eval();
  uint32_t dim = v.size();
  if (dp.size() != dim) {
    return InvalidArgumentError(
        "Dimension mismatch. dp span: %d, eigen vector: %d", dp.size(), dim);
  }
  std::copy(v.begin(), v.end(), dp.begin());
  return OkStatus();
}

template <typename T>
absl::Status DSFromEigenMatrixHelper(
    const Matrix<T, Dynamic, Dynamic, ColMajor>& matrix,
    DenseDataset<T>* dataset) {
  if (!dataset) {
    return InvalidArgumentError("Dataset cannot be nullptr");
  }
  uint32_t dim = matrix.cols();
  uint32_t size = matrix.rows();
  dataset->clear();
  dataset->set_dimensionality(dim);
  dataset->Reserve(size);
  Datapoint<T> dp;

  Matrix<T, Dynamic, Dynamic, ColMajor> m = matrix.eval();
  for (auto i : Seq(size)) {
    SCANN_RETURN_IF_ERROR(DPFromEigenVectorHelper<T>(m.row(i), &dp));
    SCANN_RETURN_IF_ERROR(dataset->Append(dp.ToPtr()));
  }
  return OkStatus();
}

EMatrixXf ToEigenMatrix(const DenseDataset<float>& dataset) {
  return ToEigenMapHelper(dataset);
}
EVectorXf ToEigenVector(const DatapointPtr<float>& dptr) {
  return ToEigenMapHelper(dptr);
}
EVectorXf ToEigenVector(ConstSpan<float> span) {
  return ToEigenMapHelper(span);
}
EMatrixXd ToEigenMatrix(const DenseDataset<double>& dataset) {
  return ToEigenMapHelper(dataset);
}
EVectorXd ToEigenVector(const DatapointPtr<double>& dptr) {
  return ToEigenMapHelper(dptr);
}
EVectorXd ToEigenVector(ConstSpan<double> span) {
  return ToEigenMapHelper(span);
}

EMatrixXfMap ToEigenMap(const DenseDataset<float>& dataset) {
  return ToEigenMapHelper(dataset);
}
EVectorXfMap ToEigenMap(const DatapointPtr<float>& dptr) {
  return ToEigenMapHelper(dptr);
}
EVectorXfMap ToEigenMap(ConstSpan<float> span) {
  return ToEigenMapHelper(span);
}
EMatrixXdMap ToEigenMap(const DenseDataset<double>& dataset) {
  return ToEigenMapHelper(dataset);
}
EVectorXdMap ToEigenMap(const DatapointPtr<double>& dptr) {
  return ToEigenMapHelper(dptr);
}
EVectorXdMap ToEigenMap(ConstSpan<double> span) {
  return ToEigenMapHelper(span);
}

absl::Status DSFromEigenMatrix(const EMatrixXf& matrix,
                               DenseDataset<float>* dataset) {
  return DSFromEigenMatrixHelper(matrix, dataset);
}
absl::Status DSFromEigenMatrix(const EMatrixXd& matrix,
                               DenseDataset<double>* dataset) {
  return DSFromEigenMatrixHelper(matrix, dataset);
}
absl::Status DPFromEigenVector(const EVectorXf& vec, Datapoint<float>* dp) {
  return DPFromEigenVectorHelper(vec, dp);
}
absl::Status DPFromEigenVector(const EVectorXd& vec, Datapoint<double>* dp) {
  return DPFromEigenVectorHelper(vec, dp);
}

absl::Status FillSpanFromEigenVector(const EVectorXf& vec,
                                     absl::Span<float> dp) {
  return FillSpanFromEigenVectorHelper(vec, dp);
}
absl::Status FillSpanFromEigenVector(const EVectorXd& vec,
                                     absl::Span<double> dp) {
  return FillSpanFromEigenVectorHelper(vec, dp);
}

}  // namespace research_scann
