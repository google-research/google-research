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



#ifndef SCANN_UTILS_LINEAR_ALGEBRA_EIGEN_UTILS_H_
#define SCANN_UTILS_LINEAR_ALGEBRA_EIGEN_UTILS_H_

#include "Eigen/Dense"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/common.h"
#include "scann/utils/linear_algebra/types.h"

namespace research_scann {

EMatrixXf ToEigenMatrix(const DenseDataset<float>& dataset);
EVectorXf ToEigenVector(const DatapointPtr<float>& dptr);
EVectorXf ToEigenVector(ConstSpan<float> span);
EMatrixXd ToEigenMatrix(const DenseDataset<double>& dataset);
EVectorXd ToEigenVector(const DatapointPtr<double>& dptr);
EVectorXd ToEigenVector(ConstSpan<double> span);

EMatrixXfMap ToEigenMap(const DenseDataset<float>& dataset);
EVectorXfMap ToEigenMap(const DatapointPtr<float>& dptr);
EVectorXfMap ToEigenMap(ConstSpan<float> span);
EMatrixXdMap ToEigenMap(const DenseDataset<double>& dataset);
EVectorXdMap ToEigenMap(const DatapointPtr<double>& dptr);
EVectorXdMap ToEigenMap(ConstSpan<double> span);

absl::Status DSFromEigenMatrix(const EMatrixXf& matrix,
                               DenseDataset<float>* dataset);
absl::Status DSFromEigenMatrix(const EMatrixXd& matrix,
                               DenseDataset<double>* dataset);

absl::Status DPFromEigenVector(const EVectorXf& vec, Datapoint<float>* dp);
absl::Status DPFromEigenVector(const EVectorXd& vec, Datapoint<double>* dp);

absl::Status FillSpanFromEigenVector(const EVectorXf& vec,
                                     absl::Span<float> dp);
absl::Status FillSpanFromEigenVector(const EVectorXd& vec,
                                     absl::Span<double> dp);

}  // namespace research_scann

#endif
