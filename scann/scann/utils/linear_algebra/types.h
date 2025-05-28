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

#ifndef SCANN_UTILS_LINEAR_ALGEBRA_TYPES_H_
#define SCANN_UTILS_LINEAR_ALGEBRA_TYPES_H_

#include "Eigen/Dense"

namespace research_scann {

using EMatrixXf = Eigen::MatrixXf;
using EMatrixXd = Eigen::MatrixXd;
using EVectorXf = Eigen::VectorXf;
using EVectorXd = Eigen::VectorXd;

using EMatrixXfMap =
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>;
using EMatrixXdMap =
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>;
using EVectorXfMap = Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>>;
using EVectorXdMap = Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>>;

}  // namespace research_scann

#endif
