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



#ifndef SCANN_UTILS_PCA_UTILS_H_
#define SCANN_UTILS_PCA_UTILS_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"

namespace research_scann {

class PcaUtils {
 public:
  static void ComputePca(bool use_propack_if_available, const Dataset& data,
                         const int32_t num_eigenvectors,
                         const bool build_covariance,
                         vector<Datapoint<float>>* eigenvectors,
                         vector<float>* eigenvalues);

  static void ComputePcaWithSignificanceThreshold(
      bool use_propack_if_available, const Dataset& data,
      float significance_threshold, float truncation_threshold,
      bool build_covariance, vector<Datapoint<float>>* eigenvectors,
      vector<float>* eigenvalues);

 private:
  static void ComputePcaWithEigen(const Dataset& data,
                                  const int32_t num_eigenvectors,
                                  bool build_covariance,
                                  vector<Datapoint<float>>* eigenvectors,
                                  vector<float>* eigenvalues);

  static constexpr uint32_t max_dims_ = 20000;
};

}  // namespace research_scann

#endif
