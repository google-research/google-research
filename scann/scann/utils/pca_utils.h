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



#ifndef SCANN_UTILS_PCA_UTILS_H_
#define SCANN_UTILS_PCA_UTILS_H_

#include <algorithm>
#include <cstdint>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"

namespace research_scann {

template <typename T>
void ComputePcaDense(DefaultDenseDatasetView<T> data, int32_t num_eigenvectors,
                     vector<Datapoint<float>>* eigenvectors,
                     vector<float>* eigenvalues, ThreadPool* pool = nullptr);

void PostprocessPcaToSignificance(float significance_threshold,
                                  float truncation_threshold,
                                  vector<Datapoint<float>>* eigenvectors,
                                  vector<float>* eigenvalues);

class PcaUtils {
 public:
  static void ComputePca(bool use_propack_if_available, const Dataset& data,
                         int32_t num_eigenvectors, bool build_covariance,
                         vector<Datapoint<float>>* eigenvectors,
                         vector<float>* eigenvalues,
                         ThreadPool* pool = nullptr);

  static void ComputePcaWithSignificanceThreshold(
      bool use_propack_if_available, const Dataset& data,
      float significance_threshold, float truncation_threshold,
      bool build_covariance, vector<Datapoint<float>>* eigenvectors,
      vector<float>* eigenvalues, ThreadPool* pool = nullptr);

 private:
  static constexpr uint32_t kMaxDims = 20000;
};

template <typename T>
void ComputePcaDense(const DefaultDenseDatasetView<T> data,
                     const int32_t num_eigenvectors,
                     vector<Datapoint<float>>* eigenvectors,
                     vector<float>* eigenvalues, ThreadPool* pool) {
  vector<double> mean(data.dimensionality());
  for (int i = 0; i < data.size(); i++) {
    for (auto [j, sum] : Enumerate(mean)) sum += data.GetPtr(i)[j];
  }
  for (double& d : mean) d /= data.size();

  const size_t num_threads = 1 + (pool ? pool->NumThreads() : 0);
  constexpr const size_t kBatchSize = 256;
  const size_t num_inner_batches = DivRoundUp(data.size(), kBatchSize);
  const size_t num_inner_batches_per_thread =
      DivRoundUp(num_inner_batches, num_threads);

  Eigen::MatrixXd covariance =
      Eigen::MatrixXd::Zero(data.dimensionality(), data.dimensionality());
  absl::Mutex covariance_mutex;

  ParallelFor<1>(
      Seq(DivRoundUp(num_inner_batches, num_inner_batches_per_thread)), pool,
      [&](size_t outer_batch_idx) {
        size_t inner_batch_idx_start =
            num_inner_batches_per_thread * outer_batch_idx;
        size_t inner_batch_idx_end =
            std::min(num_inner_batches,
                     inner_batch_idx_start + num_inner_batches_per_thread);
        Eigen::MatrixXd local_cov =
            Eigen::MatrixXd::Zero(data.dimensionality(), data.dimensionality());

        Eigen::MatrixXd centered(data.dimensionality(), kBatchSize);
        for (size_t begin = inner_batch_idx_start * kBatchSize;
             begin < inner_batch_idx_end * kBatchSize; begin += kBatchSize) {
          centered.setZero();
          size_t size = std::min(data.size() - begin, kBatchSize);

          for (size_t dp_idx = begin, i = 0; i < size; dp_idx++, i++) {
            for (size_t j = 0; j < data.dimensionality(); j++) {
              centered(j, i) = data.GetPtr(dp_idx)[j] - mean[j];
            }
          }

          local_cov.triangularView<Eigen::Lower>() +=
              centered * centered.transpose();
        }
        local_cov.template triangularView<Eigen::Upper>() =
            local_cov.transpose();
        covariance_mutex.Lock();
        covariance += local_cov;
        covariance_mutex.Unlock();
      });
  eigenvectors->resize(num_eigenvectors);
  eigenvalues->resize(num_eigenvectors);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);
  for (int i = 0; i < num_eigenvectors; i++) {
    std::vector<float>* cur = eigenvectors->at(i).mutable_values();
    cur->resize(data.dimensionality());

    eigenvalues->at(i) = eigen_solver.eigenvalues()(cur->size() - i - 1);
    for (int j = 0; j < cur->size(); j++)
      cur->at(j) = eigen_solver.eigenvectors()(j, cur->size() - i - 1);
  }
}

}  // namespace research_scann

#endif
