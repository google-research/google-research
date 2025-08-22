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



#ifndef SCANN_UTILS_GMM_UTILS_H_
#define SCANN_UTILS_GMM_UTILS_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

class GmmUtilsImplInterface;

class GmmUtils {
 public:
  struct Options {
    int32_t seed = kDeterministicSeed;

    int32_t max_iterations = 10;

    double epsilon = 1e-5;

    absl::Duration max_iteration_duration = absl::InfiniteDuration();

    int32_t min_cluster_size = 1;

    int32_t max_cluster_size = std::numeric_limits<int32_t>::max();

    double perturbation = 1e-7;

    shared_ptr<ThreadPool> parallelization_pool;

    enum PartitionAssignmentType {

      UNBALANCED,

      GREEDY_BALANCED,

      MIN_COST_MAX_FLOW,

      UNBALANCED_FLOAT32,
    };

    PartitionAssignmentType partition_assignment_type = UNBALANCED_FLOAT32;

    enum CenterReassignmentType {

      RANDOM_REASSIGNMENT,

      SPLIT_LARGEST_CLUSTERS,

      PCA_SPLITTING,
    };

    CenterReassignmentType center_reassignment_type = RANDOM_REASSIGNMENT;

    enum CenterInitializationType {

      MEAN_DISTANCE_INITIALIZATION,

      KMEANS_PLUS_PLUS,

      RANDOM_INITIALIZATION,
    };

    CenterInitializationType center_initialization_type = KMEANS_PLUS_PLUS;

    int32_t max_power_of_2_split = 1;
  };

  GmmUtils(shared_ptr<const DistanceMeasure> dist, Options opts);

  explicit GmmUtils(shared_ptr<const DistanceMeasure> dist)
      : GmmUtils(std::move(dist), Options()) {}

  struct ComputeKmeansClusteringOptions {
    ConstSpan<DatapointIndex> subset;

    vector<vector<DatapointIndex>>* final_partitions = nullptr;

    bool spherical = false;

    ConstSpan<float> weights;

    std::optional<DatapointIndex> first_n_centroids;
  };

  Status ComputeKmeansClustering(
      const Dataset& dataset, int32_t num_clusters,
      DenseDataset<double>* final_centers,
      const ComputeKmeansClusteringOptions& kmeans_opts);

  Status ComputeKmeansClustering(
      GmmUtilsImplInterface* impl, int32_t num_clusters,
      DenseDataset<double>* final_centers,
      const ComputeKmeansClusteringOptions& kmeans_opts);

  StatusOr<double> ComputeSpillingThreshold(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const DenseDataset<double>& centers,
      DatabaseSpillingConfig::SpillingType spilling_type,
      float total_spill_factor, DatapointIndex max_centers);

  template <typename FloatT>
  Status RecomputeCentroidsSimple(
      ConstSpan<pair<DatapointIndex, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means);

  template <typename FloatT>
  Status RecomputeCentroidsWeighted(
      ConstSpan<pair<DatapointIndex, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      ConstSpan<float> weights, bool spherical,
      DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means);

  Status InitializeCenters(const Dataset& dataset,
                           ConstSpan<DatapointIndex> subset,
                           int32_t num_clusters, ConstSpan<float> weights,
                           DenseDataset<double>* initial_centers);

  Status InitializeCenters(GmmUtilsImplInterface* impl, int32_t num_clusters,
                           ConstSpan<float> weights,
                           DenseDataset<double>* initial_centers);

 private:
  template <typename FloatT>
  Status ReinitializeCenters(
      ConstSpan<pair<DatapointIndex, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means);

  template <typename FloatT>
  Status RandomReinitializeCenters(
      ConstSpan<pair<DatapointIndex, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means);

  template <typename FloatT>
  Status SplitLargeClusterReinitialization(
      ConstSpan<uint32_t> partition_sizes, bool spherical,
      DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means);

  template <typename FloatT>
  Status PCAKmeansReinitialization(
      ConstSpan<pair<DatapointIndex, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDatasetView<FloatT>* centroids,
      std::vector<double>* convergence_means) const;

  Status MeanDistanceInitializeCenters(GmmUtilsImplInterface* impl,
                                       int32_t num_clusters,
                                       ConstSpan<float> weights,
                                       DenseDataset<double>* initial_centers);
  Status KMeansPPInitializeCenters(GmmUtilsImplInterface* impl,
                                   int32_t num_clusters,
                                   ConstSpan<float> weights,
                                   DenseDataset<double>* initial_centers);
  Status RandomInitializeCenters(GmmUtilsImplInterface* impl,
                                 int32_t num_clusters, ConstSpan<float> weights,
                                 DenseDataset<double>* initial_centers);

  shared_ptr<const DistanceMeasure> distance_;
  Options opts_;
  MTRandom random_;
};

class GmmUtilsImplInterface : public VirtualDestructor {
 public:
  static unique_ptr<GmmUtilsImplInterface> Create(
      const DistanceMeasure& distance, const Dataset& dataset,
      ConstSpan<DatapointIndex> subset, ThreadPool* parallelization_pool);

  template <typename T>
  static unique_ptr<GmmUtilsImplInterface> Create(
      const DistanceMeasure& distance, ConstSpan<T> data,
      DatapointIndex dimensionality, ConstSpan<DatapointIndex> subset,
      Normalization normalization = NONE,
      ThreadPool* parallelization_pool = nullptr);

  virtual size_t size() const = 0;

  virtual size_t dimensionality() const = 0;

  Normalization normalization() { return normalization_; };

  virtual Status GetCentroid(Datapoint<double>* centroid) const = 0;

  virtual DatapointPtr<double> GetPoint(size_t idx,
                                        Datapoint<double>* storage) const = 0;

  virtual DatapointPtr<float> GetPoint(size_t idx,
                                       Datapoint<float>* storage) const = 0;

  virtual DatapointIndex GetOriginalIndex(size_t idx) const = 0;

  using IterateDatasetCallback = std::function<void(
      size_t offset, const DenseDataset<double>& dataset_batch)>;
  using IterateDatasetCallbackFloat = std::function<void(
      size_t offset, DefaultDenseDatasetView<float> dataset_batch)>;
  virtual void IterateDataset(ThreadPool* parallelization_pool,
                              const IterateDatasetCallback& callback) const = 0;
  virtual void IterateDataset(
      ThreadPool* parallelization_pool,
      const IterateDatasetCallbackFloat& callback) const = 0;

  void DistancesFromPoint(DatapointPtr<double> center,
                          MutableSpan<double> distances) const;

  Status CheckDataDegeneracy();

  Status CheckAllFinite() const;

  ThreadPool* GetThreadPool() const { return parallelization_pool_; }

 private:
  template <typename T>
  static unique_ptr<GmmUtilsImplInterface> CreateTyped(
      const DistanceMeasure& distance, const Dataset& dataset,
      ConstSpan<DatapointIndex> subset, ThreadPool* parallelization_pool);

  Normalization normalization_;
  const DistanceMeasure* distance_;
  ThreadPool* parallelization_pool_;
};

}  // namespace research_scann

#endif
