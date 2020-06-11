// Copyright 2020 The Google Research Authors.
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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__UTILS_GMM_UTILS_H_
#define SCANN__UTILS_GMM_UTILS_H_

#include <limits>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace scann_ops {

class GmmUtilsImplInterface;

class GmmUtils {
 public:
  struct Options {
    int32_t seed = kDeterministicSeed;

    int32_t max_iterations = 10;

    double epsilon = 1e-5;

    int32_t min_cluster_size = 1;

    int32_t max_cluster_size = std::numeric_limits<int32_t>::max();

    double perturbation = 1e-7;

    shared_ptr<thread::ThreadPool> parallelization_pool;

    enum PartitionAssignmentType {
      UNBALANCED,

      GREEDY_BALANCED,

      MIN_COST_MAX_FLOW,
    };

    PartitionAssignmentType partition_assignment_type = UNBALANCED;

    enum CenterReassignmentType {
      RANDOM_REASSIGNMENT,

      SPLIT_LARGEST_CLUSTERS,

      PCA_SPLITTING,
    };

    CenterReassignmentType center_reassignment_type = RANDOM_REASSIGNMENT;

    enum CenterInitializationType {
      KMEANS_PLUS_PLUS,

      RANDOM_INITIALIZATION,
    };

    CenterInitializationType center_initialization_type = KMEANS_PLUS_PLUS;

    int32_t max_power_of_2_split = 1;

    double parallel_cost_multiplier = 1.0;
  };

  GmmUtils(shared_ptr<const DistanceMeasure> dist, Options opts);

  explicit GmmUtils(shared_ptr<const DistanceMeasure> dist)
      : GmmUtils(std::move(dist), Options()) {}

  Status GenericKmeans(
      const Dataset& dataset, const int32_t num_clusters,
      DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions = nullptr);
  Status GenericKmeans(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const int32_t num_clusters, DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions = nullptr);

  Status SphericalKmeans(
      const Dataset& dataset, const int32_t num_clusters,
      DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions = nullptr);
  Status SphericalKmeans(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const int32_t num_clusters, DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions = nullptr);

  StatusOr<double> ComputeSpillingThreshold(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const DenseDataset<double>& centers,
      const DatabaseSpillingConfig::SpillingType spilling_type,
      const float total_spill_factor, const uint32_t max_centers);

  Status RecomputeCentroidsSimple(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids) const;
  Status RecomputeCentroidsWithParallelCostMultiplier(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids) const;

  Status ComputeSpillingThreshold(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const DenseDataset<double>& centers,
      const DatabaseSpillingConfig::SpillingType spilling_type,
      const float total_spill_factor, const uint32_t max_centers,
      double* result) {
    DCHECK(result);
    TF_ASSIGN_OR_RETURN(*result, ComputeSpillingThreshold(
                                     dataset, subset, centers, spilling_type,
                                     total_spill_factor, max_centers));
    return OkStatus();
  }

  Status InitializeCenters(const Dataset& dataset,
                           ConstSpan<DatapointIndex> subset,
                           int32_t num_clusters,
                           DenseDataset<double>* initial_centers);

  void set_parallelization_pool(shared_ptr<thread::ThreadPool> pool) {
    opts_.parallelization_pool = std::move(pool);
  }

  Status SphericalKmeans(const Dataset& dataset,
                         ConstSpan<DatapointIndex> subset,
                         const DenseDataset<double>& initial_centers,
                         DenseDataset<double>* final_centers,
                         vector<vector<DatapointIndex>>* final_partitions);

  Status GenericKmeans(const Dataset& dataset, ConstSpan<DatapointIndex> subset,
                       const DenseDataset<double>& initial_centers,
                       DenseDataset<double>* final_centers,
                       vector<vector<DatapointIndex>>* final_partitions);

  using PartitionAssignmentFn =
      std::function<vector<pair<DatapointIndex, double>>(
          GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
          const DenseDataset<double>& center, thread::ThreadPool* pool)>;

 private:
  SCANN_OUTLINE Status KMeansImpl(
      bool spherical, const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      int32_t num_clusters, PartitionAssignmentFn partition_assignment_fn,
      DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions,
      bool preinitialized_centers = false);

  Status RandomReinitializeCenters(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids,
      std::vector<double>* convergence_means);

  Status SplitLargeClusterReinitialization(
      ConstSpan<uint32_t> partition_sizes, bool spherical,
      DenseDataset<double>* centroids, std::vector<double>* convergence_means);

  Status PCAKmeansReinitialization(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids,
      std::vector<double>* convergence_means) const;

  Status KMeansPPInitializeCenters(const Dataset& dataset,
                                   ConstSpan<DatapointIndex> subset,
                                   int32_t num_clusters,
                                   DenseDataset<double>* initial_centers);

  Status RandomInitializeCenters(const Dataset& dataset,
                                 ConstSpan<DatapointIndex> subset,
                                 int32_t num_clusters,
                                 DenseDataset<double>* initial_centers);

  shared_ptr<const DistanceMeasure> distance_;
  Options opts_;
  MTRandom random_;
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
