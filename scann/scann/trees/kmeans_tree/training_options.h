// Copyright 2021 The Google Research Authors.
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



#ifndef SCANN__TREES_KMEANS_TREE_TRAINING_OPTIONS_H_
#define SCANN__TREES_KMEANS_TREE_TRAINING_OPTIONS_H_

#include "scann/proto/partitioning.pb.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {

struct KMeansTreeTrainingOptions {
  KMeansTreeTrainingOptions();

  explicit KMeansTreeTrainingOptions(const PartitioningConfig& config);

  PartitioningConfig::PartitioningType partitioning_type =
      PartitioningConfig::GENERIC;

  GmmUtils::Options::PartitionAssignmentType balancing_type =
      GmmUtils::Options::UNBALANCED;

  GmmUtils::Options::CenterReassignmentType reassignment_type =
      GmmUtils::Options::RANDOM_REASSIGNMENT;

  GmmUtils::Options::CenterInitializationType center_initialization_type =
      GmmUtils::Options::KMEANS_PLUS_PLUS;

  shared_ptr<thread::ThreadPool> training_parallelization_pool = nullptr;

  int32_t max_num_levels = 1;

  int32_t max_leaf_size = 1;

  DatabaseSpillingConfig::SpillingType learned_spilling_type =
      DatabaseSpillingConfig::NO_SPILLING;

  double per_node_spilling_factor = 1.0;

  uint32_t max_spill_centers = numeric_limits<uint32_t>::max();

  int32_t max_iterations = 10;

  double convergence_epsilon = 1e-5;

  int32_t min_cluster_size = 1;

  int32_t seed = 0;

  bool compute_residual_stdev = false;

  double residual_stdev_min_value = 1e-5;
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
