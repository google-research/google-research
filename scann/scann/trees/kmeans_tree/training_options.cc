// Copyright 2022 The Google Research Authors.
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



#include "scann/trees/kmeans_tree/training_options.h"

#include "scann/proto/partitioning.pb.h"
#include "scann/utils/gmm_utils.h"

namespace research_scann {

KMeansTreeTrainingOptions::KMeansTreeTrainingOptions() {}

KMeansTreeTrainingOptions::KMeansTreeTrainingOptions(
    const PartitioningConfig& config)
    : partitioning_type(config.partitioning_type()),
      max_num_levels(config.max_num_levels()),
      max_leaf_size(config.max_leaf_size()),
      learned_spilling_type(config.database_spilling().spilling_type()),
      per_node_spilling_factor(config.database_spilling().replication_factor()),
      max_spill_centers(config.database_spilling().max_spill_centers()),
      max_iterations(config.max_clustering_iterations()),
      convergence_epsilon(config.clustering_convergence_tolerance()),
      min_cluster_size(config.min_cluster_size()),
      seed(config.clustering_seed()),
      compute_residual_stdev(config.compute_residual_stdev()),
      residual_stdev_min_value(config.residual_stdev_min_value()) {
  switch (config.balancing_type()) {
    case PartitioningConfig::DEFAULT_UNBALANCED:
      balancing_type = GmmUtils::Options::UNBALANCED;
      break;
    case PartitioningConfig::GREEDY_BALANCED:
      balancing_type = GmmUtils::Options::GREEDY_BALANCED;
      break;
  }
  switch (config.trainer_type()) {
    case PartitioningConfig::DEFAULT_SAMPLING_TRAINER:
    case PartitioningConfig::FLUME_KMEANS_TRAINER:
      reassignment_type = GmmUtils::Options::RANDOM_REASSIGNMENT;
      break;
    case PartitioningConfig::PCA_KMEANS_TRAINER:
    case PartitioningConfig::SAMPLING_PCA_KMEANS_TRAINER:
      reassignment_type = GmmUtils::Options::PCA_SPLITTING;
      break;
  }
  switch (config.single_machine_center_initialization()) {
    case PartitioningConfig::DEFAULT_KMEANS_PLUS_PLUS:
      center_initialization_type = GmmUtils::Options::KMEANS_PLUS_PLUS;
      break;
    case PartitioningConfig::RANDOM_INITIALIZATION:
      center_initialization_type = GmmUtils::Options::RANDOM_INITIALIZATION;
      break;
  }
}

}  // namespace research_scann
