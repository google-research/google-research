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

#include "scann/utils/single_machine_autopilot.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/log.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/auto_tuning.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/projection.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<ScannConfig> AutopilotTreeAh(const ScannConfig& config,
                                      shared_ptr<const Dataset> dataset,
                                      DatapointIndex n, DimensionIndex dim) {
  const auto dist = config.distance_measure().distance_measure();
  if (dist != "SquaredL2Distance" && dist != "DotProductDistance") {
    return UnimplementedError(
        "Autopilot is currently implemented only for DotProductDistance "
        "and SquaredL2Distance.");
  }
  auto result = config;
  result.clear_brute_force();
  result.clear_hash();
  result.clear_partitioning();
  result.clear_exact_reordering();
  if (dataset != nullptr) {
    dim = dataset->dimensionality();
    n = dataset->size();
  }

  const int ah_size = 2;

  const int kmeans_stable_size = 100;

  const int safety = 2;

  const int magic = 42;

  const int l1_size = config.autopilot().tree_ah().l1_size();
  const int k = config.num_neighbors();
  const int l3_size_bound =
      std::ceil(config.autopilot().tree_ah().l3_size() / dim / sizeof(float));

  int ah2_leaf_size = std::ceil(ah_size * 2 * l1_size / dim);
  ah2_leaf_size = std::max(ah2_leaf_size, safety * kmeans_stable_size);

  const int approx_num_neighbors =
      std::ceil(std::max(1.0 * safety * k, 100 * sqrt(k)));

  int treeah_bound =
      std::max(safety * approx_num_neighbors, magic * ah2_leaf_size);
  VLOG(2) << "Minimal Tree AH Size: " << treeah_bound;

  if (n < treeah_bound) {
    result.mutable_brute_force();
    return result;
  }

  result.mutable_exact_reordering()->set_approx_num_neighbors(
      approx_num_neighbors);
  if (config.autopilot().tree_ah().reordering_dtype() ==
      AutopilotTreeAH::INT8) {
    result.mutable_exact_reordering()->mutable_fixed_point()->set_enabled(true);
  } else if (config.autopilot().tree_ah().reordering_dtype() ==
             AutopilotTreeAH::BFLOAT16) {
    result.mutable_exact_reordering()->mutable_bfloat16()->set_enabled(true);
  }

  auto ah = result.mutable_hash()->mutable_asymmetric_hash();
  ah->mutable_quantization_distance()->set_distance_measure(
      "SquaredL2Distance");
  ah->set_num_clusters_per_block(16);
  ah->set_max_clustering_iterations(10);
  ah->set_lookup_type(ah->INT8_LUT16);

  ah->set_expected_sample_size(16 * kmeans_stable_size * safety * 10);
  if (dist == "DotProductDistance") {
    ah->set_use_residual_quantization(true);
    ah->set_use_global_topn(true);
    ah->set_noise_shaping_threshold(0.2);
  }

  int full_blocks = dim / ah_size;
  int partial_block_dims = dim % ah_size;
  auto ah_proj = ah->mutable_projection();
  ah_proj->set_input_dim(dim);
  ah_proj->set_num_dims_per_block(ah_size);
  if (partial_block_dims == 0) {
    ah_proj->set_projection_type(ah_proj->CHUNK);
    ah_proj->set_num_blocks(full_blocks);
    ah_proj->set_num_dims_per_block(ah_size);
  } else {
    ah_proj->set_projection_type(ah_proj->VARIABLE_CHUNK);
    auto vblock = ah_proj->add_variable_blocks();
    vblock->set_num_blocks(full_blocks);
    vblock->set_num_dims_per_block(ah_size);
    vblock = ah_proj->add_variable_blocks();
    vblock->set_num_blocks(1);
    vblock->set_num_dims_per_block(partial_block_dims);
  }

  VLOG(2) << "AH2 Leaf Size: " << ah2_leaf_size;
  int tree_size = n / ah2_leaf_size;

  const int train_size_bound = std::ceil(
      std::sqrt(60.0 * 32 * 2e9 / dim / (safety * kmeans_stable_size)));
  VLOG(2) << "L1 recommended tree size: " << tree_size;
  VLOG(2) << "L3 recommended tree size: " << l3_size_bound;
  VLOG(2) << "Training time recommended tree size: " << train_size_bound;
  tree_size = std::min(tree_size, l3_size_bound);
  tree_size = std::min(tree_size, train_size_bound);

  auto part = result.mutable_partitioning();
  part->set_num_children(tree_size);
  part->set_expected_sample_size(tree_size * kmeans_stable_size * safety);
  part->set_min_cluster_size(10);
  part->set_max_clustering_iterations(10);
  part->set_single_machine_center_initialization(part->RANDOM_INITIALIZATION);
  part->mutable_partitioning_distance()->set_distance_measure(
      "SquaredL2Distance");
  part->mutable_query_spilling()->set_spilling_type(
      part->query_spilling().FIXED_NUMBER_OF_CENTERS);

  const int leaves_to_search =
      std::ceil(magic * std::pow(2.0, std::log(1.0 * tree_size / magic) /
                                          std::log(10.0)));

  part->mutable_query_spilling()->set_max_spill_centers(
      std::min(tree_size, leaves_to_search));
  part->mutable_query_tokenization_distance_override()->set_distance_measure(
      dist);
  part->set_partitioning_type(part->GENERIC);

  part->set_query_tokenization_type(part->FLOAT);

  switch (config.autopilot().tree_ah().incremental_mode()) {
    case (AutopilotTreeAH::ONLINE_INCREMENTAL):
      part->mutable_incremental_training_config()->set_autopilot(true);
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    case (AutopilotTreeAH::ONLINE):
      part->mutable_incremental_training_config()->set_fraction(0.5);
      break;
    default:
      break;
  }
  VLOG(1) << "Autopilot Tree AH Result: " << result.DebugString();
  return result;
}

StatusOr<ScannConfig> Autopilot(const ScannConfig& config,
                                shared_ptr<const Dataset> dataset,
                                DatapointIndex n, DimensionIndex dim) {
  if (!config.has_autopilot())
    return FailedPreconditionError("Autopilot config is not present.");
  auto ds = std::static_pointer_cast<const DenseDataset<float>>(dataset);
  if (ds == nullptr &&
      (n == kInvalidDatapointIndex || dim == kInvalidDimension))
    return FailedPreconditionError(
        "Autopilot requires either original, uncompressed Dataset, or "
        "explicitly specified dimensionality and size.");
  switch (config.autopilot().autopilot_option_case()) {
    case (AutopilotConfig::AutopilotOptionCase::AUTOPILOT_OPTION_NOT_SET):
    case (AutopilotConfig::AutopilotOptionCase::kTreeAh):
      return AutopilotTreeAh(config, ds, n, dim);
    default:
      return FailedPreconditionError("Autopilot option not supported: %s",
                                     config.autopilot().DebugString());
  }
}

}  // namespace research_scann
