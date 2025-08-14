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



#include "scann/trees/kmeans_tree/kmeans_tree_node.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_castops.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

KMeansTreeNode::KMeansTreeNode() = default;

KMeansTreeNode KMeansTreeNode::CreateFlat(DenseDataset<float> centers) {
  KMeansTreeNode root;
  root.float_centers_ = std::move(centers);
  root.children_ = vector<KMeansTreeNode>(root.float_centers_.size());
  root.NumberLeaves(0);
  root.MaybeInitializeThreadSharding();
  return root;
}

void KMeansTreeNode::Reset() {
  leaf_id_ = -1;
  learned_spilling_threshold_ = numeric_limits<double>::quiet_NaN();
  indices_.clear();
  children_.clear();
}

void KMeansTreeNode::UnionIndices(vector<DatapointIndex>* result) const {
  CHECK(result);
  absl::flat_hash_set<DatapointIndex> union_hash;
  UnionIndicesImpl(&union_hash);
  result->clear();
  for (DatapointIndex elem : union_hash) {
    result->push_back(elem);
  }
}

namespace {

template <typename T>
Datapoint<float> ToDatapoint(google::protobuf::RepeatedField<T> values) {
  Datapoint<float> dp;
  dp.mutable_values()->reserve(values.size());
  for (const auto& elem : values) {
    dp.mutable_values()->push_back(elem);
  }
  return dp;
}

}  // namespace

void KMeansTreeNode::BuildFromProto(const SerializedKMeansTree::Node& proto) {
  float_centers_.clear();
  Datapoint<float> dp;
  for (size_t i = 0; i < proto.centers_size(); ++i) {
    if (!proto.centers(i).float_dimension().empty()) {
      dp = ToDatapoint(proto.centers(i).float_dimension());
    } else {
      dp = ToDatapoint(proto.centers(i).dimension());
    }

    if (i == 0) {
      float_centers_.set_dimensionality(dp.dimensionality());
      float_centers_.Reserve(proto.centers_size());
    }

    float_centers_.AppendOrDie(dp.ToPtr(), "");
  }

  MaybeInitializeThreadSharding();
  learned_spilling_threshold_ = proto.learned_spilling_threshold();
  leaf_id_ = proto.leaf_id();

  indices_.clear();
  children_.clear();
  if (proto.children_size() == 0) {
    indices_.insert(indices_.begin(), proto.indices().begin(),
                    proto.indices().end());
  } else {
    children_ = vector<KMeansTreeNode>(proto.children_size());
    for (size_t i = 0; i < proto.children_size(); ++i) {
      children_[i].BuildFromProto(proto.children(i));
    }
  }
}

namespace kmeans_tree_internal {

Status PostprocessDistancesForSpilling(
    ConstSpan<float> distances, QuerySpillingConfig::SpillingType spilling_type,
    double spilling_threshold, int32_t max_centers,
    int32_t num_tokenized_branch,
    std::vector<pair<DatapointIndex, float>>* child_centers) {
  float epsilon = std::numeric_limits<float>::infinity();
  if (spilling_type != QuerySpillingConfig::NO_SPILLING &&
      spilling_type != QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    const size_t nearest_center_index =
        std::distance(distances.begin(),
                      std::min_element(distances.begin(), distances.end()));
    const float nearest_center_distance = distances[nearest_center_index];

    using cast_ops::DoubleToFloat;

    float spill_thresh = std::nextafter(DoubleToFloat(spilling_threshold),
                                        std::numeric_limits<float>::infinity());
    SCANN_ASSIGN_OR_RETURN(
        float max_dist_to_consider,
        ComputeThreshold(nearest_center_distance, spill_thresh, spilling_type));
    epsilon = std::nextafter(max_dist_to_consider,
                             std::numeric_limits<float>::infinity());
  }
  const int32_t max_results =
      (spilling_type == QuerySpillingConfig::NO_SPILLING)
          ? std::max(1, num_tokenized_branch)
          : max_centers;
  FastTopNeighbors<float> top_n(max_results, epsilon);
  top_n.PushBlock(distances, 0);
  top_n.FinishUnsorted(child_centers);
  return OkStatus();
}

}  // namespace kmeans_tree_internal

Status KMeansTreeNode::Train(const Dataset& training_data,
                             vector<DatapointIndex> subset,
                             const DistanceMeasure& training_distance,
                             int32_t k_per_level, int32_t current_level,
                             KMeansTreeTrainingOptions* opts) {
  indices_ = std::move(subset);
  if (indices_.size() <= opts->max_leaf_size) {
    return OkStatus();
  }

  if (opts->max_num_levels <= current_level) {
    return OkStatus();
  }

  GmmUtils::Options gmm_opts;
  gmm_opts.max_iterations = opts->max_iterations;
  gmm_opts.epsilon = opts->convergence_epsilon;
  gmm_opts.max_iteration_duration = opts->max_iteration_duration;
  gmm_opts.seed = opts->seed + kDeterministicSeed;
  gmm_opts.min_cluster_size = opts->min_cluster_size;
  gmm_opts.parallelization_pool = opts->training_parallelization_pool;
  gmm_opts.partition_assignment_type = opts->balancing_type;
  gmm_opts.center_reassignment_type = opts->reassignment_type;
  gmm_opts.center_initialization_type = opts->center_initialization_type;
  GmmUtils gmm(MakeDummyShared(&training_distance), gmm_opts);

  vector<vector<DatapointIndex>> subpartitions;
  DenseDataset<double> centers;
  SCANN_RETURN_IF_ERROR(gmm.ComputeKmeansClustering(
      training_data, k_per_level, &centers,
      {.subset = indices_,
       .final_partitions = &subpartitions,
       .spherical = opts->partitioning_type == PartitioningConfig::SPHERICAL,
       .first_n_centroids = opts->first_n_centroids}));

  DatabaseSpillingConfig::SpillingType spilling_type =
      opts->learned_spilling_type;
  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    SCANN_ASSIGN_OR_RETURN(
        learned_spilling_threshold_,
        gmm.ComputeSpillingThreshold(
            training_data, indices_, centers, opts->learned_spilling_type,
            opts->per_node_spilling_factor, opts->max_spill_centers));
  }

  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    vector<vector<DatapointIndex>> spilled(centers.size());
    for (DatapointIndex i : indices_) {
      Datapoint<double> double_dp;
      training_data.GetDatapoint(i, &double_dp);
      vector<pair<DatapointIndex, float>> spill_centers;

      {
        std::vector<float> tmp_dists(centers.size());
        kmeans_tree_internal::GetAllDistancesFloatingPointNoThreadSharding<
            double, float>(training_distance, double_dp.ToPtr(), centers,
                           MakeMutableSpan(tmp_dists));

        SCANN_RETURN_IF_ERROR(
            kmeans_tree_internal::PostprocessDistancesForSpilling(
                tmp_dists,
                static_cast<QuerySpillingConfig::SpillingType>(spilling_type),
                learned_spilling_threshold_, opts->max_spill_centers, -1,
                &spill_centers));
      }

      for (const auto& center_index : spill_centers) {
        spilled[center_index.first].push_back(i);
      }
    }

    const size_t max_subpartition_size =
        static_cast<size_t>(floor(0.99 * indices_.size()));
    for (const auto& subpartition : spilled) {
      if (subpartition.size() >= max_subpartition_size) {
        LOG(INFO) << "KILL SPILL " << subpartition.size();
        learned_spilling_threshold_ = NAN;
        spilling_type = DatabaseSpillingConfig::NO_SPILLING;
        break;
      }
    }

    if (!std::isnan(learned_spilling_threshold_)) {
      subpartitions.swap(spilled);
    }
  }

  FreeBackingStorage(&indices_);
  children_ = vector<KMeansTreeNode>(centers.size());
  for (size_t i = 0; i < children_.size(); ++i) {
    children_[i].Reset();
    Status status = children_[i].Train(
        training_data, std::move(subpartitions[i]), training_distance,
        k_per_level, current_level + 1, opts);
    if (!status.ok()) return status;
  }

  centers.ConvertType(&float_centers_);
  MaybeInitializeThreadSharding();
  return OkStatus();
}

void KMeansTreeNode::CreateFixedPointCenters() {
  if (!fixed_point_centers_.empty()) return;

  center_squared_l2_norms_.resize(float_centers_.size());
  for (auto [i, norm] : Enumerate(center_squared_l2_norms_))
    norm = SquaredL2Norm(float_centers_[i]);
  ScalarQuantizationResults results =
      ScalarQuantizeFloatDataset(float_centers_, 1.0, NAN);
  inv_int8_multipliers_ = std::move(results.inverse_multiplier_by_dimension);
  fixed_point_centers_ = std::move(results.quantized_dataset);

  for (KMeansTreeNode& child : children_) {
    child.CreateFixedPointCenters();
  }
}

Status KMeansTreeNode::CheckDimensionality(DimensionIndex query_dims) const {
  if (float_centers_.empty()) {
    return OkStatus();
  } else if (float_centers_.dimensionality() == query_dims) {
    return OkStatus();
  } else {
    const std::string error_msg =
        StrFormat("Incorrect query dimensionality.  Expected %u, got %u.\n",
                  static_cast<uint64_t>(float_centers_.dimensionality()),
                  static_cast<uint64_t>(query_dims));
    return FailedPreconditionError(error_msg);
  }
}

int32_t KMeansTreeNode::NumberLeaves(int32_t m) {
  if (IsLeaf()) {
    leaf_id_ = m;
    return m + 1;
  } else {
    leaf_id_ = -1;
    for (KMeansTreeNode& child : children_) {
      m = child.NumberLeaves(m);
    }
  }

  return m;
}

void KMeansTreeNode::PopulateCurNodeCenters() {
  for (size_t i = 0; i < children_.size(); ++i) {
    children_[i].cur_node_center_ = Centers()[i];
    children_[i].PopulateCurNodeCenters();
  }
}

void KMeansTreeNode::CopyToProto(SerializedKMeansTree::Node* proto,
                                 bool with_indices) const {
  CHECK(proto != nullptr);
  for (DatapointIndex i = 0; i < float_centers_.size(); ++i) {
    const DatapointPtr<float> center = float_centers_[i];
    DCHECK(center.IsDense());
    auto center_proto = proto->add_centers();
    for (const float& elem : center.values_span()) {
      center_proto->add_dimension(elem);
    }
  }

  proto->set_leaf_id(leaf_id_);
  proto->set_learned_spilling_threshold(learned_spilling_threshold_);

  if (IsLeaf() && with_indices) {
    for (const auto& index : indices_) {
      proto->add_indices(index);
    }
  } else {
    for (const auto& child : children_) {
      auto child_proto = proto->add_children();
      child.CopyToProto(child_proto, with_indices);
    }
  }
}

void KMeansTreeNode::UnionIndicesImpl(
    absl::flat_hash_set<DatapointIndex>* union_hash) const {
  CHECK(union_hash);
  if (IsLeaf()) {
    for (auto index : indices_) {
      union_hash->insert(index);
    }
  } else {
    for (const auto& child : children_) {
      child.UnionIndicesImpl(union_hash);
    }
  }
}

void KMeansTreeNode::MaybeInitializeThreadSharding() {}

}  // namespace research_scann
