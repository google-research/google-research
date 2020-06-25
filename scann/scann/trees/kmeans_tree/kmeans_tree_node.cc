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



#include "scann/trees/kmeans_tree/kmeans_tree_node.h"

#include <math.h>

#include <hash_set>

#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace tensorflow {
namespace scann_ops {

static const int32_t kMaxIterations = 1000;
static const double kEpsilon = 1e-3;

KMeansTreeNode::KMeansTreeNode() {}

void KMeansTreeNode::Reset() {
  leaf_id_ = -1;
  learned_spilling_threshold_ = numeric_limits<double>::quiet_NaN();
  indices_.clear();
  children_.clear();
  residual_stdevs_.clear();
}

void KMeansTreeNode::UnionIndices(vector<DatapointIndex>* result) const {
  CHECK(result);
  std::unordered_set<DatapointIndex> union_hash;
  UnionIndicesImpl(&union_hash);
  result->clear();
  for (DatapointIndex elem : union_hash) {
    result->push_back(elem);
  }
}

namespace {

template <typename T>
Datapoint<float> ToDatapoint(protobuf::RepeatedField<T> values) {
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

  learned_spilling_threshold_ = proto.learned_spilling_threshold();
  leaf_id_ = proto.leaf_id();

  indices_.clear();
  children_.clear();
  residual_stdevs_.clear();
  residual_stdevs_.insert(residual_stdevs_.begin(),
                          proto.residual_stdevs().begin(),
                          proto.residual_stdevs().end());
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
  gmm_opts.seed = opts->seed + kDeterministicSeed;
  gmm_opts.min_cluster_size = opts->min_cluster_size;
  gmm_opts.parallelization_pool = opts->training_parallelization_pool;
  gmm_opts.partition_assignment_type = opts->balancing_type;
  gmm_opts.center_reassignment_type = opts->reassignment_type;
  gmm_opts.center_initialization_type = opts->center_initialization_type;
  GmmUtils gmm(MakeDummyShared(&training_distance), gmm_opts);

  vector<vector<DatapointIndex>> subpartitions;
  DenseDataset<double> centers;
  if (opts->partitioning_type == PartitioningConfig::SPHERICAL) {
    SCANN_RETURN_IF_ERROR(gmm.SphericalKmeans(
        training_data, indices_, k_per_level, &centers, &subpartitions));
  } else {
    DCHECK_EQ(opts->partitioning_type, PartitioningConfig::GENERIC);
    SCANN_RETURN_IF_ERROR(gmm.GenericKmeans(
        training_data, indices_, k_per_level, &centers, &subpartitions));
  }

  DatabaseSpillingConfig::SpillingType spilling_type =
      opts->learned_spilling_type;
  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    SCANN_RETURN_IF_ERROR(gmm.ComputeSpillingThreshold(
        training_data, indices_, centers, opts->learned_spilling_type,
        opts->per_node_spilling_factor, opts->max_spill_centers,
        &learned_spilling_threshold_));
  }

  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    vector<vector<DatapointIndex>> spilled(centers.size());
    for (DatapointIndex i : indices_) {
      Datapoint<double> double_dp;
      training_data.GetDatapoint(i, &double_dp);
      vector<pair<DatapointIndex, float>> spill_centers;
      Status status =
          kmeans_tree_internal::FindChildrenWithSpilling<double, double>(
              double_dp.ToPtr(),
              static_cast<QuerySpillingConfig::SpillingType>(spilling_type),
              learned_spilling_threshold_, opts->max_spill_centers,
              training_distance, centers, NAN, &spill_centers);

      SCANN_RETURN_IF_ERROR(status);
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

  if (opts->compute_residual_stdev) {
    residual_stdevs_.resize(centers.size());
    ParallelFor<1>(Seq(centers.size()),
                   opts->training_parallelization_pool.get(), [&](size_t i) {
                     double sq_residual_sum = 0.0;
                     uint32_t count = subpartitions[i].size();
                     DatapointPtr<double> center = centers.at(i);
                     for (auto j : subpartitions[i]) {
                       Datapoint<double> double_dp;
                       training_data.GetDatapoint(j, &double_dp);
                       sq_residual_sum +=
                           SquaredL2DistanceBetween(double_dp.ToPtr(), center);
                     }

                     residual_stdevs_[i] = std::max(
                         count == 0
                             ? 1.0
                             : count == 1
                                   ? std::sqrt(sq_residual_sum)
                                   : std::sqrt(sq_residual_sum / (count - 1)),
                         opts->residual_stdev_min_value);
                   });
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

  return OkStatus();
}

void KMeansTreeNode::CreateFixedPointCenters() {
  if (!fixed_point_centers_.empty()) return;
  double max_abs_value = 0.0;
  auto centers_data = float_centers_.data();
  for (double v : centers_data) {
    max_abs_value = std::max(std::abs(v), max_abs_value);
  }

  double fixed_point_multiplier = numeric_limits<int8_t>::max() / max_abs_value;
  fixed_point_multiplier_ = static_cast<float>(fixed_point_multiplier);
  vector<int8_t> fixed_point_data(centers_data.size());
  unique_ptr<VariableLengthDocidCollection> docids(
      new VariableLengthDocidCollection);
  *docids = VariableLengthDocidCollection::CreateWithEmptyDocids(
      float_centers_.size());
  for (size_t i = 0; i < centers_data.size(); ++i) {
    double fp = centers_data[i] * fixed_point_multiplier;
    if (fp > numeric_limits<int8_t>::max()) fp = numeric_limits<int8_t>::max();
    if (fp < numeric_limits<int8_t>::min()) fp = numeric_limits<int8_t>::min();
    fixed_point_data[i] = static_cast<int8_t>(fp);
  }

  fixed_point_centers_ =
      DenseDataset<int8_t>(std::move(fixed_point_data), std::move(docids));
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
    for (const float& elem : center.values_slice()) {
      center_proto->add_dimension(elem);
    }
  }

  proto->set_leaf_id(leaf_id_);
  proto->set_learned_spilling_threshold(learned_spilling_threshold_);

  for (const double& residual_stdev : residual_stdevs_) {
    proto->add_residual_stdevs(residual_stdev);
  }

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
    std::unordered_set<DatapointIndex>* union_hash) const {
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

namespace kmeans_tree_internal {

template <>
Status FindChildrenWithSpilling(
    const DatapointPtr<float>& query,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const DistanceMeasure& dist,
    const DenseDataset<int8_t>& centers, float fixed_point_multiplier,
    std::vector<pair<DatapointIndex, float>>* child_centers) {
  DCHECK_GT(centers.size(), 0);
  DCHECK(child_centers);

  std::vector<float> distances(centers.size());
  DCHECK(centers.IsDense());
  SCANN_RETURN_IF_ERROR(GetAllDistances<float>(
      dist, query, centers, fixed_point_multiplier, &distances));
  const size_t nearest_center_index = std::distance(
      distances.begin(), std::min_element(distances.begin(), distances.end()));
  const float nearest_center_distance = distances[nearest_center_index];

  TF_ASSIGN_OR_RETURN(
      float max_dist_to_consider,
      ComputeThreshold(nearest_center_distance,
                       cast_ops::DoubleToFloat(spilling_threshold),
                       spilling_type));
  FastTopNeighbors<float> top_n(
      (spilling_type == QuerySpillingConfig::NO_SPILLING) ? 1 : max_centers,
      max_dist_to_consider);
  {
    FastTopNeighbors<float>::Mutator m;
    top_n.AcquireMutator(&m);
    m.PushDistanceBlock(MakeConstSpan(distances), 0);
  }
  top_n.FinishUnsorted(child_centers);
  return OkStatus();
}

}  // namespace kmeans_tree_internal

}  // namespace scann_ops
}  // namespace tensorflow
