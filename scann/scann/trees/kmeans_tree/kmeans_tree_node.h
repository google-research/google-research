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

#ifndef SCANN__TREES_KMEANS_TREE_KMEANS_TREE_NODE_H_
#define SCANN__TREES_KMEANS_TREE_KMEANS_TREE_NODE_H_

#include <type_traits>

#include "absl/flags/flag.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/oss_wrappers/scann_castops.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace scann_ops {

class KMeansTreeNode {
 public:
  KMeansTreeNode();

  explicit KMeansTreeNode(int32_t leaf_id) { leaf_id_ = leaf_id; }

  void Reset();

  bool IsLeaf() const { return children_.size() == 0; }

  int32_t LeafId() const { return leaf_id_; }

  const DenseDataset<float>& Centers() const { return FloatCenters(); }

  const DenseDataset<float>& FloatCenters() const { return float_centers_; }

  const std::vector<KMeansTreeNode>& Children() const { return children_; }

  const std::vector<DatapointIndex>& indices() const { return indices_; }

  const std::vector<double>& residual_stdevs() const {
    return residual_stdevs_;
  }

  void UnionIndices(std::vector<DatapointIndex>* result) const;

  double learned_spilling_threshold() const {
    return learned_spilling_threshold_;
  }

  DatapointPtr<float> cur_node_center() const { return cur_node_center_; }

 private:
  friend class KMeansTree;

  Status Train(const Dataset& training_data, std::vector<DatapointIndex> subset,
               const DistanceMeasure& training_distance, int32_t k_per_level,
               int32_t current_level, KMeansTreeTrainingOptions* opts);

  void CreateFloatCenters();

  void CreateFixedPointCenters();

  void BuildFromProto(const SerializedKMeansTree::Node& proto);

  void CopyToProto(SerializedKMeansTree::Node* proto, bool with_indices) const;

  Status CheckDimensionality(DimensionIndex query_dims) const;

  int32_t NumberLeaves(int32_t m);

  void PopulateCurNodeCenters();

  template <typename Real>
  const DenseDataset<Real>& GetCentersByTemplateType() const;

  void UnionIndicesImpl(std::unordered_set<DatapointIndex>* union_hash) const;

  DenseDataset<float> float_centers_;

  DenseDataset<int8_t> fixed_point_centers_;

  float fixed_point_multiplier_ = NAN;

  std::vector<DatapointIndex> indices_ = {};

  std::vector<KMeansTreeNode> children_ = {};

  std::vector<double> residual_stdevs_ = {};

  double learned_spilling_threshold_ = numeric_limits<double>::quiet_NaN();

  int32_t leaf_id_ = -1;

  DatapointPtr<float> cur_node_center_;
};

template <>
inline const DenseDataset<float>&
KMeansTreeNode::GetCentersByTemplateType<float>() const {
  return float_centers_;
}

template <>
inline const DenseDataset<int8_t>&
KMeansTreeNode::GetCentersByTemplateType<int8_t>() const {
  return fixed_point_centers_;
}

namespace kmeans_tree_internal {

template <typename Real>
inline Status GetAllDistances(const DistanceMeasure& dist,
                              const DatapointPtr<Real>& query,
                              const DenseDataset<Real>& centers,
                              float fixed_point_multiplier,
                              std::vector<double>* distances) {
  if (query.IsDense()) {
    DenseDistanceOneToMany<Real, double>(dist, query, centers,
                                         MakeMutableSpan(*distances));
  } else {
    for (size_t i = 0; i < centers.size(); ++i) {
      (*distances)[i] = dist.GetDistance(query, centers[i]);
    }
  }

  return OkStatus();
}

template <typename OutT = double>
inline Status GetAllDistances(const DistanceMeasure& dist,
                              const DatapointPtr<float>& query,
                              const DenseDataset<int8_t>& centers,
                              float fixed_point_multiplier,
                              std::vector<OutT>* distances) {
  if (dist.specially_optimized_distance_tag() != DistanceMeasure::DOT_PRODUCT) {
    return InvalidArgumentError(
        "Fixed-point tokenization in K-Means trees currently works only for "
        "dot-product distance.");
  }

  const float inv_fixed_point_multiplier = 1.0 / fixed_point_multiplier;
  Datapoint<float> inv_adjusted;
  CopyToDatapoint(query, &inv_adjusted);
  for (float& f : *inv_adjusted.mutable_values()) {
    f *= inv_fixed_point_multiplier;
  }

  DenseDotProductDistanceOneToManyInt8Float(inv_adjusted.ToPtr(), centers,
                                            MakeMutableSpan(*distances));
  return OkStatus();
}

template <typename Real>
inline StatusOr<Real> ComputeThreshold(
    const Real nearest_center_distance, const Real spilling_threshold,
    QuerySpillingConfig::SpillingType spilling_type) {
  Real max_dist_to_consider;
  if (std::isnan(spilling_threshold)) {
    spilling_type = QuerySpillingConfig::NO_SPILLING;
  }

  switch (spilling_type) {
    case QuerySpillingConfig::NO_SPILLING:
      max_dist_to_consider = nearest_center_distance;
      break;
    case QuerySpillingConfig::MULTIPLICATIVE:
      max_dist_to_consider = nearest_center_distance * spilling_threshold;
      break;
    case QuerySpillingConfig::ADDITIVE:
      max_dist_to_consider = nearest_center_distance + spilling_threshold;
      break;
    case QuerySpillingConfig::ABSOLUTE_DISTANCE:

      max_dist_to_consider =
          std::max(spilling_threshold, nearest_center_distance);
      break;
    case QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS:
      max_dist_to_consider = numeric_limits<Real>::infinity();
      break;
    default:
      return InvalidArgumentError("Unknown spilling type.");
  }
  return max_dist_to_consider;
}

template <typename Real, typename DataType>
Status FindChildrenWithSpilling(
    const DatapointPtr<Real>& query,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const DistanceMeasure& dist,
    const DenseDataset<DataType>& centers, float fixed_point_multiplier,
    std::vector<pair<DatapointIndex, float>>* child_centers) {
  DCHECK_GT(centers.size(), 0);
  DCHECK(child_centers);

  std::vector<double> distances(centers.size());
  DCHECK(centers.IsDense());
  SCANN_RETURN_IF_ERROR(GetAllDistances(dist, query, centers,
                                        fixed_point_multiplier, &distances));
  const size_t nearest_center_index = std::distance(
      distances.begin(), std::min_element(distances.begin(), distances.end()));
  const double nearest_center_distance = distances[nearest_center_index];

  TF_ASSIGN_OR_RETURN(double max_dist_to_consider,
                      ComputeThreshold(nearest_center_distance,
                                       spilling_threshold, spilling_type));
  using cast_ops::DoubleToFloat;
  FastTopNeighbors<float> top_n(
      (spilling_type == QuerySpillingConfig::NO_SPILLING) ? 1 : max_centers,
      DoubleToFloat(max_dist_to_consider));
  {
    FastTopNeighbors<float>::Mutator m;
    top_n.AcquireMutator(&m);
    double eps = max_dist_to_consider;

    for (size_t i = 0; i < distances.size(); ++i) {
      if (distances[i] > eps) continue;
      if (m.Push(i, DoubleToFloat(distances[i]))) {
        m.GarbageCollect();
        eps = m.epsilon();
      }
    }
  }
  top_n.FinishUnsorted(child_centers);
  return OkStatus();
}

template <>
Status FindChildrenWithSpilling(
    const DatapointPtr<float>& query,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const DistanceMeasure& dist,
    const DenseDataset<int8_t>& centers, float fixed_point_multiplier,
    std::vector<pair<DatapointIndex, float>>* child_centers);

}  // namespace kmeans_tree_internal

}  // namespace scann_ops
}  // namespace tensorflow

#endif
