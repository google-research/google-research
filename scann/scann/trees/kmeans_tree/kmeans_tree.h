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



#ifndef SCANN_TREES_KMEANS_TREE_KMEANS_TREE_H_
#define SCANN_TREES_KMEANS_TREE_KMEANS_TREE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree_node.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

struct KMeansTreeSearchResult {
  const KMeansTreeNode* node = nullptr;

  double distance_to_center = NAN;

  bool operator<(const KMeansTreeSearchResult& rhs) const;
};

class KMeansTreeTrainerInterface {
 public:
  virtual ~KMeansTreeTrainerInterface() {}

  virtual Status Train(const Dataset& training_data,
                       const DistanceMeasure& training_distance,
                       int32_t k_per_level,
                       KMeansTreeTrainingOptions* training_options) = 0;

  virtual void Serialize(SerializedKMeansTree* result) const = 0;

  virtual void SerializeWithoutIndices(SerializedKMeansTree* result) const = 0;
};

template <class C>
class KMeansTreeTokenizerInterface {
 public:
  virtual ~KMeansTreeTokenizerInterface() {}

  template <typename T>
  Status TokenizeWithLearnedSpillingThresholds(
      const DatapointPtr<T>& query, const DistanceMeasure& dist,
      std::vector<KMeansTreeSearchResult>* results) const {
    return down_cast<const C*>(this)->TokenizeWithLearnedSpillingThresholds(
        query, dist, results);
  }
};

class KMeansTree final : public KMeansTreeTrainerInterface,
                         public KMeansTreeTokenizerInterface<KMeansTree> {
 public:
  KMeansTree();

  explicit KMeansTree(const SerializedKMeansTree& serialized);

  static KMeansTree CreateFlat(DenseDataset<float> centers);

  KMeansTree(const KMeansTree&) = delete;
  KMeansTree& operator=(const KMeansTree&) = delete;

  KMeansTree(KMeansTree&& rhs) = default;
  KMeansTree& operator=(KMeansTree&& rhs) = default;

  Status Train(const Dataset& training_data,
               const DistanceMeasure& training_distance, int32_t k_per_level,
               KMeansTreeTrainingOptions* training_options) override;

  template <typename T>
  Status ApplyAvq(const DenseDataset<T>& dataset,
                  ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
                  float avq_eta, ThreadPool* pool_or_null = nullptr);

  enum TokenizationType {

    FLOAT = 1,

    FIXED_POINT_INT8 = 2
  };

  struct TokenizationOptions {
    enum SpillingType {

      NONE,

      LEARNED,

      USER_SPECIFIED
    };

    static TokenizationOptions NoSpilling(
        TokenizationType tokenization_type = FLOAT) {
      TokenizationOptions result;
      result.tokenization_type = tokenization_type;
      return result;
    }

    static TokenizationOptions LearnedSpilling(
        TokenizationType tokenization_type = FLOAT) {
      auto result = NoSpilling(tokenization_type);
      result.spilling_type = LEARNED;
      return result;
    }

    static TokenizationOptions UserSpecifiedSpilling(
        QuerySpillingConfig::SpillingType user_specified_spilling_type,
        double spilling_threshold, int32_t max_spilling_centers,
        TokenizationType tokenization_type = FLOAT) {
      auto result = NoSpilling(tokenization_type);
      result.spilling_type = USER_SPECIFIED;
      result.user_specified_spilling_type = user_specified_spilling_type;
      result.spilling_threshold = spilling_threshold;
      result.max_spilling_centers = max_spilling_centers;
      return result;
    }

    SpillingType spilling_type = NONE;

    QuerySpillingConfig::SpillingType user_specified_spilling_type;
    double spilling_threshold = NAN;
    int32_t max_spilling_centers = -1;

    TokenizationType tokenization_type = FLOAT;

    int32_t num_tokenized_branch = 1;
  };

  template <typename T, typename ResT>
  Status Tokenize(const DatapointPtr<T>& query, const DistanceMeasure& dist,
                  const TokenizationOptions& opts,
                  std::vector<ResT>* result) const;

  const KMeansTreeNode* root() const { return &root_; }

  void Serialize(SerializedKMeansTree* result) const override;

  void SerializeWithoutIndices(SerializedKMeansTree* result) const override;

  void CheckIfFlat();

  int32_t n_tokens() const { return n_tokens_; }

  bool is_flat() const { return is_flat_; }

  bool is_trained() const { return n_tokens_ > 0; }

  DatabaseSpillingConfig::SpillingType learned_spilling_type() const {
    return learned_spilling_type_;
  }

  DatapointPtr<float> CenterForToken(int32_t token) const {
    auto raw = CenterForTokenImpl(token, &root_);
    DCHECK(raw.first);
    return raw.second;
  }

  template <typename T>
  Status TokenizeWithLearnedSpillingThresholds(
      const DatapointPtr<T>& query, const DistanceMeasure& dist,
      std::vector<KMeansTreeSearchResult>* results) const;

 private:
  template <typename CentersType, typename ResT>
  Status TokenizeImpl(const DatapointPtr<float>& query,
                      const DistanceMeasure& dist,
                      const TokenizationOptions& opts,
                      std::vector<ResT>* result) const;

  template <typename CentersType>
  Status TokenizeWithoutSpillingImpl(const DatapointPtr<float>& query,
                                     const DistanceMeasure& dist,
                                     int32_t num_tokenized_branch,
                                     const KMeansTreeNode* root,
                                     KMeansTreeSearchResult* result) const;

  template <typename CentersType>
  Status TokenizeWithSpillingImpl(
      const DatapointPtr<float>& query, const DistanceMeasure& dist,
      QuerySpillingConfig::SpillingType spilling_type,
      double spilling_threshold, int32_t max_centers,
      const KMeansTreeNode* current_node,
      std::vector<KMeansTreeSearchResult>* results) const;

  template <typename CentersType>
  Status TokenizeWithSpillingImpl(
      const DatapointPtr<float>& query, const DistanceMeasure& dist,
      QuerySpillingConfig::SpillingType spilling_type,
      double spilling_threshold, int32_t max_centers,
      const KMeansTreeNode* current_node,
      std::vector<pair<DatapointIndex, float>>* results) const;

  template <typename CallbackType, typename RetValueType>
  pair<bool, RetValueType> NodeIteratingHelper(
      int32_t token, const KMeansTreeNode* node, CallbackType success_callback,
      const RetValueType& fallback_value) const {
    DCHECK(!node->IsLeaf());
    DCHECK_LT(token, n_tokens_);
    ConstSpan<KMeansTreeNode> children = node->Children();
    DCHECK(!children.empty());

    const bool is_all_leaf_range =
        children.front().IsLeaf() && children.back().IsLeaf() &&
        children.back().LeafId() - children.front().LeafId() + 1 ==
            children.size();
    if (is_all_leaf_range) {
      if (children.front().LeafId() > token ||
          children.back().LeafId() < token) {
        return std::make_pair(false, fallback_value);
      }
      const int32_t idx = token - children.front().LeafId();
      DCHECK_LT(idx, children.size())
          << token << " " << children.front().LeafId() << " "
          << children.back().LeafId();
      DCHECK_EQ(children[idx].LeafId(), token);

      return success_callback(*node, idx);
    }

    for (size_t i = 0; i < children.size(); ++i) {
      if (children[i].IsLeaf()) {
        if (children[i].LeafId() == token) {
          return success_callback(*node, i);
        }
      } else {
        auto recursion_result = NodeIteratingHelper(
            token, &children[i], success_callback, fallback_value);
        if (recursion_result.first) return recursion_result;
      }
    }
    return std::make_pair(false, fallback_value);
  }

  pair<bool, DatapointPtr<float>> CenterForTokenImpl(
      int32_t token, const KMeansTreeNode* node) const {
    return NodeIteratingHelper(
        token, node,
        [](const KMeansTreeNode& node,
           int32_t idx) -> pair<bool, DatapointPtr<float>> {
          return std::make_pair(true, node.Centers()[idx]);
        },
        DatapointPtr<float>());
  }

  KMeansTreeNode root_;

  DatabaseSpillingConfig::SpillingType learned_spilling_type_ =
      DatabaseSpillingConfig::NO_SPILLING;

  int32_t max_spill_centers_ = -1;

  int32_t n_tokens_ = -1;

  bool is_flat_ = false;
};

inline bool KMeansTreeSearchResult::operator<(
    const KMeansTreeSearchResult& rhs) const {
  DCHECK(node);
  DCHECK(rhs.node);

  const bool is_eq_or_nan =
      (distance_to_center == rhs.distance_to_center ||
       std::isunordered(distance_to_center, rhs.distance_to_center));

  if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
    return node->LeafId() < rhs.node->LeafId();
  }
  return distance_to_center < rhs.distance_to_center;
}

template <typename T, typename ResT>
Status KMeansTree::Tokenize(const DatapointPtr<T>& query,
                            const DistanceMeasure& dist,
                            const TokenizationOptions& opts,
                            std::vector<ResT>* result) const {
  static_assert(std::is_same_v<ResT, KMeansTreeSearchResult> ||
                std::is_same_v<ResT, pair<DatapointIndex, float>>);
  SCANN_RETURN_IF_ERROR(root_.CheckDimensionality(query.dimensionality()));

  Datapoint<float> converted_values;
  const DatapointPtr<float> query_float = ToFloat(query, &converted_values);
  if (opts.tokenization_type == FLOAT) {
    return TokenizeImpl<float, ResT>(query_float, dist, opts, result);
  } else if (opts.tokenization_type == FIXED_POINT_INT8) {
    return TokenizeImpl<int8_t, ResT>(query_float, dist, opts, result);
  } else {
    return InternalError(
        absl::StrCat("Invalid tokenization type:  ", opts.tokenization_type));
  }
}

template <typename CentersType, typename ResT>
Status KMeansTree::TokenizeImpl(const DatapointPtr<float>& query,
                                const DistanceMeasure& dist,
                                const TokenizationOptions& opts,
                                std::vector<ResT>* result) const {
  static_assert(std::is_same_v<ResT, KMeansTreeSearchResult> ||
                std::is_same_v<ResT, pair<DatapointIndex, float>>);
  switch (opts.spilling_type) {
    case TokenizationOptions::NONE:
      result->resize(1);
      if constexpr (std::is_same_v<ResT, KMeansTreeSearchResult>) {
        return TokenizeWithoutSpillingImpl<CentersType>(
            query, dist, opts.num_tokenized_branch, &root_, result->data());
      } else {
        KMeansTreeSearchResult kmeans_res;
        SCANN_RETURN_IF_ERROR(TokenizeWithoutSpillingImpl<CentersType>(
            query, dist, opts.num_tokenized_branch, &root_, &kmeans_res));
        DCHECK(kmeans_res.node != nullptr);
        result->front() = {kmeans_res.node->LeafId(),
                           kmeans_res.distance_to_center};
        return OkStatus();
      }
    case TokenizationOptions::LEARNED:
      return TokenizeWithSpillingImpl<CentersType>(
          query, dist,
          static_cast<QuerySpillingConfig::SpillingType>(
              learned_spilling_type_),
          NAN, max_spill_centers_, &root_, result);
    case TokenizationOptions::USER_SPECIFIED:
      return TokenizeWithSpillingImpl<CentersType>(
          query, dist, opts.user_specified_spilling_type,
          opts.spilling_threshold, opts.max_spilling_centers, &root_, result);
    default:
      return InternalError(
          absl::StrCat("Invalid spilling type:  ", opts.spilling_type));
  }
}

template <typename T>
Status KMeansTree::TokenizeWithLearnedSpillingThresholds(
    const DatapointPtr<T>& query, const DistanceMeasure& dist,
    std::vector<KMeansTreeSearchResult>* results) const {
  return Tokenize(query, dist, TokenizationOptions::LearnedSpilling(FLOAT),
                  results);
}

template <typename CentersType>
Status KMeansTree::TokenizeWithoutSpillingImpl(
    const DatapointPtr<float>& query, const DistanceMeasure& dist,
    int32_t num_tokenized_branch, const KMeansTreeNode* root,
    KMeansTreeSearchResult* result) const {
  CHECK(result);
  if (root->IsLeaf()) {
    result->node = root;
    result->distance_to_center = NAN;
    return OkStatus();
  }
  const DenseDataset<CentersType>& centers =
      root->GetCentersByTemplateType<CentersType>();
  std::vector<double> distances(centers.size());
  if (std::is_same_v<CentersType, int8_t>) {
    SCANN_RETURN_IF_ERROR(root->GetAllDistancesInt8(dist, query, &distances));
  } else {
    root->GetAllDistancesFloatingPoint(dist, query, &distances);
  }
  if (root->Children().empty()) {
    return OkStatus();
  }

  if (root->Children()[0].IsLeaf() || num_tokenized_branch <= 1) {
    size_t nearest_center_index;
    double nearest_center_distance = numeric_limits<double>::max();
    auto min_it = std::min_element(distances.begin(), distances.end());
    nearest_center_distance = *min_it;
    nearest_center_index = min_it - distances.begin();
    FreeBackingStorage(&distances);
    if (root->Children()[nearest_center_index].IsLeaf()) {
      result->node = &root->Children()[nearest_center_index];
      result->distance_to_center = nearest_center_distance;
      return OkStatus();
    } else {
      return TokenizeWithoutSpillingImpl<CentersType>(
          query, dist, num_tokenized_branch,
          &root->Children()[nearest_center_index], result);
    }
  } else {
    std::vector<std::pair<int, double>> index_distances;
    double nearest_center_distance = numeric_limits<double>::max();
    for (const auto& [index, distance] : Enumerate(distances)) {
      index_distances.push_back({index, distance});
    }
    FreeBackingStorage(&distances);
    SortBranchOptimized(index_distances.begin(), index_distances.end(),
                        DistanceComparatorBranchOptimized());
    for (int branch_id = 0;
         branch_id < std::min(num_tokenized_branch,
                              static_cast<int>(root->Children().size()));
         branch_id++) {
      const auto& child = root->Children()[index_distances[branch_id].first];
      KMeansTreeSearchResult child_result;
      auto status = TokenizeWithoutSpillingImpl<CentersType>(
          query, dist, num_tokenized_branch, &child, &child_result);
      if (status.ok()) {
        DCHECK_NE(child_result.node, nullptr);
        if (child.IsLeaf()) {
          DCHECK_EQ(child_result.node, &child);
          if (result->node == nullptr) {
            result->node = &child;

            result->distance_to_center = child_result.distance_to_center;
          }
        }
        if (child_result.distance_to_center < nearest_center_distance) {
          nearest_center_distance = child_result.distance_to_center;
          result->node = child_result.node;
          result->distance_to_center = nearest_center_distance;
        }
      } else {
        FreeBackingStorage(&index_distances);
        return status;
      }
    }
    FreeBackingStorage(&index_distances);
    return OkStatus();
  }
}

template <typename CentersType>
Status KMeansTree::TokenizeWithSpillingImpl(
    const DatapointPtr<float>& query, const DistanceMeasure& dist,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const KMeansTreeNode* current_node,
    std::vector<KMeansTreeSearchResult>* results) const {
  DCHECK(results);
  DCHECK(current_node);

  if (current_node->IsLeaf()) {
    KMeansTreeSearchResult result;
    result.node = current_node;
    result.distance_to_center = NAN;
    results->push_back(result);
    return OkStatus();
  }

  const double possibly_learned_spilling_threshold =
      (std::isnan(spilling_threshold))
          ? current_node->learned_spilling_threshold()
          : spilling_threshold;

  std::vector<pair<DatapointIndex, float>> children_to_search;
  SCANN_RETURN_IF_ERROR(
      (current_node->FindChildrenWithSpilling<float, CentersType>(
          query, spilling_type, possibly_learned_spilling_threshold,
          max_centers, dist, &children_to_search)));
  for (const auto& elem : children_to_search) {
    const int32_t child_index = elem.first;
    const float distance_to_child_center = elem.second;
    if (current_node->Children()[child_index].IsLeaf()) {
      KMeansTreeSearchResult result;
      result.node = &current_node->Children()[child_index];
      result.distance_to_center = distance_to_child_center;
      results->push_back(result);
    } else {
      SCANN_RETURN_IF_ERROR((TokenizeWithSpillingImpl<CentersType>(
          query, dist, spilling_type, spilling_threshold, max_centers,
          &current_node->Children()[child_index], results)));
    }
  }

  ZipSortBranchOptimized(results->begin(), results->end());
  return OkStatus();
}

template <typename CentersType>
Status KMeansTree::TokenizeWithSpillingImpl(
    const DatapointPtr<float>& query, const DistanceMeasure& dist,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const KMeansTreeNode* current_node,
    std::vector<pair<DatapointIndex, float>>* results) const {
  if (ABSL_PREDICT_TRUE(is_flat_)) {
    const double possibly_learned_spilling_threshold =
        (std::isnan(spilling_threshold))
            ? current_node->learned_spilling_threshold()
            : spilling_threshold;
    SCANN_RETURN_IF_ERROR(
        (current_node->FindChildrenWithSpilling<float, CentersType>(
            query, spilling_type, possibly_learned_spilling_threshold,
            max_centers, dist, results)));
    ZipSortBranchOptimized(DistanceComparatorBranchOptimized(),
                           results->begin(), results->end());
    return OkStatus();
  } else {
    vector<KMeansTreeSearchResult> full_results;
    SCANN_RETURN_IF_ERROR(TokenizeWithSpillingImpl<CentersType>(
        query, dist, spilling_type, spilling_threshold, max_centers,
        current_node, &full_results));
    results->resize(full_results.size());
    for (const auto& [i, full_res] : Enumerate(full_results)) {
      (*results)[i] = {full_res.node->LeafId(), full_res.distance_to_center};
    }
    return OkStatus();
  }
}

template <typename T>
Status KMeansTree::ApplyAvq(
    const DenseDataset<T>& dataset,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token, float avq_eta,
    ThreadPool* pool_or_null) {
  const bool need_reenable_fixed8 = !root_.fixed_point_centers_.empty();
  SCANN_RETURN_IF_ERROR(
      root_.ApplyAvq(dataset, datapoints_by_token, avq_eta, pool_or_null));
  root_.PopulateCurNodeCenters();
  if (need_reenable_fixed8) {
    root_.CreateFixedPointCenters();
  }

  return OkStatus();
}

}  // namespace research_scann

#endif
