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



#ifndef SCANN__TREES_KMEANS_TREE_KMEANS_TREE_H_
#define SCANN__TREES_KMEANS_TREE_KMEANS_TREE_H_

#include <cmath>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree_node.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {

struct KMeansTreeSearchResult {
  const KMeansTreeNode* node;

  double distance_to_center;

  double residual_stdev;

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

  KMeansTree(KMeansTree&& rhs) = default;
  KMeansTree& operator=(KMeansTree&& rhs) = default;

  Status Train(const Dataset& training_data,
               const DistanceMeasure& training_distance, int32_t k_per_level,
               KMeansTreeTrainingOptions* training_options) override;

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
        TokenizationType tokenization_type = FLOAT,
        bool populate_residual_stdev = false) {
      TokenizationOptions result;
      result.tokenization_type = tokenization_type;
      result.populate_residual_stdev = populate_residual_stdev;
      return result;
    }

    static TokenizationOptions LearnedSpilling(
        TokenizationType tokenization_type = FLOAT,
        bool populate_residual_stdev = false) {
      auto result = NoSpilling(tokenization_type);
      result.spilling_type = LEARNED;
      result.populate_residual_stdev = populate_residual_stdev;
      return result;
    }

    static TokenizationOptions UserSpecifiedSpilling(
        QuerySpillingConfig::SpillingType user_specified_spilling_type,
        double spilling_threshold, int32_t max_spilling_centers,
        TokenizationType tokenization_type = FLOAT,
        bool populate_residual_stdev = false) {
      auto result = NoSpilling(tokenization_type);
      result.spilling_type = USER_SPECIFIED;
      result.user_specified_spilling_type = user_specified_spilling_type;
      result.spilling_threshold = spilling_threshold;
      result.max_spilling_centers = max_spilling_centers;
      result.populate_residual_stdev = populate_residual_stdev;
      return result;
    }

    SpillingType spilling_type = NONE;

    QuerySpillingConfig::SpillingType user_specified_spilling_type;
    double spilling_threshold = NAN;
    int32_t max_spilling_centers = -1;

    bool populate_residual_stdev = false;

    TokenizationType tokenization_type = FLOAT;
  };

  template <typename T>
  Status Tokenize(const DatapointPtr<T>& query, const DistanceMeasure& dist,
                  const TokenizationOptions& opts,
                  std::vector<KMeansTreeSearchResult>* result) const;

  const KMeansTreeNode* root() const { return &root_; }

  void Serialize(SerializedKMeansTree* result) const override;

  void SerializeWithoutIndices(SerializedKMeansTree* result) const override;

  int32_t n_tokens() const { return n_tokens_; }

  bool is_trained() const { return n_tokens_ > 0; }

  DatabaseSpillingConfig::SpillingType learned_spilling_type() const {
    return learned_spilling_type_;
  }

  DatapointPtr<float> CenterForToken(int32_t token) const {
    auto raw = CenterForTokenImpl(token, &root_);
    DCHECK(raw.first);
    return raw.second;
  }

  StatusOr<double> ResidualStdevForToken(int32_t token) const {
    auto raw = ResidualStdevForTokenImpl(token, &root_);
    if (!raw.first) {
      return InternalError(
          "Didn't find residual stdev. Check if compute_residual_stdev is set "
          "in the partitioning config and GmmUtils options");
    }
    return raw.second;
  }

  template <typename T>
  Status TokenizeWithLearnedSpillingThresholds(
      const DatapointPtr<T>& query, const DistanceMeasure& dist,
      std::vector<KMeansTreeSearchResult>* results) const;

 private:
  template <typename QueryType, typename CentersType>
  Status TokenizeImpl(const DatapointPtr<QueryType>& query,
                      const DistanceMeasure& dist,
                      const TokenizationOptions& opts,
                      std::vector<KMeansTreeSearchResult>* result) const;

  template <typename QueryType, typename CentersType>
  Status TokenizeWithoutSpillingImpl(
      const DatapointPtr<QueryType>& query, const DistanceMeasure& dist,
      const KMeansTreeNode* root, KMeansTreeSearchResult* result,
      bool populate_residual_stdev = false) const;

  template <typename QueryType, typename CentersType>
  Status TokenizeWithSpillingImpl(
      const DatapointPtr<QueryType>& query, const DistanceMeasure& dist,
      QuerySpillingConfig::SpillingType spilling_type,
      double spilling_threshold, int32_t max_centers,
      const KMeansTreeNode* current_node,
      std::vector<KMeansTreeSearchResult>* results,
      bool populate_residual_stdev = false) const;

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
        std::make_pair(false, fallback_value);
      }
      const int32_t idx = token - children.front().LeafId();
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

  pair<bool, double> ResidualStdevForTokenImpl(
      int32_t token, const KMeansTreeNode* node) const {
    return NodeIteratingHelper(
        token, node,
        [](const KMeansTreeNode& node, int32_t idx) -> pair<bool, double> {
          if (idx < node.residual_stdevs().size()) {
            return std::make_pair(true, node.residual_stdevs()[idx]);
          } else {
            return std::make_pair(false, std::nan(""));
          }
        },
        std::nan(""));
  }

  KMeansTreeNode root_;

  DatabaseSpillingConfig::SpillingType learned_spilling_type_ =
      DatabaseSpillingConfig::NO_SPILLING;

  int32_t max_spill_centers_ = -1;

  int32_t n_tokens_ = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(KMeansTree);
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

template <typename T>
Status KMeansTree::Tokenize(const DatapointPtr<T>& query,
                            const DistanceMeasure& dist,
                            const TokenizationOptions& opts,
                            std::vector<KMeansTreeSearchResult>* result) const {
  SCANN_RETURN_IF_ERROR(root_.CheckDimensionality(query.dimensionality()));

  Status status;

  Datapoint<float> converted_values;
  const DatapointPtr<float> query_float = ToFloat(query, &converted_values);
  if (opts.tokenization_type == FLOAT) {
    status = TokenizeImpl<float, float>(query_float, dist, opts, result);
  } else if (opts.tokenization_type == FIXED_POINT_INT8) {
    status = TokenizeImpl<float, int8_t>(query_float, dist, opts, result);
  } else {
    return InternalError(
        absl::StrCat("Invalid tokenization type:  ", opts.tokenization_type));
  }

  if (status.ok()) ZipSortBranchOptimized(result->begin(), result->end());
  return status;
}

template <typename QueryType, typename CentersType>
Status KMeansTree::TokenizeImpl(
    const DatapointPtr<QueryType>& query, const DistanceMeasure& dist,
    const TokenizationOptions& opts,
    std::vector<KMeansTreeSearchResult>* result) const {
  switch (opts.spilling_type) {
    case TokenizationOptions::NONE:
      result->resize(1);
      return TokenizeWithoutSpillingImpl<QueryType, CentersType>(
          query, dist, &root_, result->data(), opts.populate_residual_stdev);
    case TokenizationOptions::LEARNED:
      return TokenizeWithSpillingImpl<QueryType, CentersType>(
          query, dist,
          static_cast<QuerySpillingConfig::SpillingType>(
              learned_spilling_type_),
          NAN, max_spill_centers_, &root_, result,
          opts.populate_residual_stdev);
    case TokenizationOptions::USER_SPECIFIED:
      return TokenizeWithSpillingImpl<QueryType, CentersType>(
          query, dist, opts.user_specified_spilling_type,
          opts.spilling_threshold, opts.max_spilling_centers, &root_, result,
          opts.populate_residual_stdev);
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

template <typename QueryType, typename CentersType>
Status KMeansTree::TokenizeWithoutSpillingImpl(
    const DatapointPtr<QueryType>& query, const DistanceMeasure& dist,
    const KMeansTreeNode* root, KMeansTreeSearchResult* result,
    bool populate_residual_stdev) const {
  CHECK(result);
  if (root->IsLeaf()) {
    result->node = root;
    result->distance_to_center = NAN;
    return OkStatus();
  }

  size_t nearest_center_index;
  double nearest_center_distance = numeric_limits<double>::max();
  const DenseDataset<CentersType>& centers =
      root->GetCentersByTemplateType<CentersType>();
  std::vector<double> distances(centers.size());
  SCANN_RETURN_IF_ERROR(kmeans_tree_internal::GetAllDistances(
      dist, query, centers, root->fixed_point_multiplier_, &distances));
  auto min_it = std::min_element(distances.begin(), distances.end());
  nearest_center_distance = *min_it;
  nearest_center_index = min_it - distances.begin();
  FreeBackingStorage(&distances);

  if (root->Children()[nearest_center_index].IsLeaf()) {
    result->node = &root->Children()[nearest_center_index];
    result->distance_to_center = nearest_center_distance;
    result->residual_stdev =
        (populate_residual_stdev &&
         nearest_center_index < root->residual_stdevs().size())
            ? result->residual_stdev =
                  root->residual_stdevs()[nearest_center_index]
            : 1.0;
    return OkStatus();
  }

  return TokenizeWithoutSpillingImpl<QueryType, CentersType>(
      query, dist, &root->Children()[nearest_center_index], result);
}

template <typename QueryType, typename CentersType>
Status KMeansTree::TokenizeWithSpillingImpl(
    const DatapointPtr<QueryType>& query, const DistanceMeasure& dist,
    QuerySpillingConfig::SpillingType spilling_type, double spilling_threshold,
    int32_t max_centers, const KMeansTreeNode* current_node,
    std::vector<KMeansTreeSearchResult>* results,
    bool populate_residual_stdev) const {
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
  const DenseDataset<CentersType>& current_node_centers =
      current_node->GetCentersByTemplateType<CentersType>();
  Status status =
      kmeans_tree_internal::FindChildrenWithSpilling<QueryType, CentersType>(
          query, spilling_type, possibly_learned_spilling_threshold,
          max_centers, dist, current_node_centers,
          current_node->fixed_point_multiplier_, &children_to_search);
  if (!status.ok()) return status;
  for (const auto& elem : children_to_search) {
    const int32_t child_index = elem.first;
    const float distance_to_child_center = elem.second;
    if (current_node->Children()[child_index].IsLeaf()) {
      KMeansTreeSearchResult result;
      result.node = &current_node->Children()[child_index];
      result.distance_to_center = distance_to_child_center;
      result.residual_stdev =
          (populate_residual_stdev &&
           child_index < current_node->residual_stdevs().size())
              ? result.residual_stdev =
                    current_node->residual_stdevs()[child_index]
              : 1.0;
      results->push_back(result);
    } else {
      status = TokenizeWithSpillingImpl<QueryType, CentersType>(
          query, dist, spilling_type, spilling_threshold, max_centers,
          &current_node->Children()[child_index], results,
          populate_residual_stdev);
      if (!status.ok()) return status;
    }
  }

  ZipSortBranchOptimized(results->begin(), results->end());
  return OkStatus();
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
