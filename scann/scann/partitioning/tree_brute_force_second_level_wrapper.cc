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

#include "scann/partitioning/tree_brute_force_second_level_wrapper.h"

#include <memory>

#include "absl/log/check.h"
#include "scann/brute_force/bfloat16_brute_force.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/kmeans_tree_partitioner.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"

namespace research_scann {

template <typename T>
TreeBruteForceSecondLevelWrapper<T>::TreeBruteForceSecondLevelWrapper(
    unique_ptr<KMeansTreeLikePartitioner<T>> base)
    : base_(std::move(base)) {}

template <typename T>
Status TreeBruteForceSecondLevelWrapper<T>::CreatePartitioning(
    const BottomUpTopLevelPartitioner& config,
    std::optional<SerializedKMeansTreePartitioner> serialized) {
  SCANN_ASSIGN_OR_RETURN(top_level_,
                         CreateTopLevel(*base_, config, std::move(serialized)));
  config_ = config;
  return OkStatus();
}

template <typename T>
Status TreeBruteForceSecondLevelWrapper<T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<pair<DatapointIndex, float>>* result) const {
  if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    return base_->TokensForDatapointWithSpilling(dptr, max_centers_override,
                                                 result);
  } else {
    const auto max_centers = max_centers_override > 0
                                 ? max_centers_override
                                 : base_->query_spilling_max_centers();
    SearchParameters params;
    params.set_pre_reordering_num_neighbors(max_centers);
    if (base_->query_spilling_type() ==
        QuerySpillingConfig::ABSOLUTE_DISTANCE) {
      params.set_pre_reordering_epsilon(base_->query_spilling_threshold());
    } else {
      params.set_pre_reordering_epsilon(numeric_limits<float>::infinity());
    }

    auto try_ensure_one_center = [&](DatapointPtr<float> dptr,
                                     absl::Status orig_status) {
      bool at_least_one_center = base_->query_spilling_type() ==
                                 QuerySpillingConfig::ABSOLUTE_DISTANCE;

      if (!orig_status.ok() || !result->empty() || !at_least_one_center) {
        return orig_status;
      }

      params.set_pre_reordering_epsilon(numeric_limits<float>::infinity());
      params.set_pre_reordering_num_neighbors(1);
      return top_level_->FindNeighbors(dptr, params, result);
    };

    if constexpr (std::is_same_v<T, float>) {
      auto st = top_level_->FindNeighbors(dptr, params, result);
      return try_ensure_one_center(dptr, st);
    } else {
      Datapoint<float> float_dp;
      auto float_dp_ptr = MaybeConvertDatapoint(dptr, &float_dp);
      auto st = top_level_->FindNeighbors(float_dp_ptr, params, result);
      return try_ensure_one_center(float_dp_ptr, st);
    }
  }
}

namespace {

template <typename T>
StatusOrPtr<TreeXHybridSMMD<float>> CreateTopLevelSearcher(
    const KMeansTreeLikePartitioner<T>& base,
    const BottomUpTopLevelPartitioner& config,
    vector<std::vector<DatapointIndex>> token_to_datapoints) {
  auto result = make_unique<TreeXHybridSMMD<float>>(
      MakeDummyShared(&base.LeafCenters()), nullptr, config.num_centroids(),
      numeric_limits<float>::infinity());
  SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
      std::move(token_to_datapoints),
      [&](shared_ptr<TypedDataset<float>> dataset_partition,
          shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
          int32_t) -> StatusOr<unique_ptr<SingleMachineSearcherBase<float>>> {
        SCANN_RET_CHECK(!hashed_dataset_partition);
        auto dataset_dense =
            std::dynamic_pointer_cast<const DenseDataset<float>>(
                dataset_partition);
        SCANN_RET_CHECK(dataset_dense != nullptr);

        const int num_neighbors =
            std::max<int>(1, base.query_spilling_max_centers());
        if (config.quantization() == BottomUpTopLevelPartitioner::FIXED8) {
          return make_unique<ScalarQuantizedBruteForceSearcher>(
              base.query_tokenization_distance(), dataset_dense, num_neighbors,
              numeric_limits<float>::infinity(),
              ScalarQuantizedBruteForceSearcher::Options{
                  .noise_shaping_threshold = config.noise_shaping_threshold()});
        } else if (config.quantization() ==
                   BottomUpTopLevelPartitioner::BFLOAT16) {
          return make_unique<Bfloat16BruteForceSearcher>(
              base.query_tokenization_distance(), dataset_dense, num_neighbors,
              numeric_limits<float>::infinity(),
              config.noise_shaping_threshold());
        } else {
          return make_unique<BruteForceSearcher<float>>(
              base.query_tokenization_distance(), dataset_dense, num_neighbors,
              numeric_limits<float>::infinity());
        }
      }));
  if (config.soar().enabled()) {
    result->set_spilling_overretrieve_factor(
        config.soar().overretrieve_factor());
  }
  return result;
}
}  // namespace

template <typename T>
unique_ptr<Partitioner<T>> TreeBruteForceSecondLevelWrapper<T>::Clone() const {
  unique_ptr<KMeansTreeLikePartitioner<T>> cloned_base(
      dynamic_cast<KMeansTreeLikePartitioner<T>*>(base_->Clone().release()));
  QCHECK(cloned_base != nullptr);
  cloned_base->set_tokenization_mode(this->tokenization_mode());
  auto cloned_query_tokenizer = top_level_->query_tokenizer()->Clone();
  cloned_query_tokenizer->set_tokenization_mode(UntypedPartitioner::QUERY);
  auto top_level_clone =
      CreateTopLevelSearcher(*cloned_base, config_,
                             {top_level_->datapoints_by_token().begin(),
                              top_level_->datapoints_by_token().end()})
          .value();
  top_level_clone->set_query_tokenizer(std::move(cloned_query_tokenizer));
  auto result =
      make_unique<TreeBruteForceSecondLevelWrapper<T>>(std::move(cloned_base));
  result->top_level_ = std::move(top_level_clone);
  result->config_ = config_;
  result->set_tokenization_mode(this->tokenization_mode());
  return result;
}

template <typename T>
Status
TreeBruteForceSecondLevelWrapper<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<std::vector<pair<DatapointIndex, float>>> results,
    ThreadPool* pool) const {
  if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    return base_->TokensForDatapointWithSpillingBatched(
        queries, max_centers_override, results, pool);
  }

  if (ABSL_PREDICT_FALSE(queries.IsSparse())) {
    for (size_t i : IndicesOf(queries)) {
      SCANN_RETURN_IF_ERROR(TokensForDatapointWithSpilling(
          queries[i],
          max_centers_override.empty() ? 0 : max_centers_override[i],
          &results[i]));
    }
  }

  vector<SearchParameters> params(queries.size());
  for (size_t i : IndicesOf(params)) {
    const auto max_centers = max_centers_override.empty()
                                 ? base_->query_spilling_max_centers()
                                 : max_centers_override[i];
    params[i].set_pre_reordering_num_neighbors(max_centers);
    if (base_->query_spilling_type() ==
        QuerySpillingConfig::ABSOLUTE_DISTANCE) {
      params[i].set_pre_reordering_epsilon(base_->query_spilling_threshold());
    } else {
      params[i].set_pre_reordering_epsilon(numeric_limits<float>::infinity());
    }
  }

  auto try_ensure_one_center = [&](const TypedDataset<float>& queries,
                                   absl::Status orig_status) -> absl::Status {
    bool at_least_one_center =
        base_->query_spilling_type() == QuerySpillingConfig::ABSOLUTE_DISTANCE;
    if (!orig_status.ok() || !at_least_one_center) return orig_status;

    for (size_t i : IndicesOf(queries)) {
      auto& result = results[i];
      DatapointPtr<float> query = queries[i];
      auto& param = params[i];

      if (!result.empty()) continue;

      param.set_pre_reordering_epsilon(numeric_limits<float>::infinity());
      param.set_pre_reordering_num_neighbors(1);
      SCANN_RETURN_IF_ERROR(top_level_->FindNeighbors(query, param, &result));
    }
    return OkStatus();
  };

  if constexpr (std::is_same_v<T, float>) {
    auto st = top_level_->FindNeighborsBatchedNoSortNoExactReorder(
        queries, params, results);
    return try_ensure_one_center(queries, st);
  } else {
    DenseDataset<float> float_queries;
    down_cast<const DenseDataset<T>*>(&queries)->ConvertType(&float_queries);
    auto st = top_level_->FindNeighborsBatchedNoSortNoExactReorder(
        float_queries, params, results);
    return try_ensure_one_center(float_queries, st);
  }
}

template <typename T>
Status TreeBruteForceSecondLevelWrapper<T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& query, std::vector<int32_t>* result) const {
  vector<pair<DatapointIndex, float>> with_dists;
  SCANN_RETURN_IF_ERROR(TokensForDatapointWithSpilling(query, 0, &with_dists));
  result->clear();
  result->reserve(with_dists.size());
  for (const auto& with_dist : with_dists) {
    result->push_back(with_dist.first);
  }
  return OkStatus();
}

template <typename T>
void TreeBruteForceSecondLevelWrapper<T>::CopyToProto(
    SerializedPartitioner* result) const {
  DCHECK(result);
  result->Clear();
  base_->CopyToProto(result);
  SerializedKMeansTreePartitioner* next_lower_kmeans_tree_partitioner =
      result->mutable_kmeans()->mutable_next_bottom_up_level();
  while (next_lower_kmeans_tree_partitioner->has_next_bottom_up_level()) {
    next_lower_kmeans_tree_partitioner =
        next_lower_kmeans_tree_partitioner->mutable_next_bottom_up_level();
  }
  SerializedPartitioner tmp;
  top_level_->query_tokenizer()->CopyToProto(&tmp);
  *next_lower_kmeans_tree_partitioner = std::move(*tmp.mutable_kmeans());
  auto& root = *next_lower_kmeans_tree_partitioner->mutable_kmeans_tree()
                    ->mutable_root();
  for (auto [token, dp_idxs] : Enumerate(top_level_->datapoints_by_token())) {
    auto& node = *root.mutable_children(token);
    node.mutable_indices()->Assign(dp_idxs.begin(), dp_idxs.end());
  }
}

template <typename T>
StatusOrPtr<TreeXHybridSMMD<float>>
TreeBruteForceSecondLevelWrapper<T>::CreateTopLevel(
    const KMeansTreeLikePartitioner<T>& base,
    const BottomUpTopLevelPartitioner& config,
    std::optional<SerializedKMeansTreePartitioner> serialized) {
  if (config.num_centroids_to_search() < 0) {
    return InvalidArgumentError(
        "Must specify a value >=0 for "
        "BottomUpTopLevelPartitioner.num_centroids_to_search.");
  }
  if (config.num_centroids_to_search() > config.num_centroids()) {
    return InvalidArgumentError(
        "BottomUpTopLevelPartitioner.num_centroids_to_search must be <= "
        "BottomUpTopLevelPartitioner.num_centroids.  (Got %d vs %d)",
        config.num_centroids_to_search(), config.num_centroids());
  }
  if (static_cast<int32_t>(base.query_spilling_max_centers()) < 0) {
    return InvalidArgumentError(
        "Must specify a positive value of query_spilling.max_spill_centers "
        "that fits in an int32_t.  (Got %d vs %d)",
        base.query_spilling_max_centers(), base.n_tokens());
  }

  unique_ptr<KMeansTreePartitioner<float>> top_partitioner;
  vector<std::vector<DatapointIndex>> token_to_datapoints;
  if (serialized) {
    top_partitioner = make_unique<KMeansTreePartitioner<float>>(
        make_unique<SquaredL2Distance>(), base.query_tokenization_distance(),
        *serialized);
    token_to_datapoints.resize(top_partitioner->n_tokens());
    auto root = serialized->kmeans_tree().root();
    SCANN_RET_CHECK_EQ(root.children_size(), token_to_datapoints.size());
    for (auto& child : root.children()) {
      token_to_datapoints[child.leaf_id()].assign(child.indices().begin(),
                                                  child.indices().end());
    }
  } else {
    top_partitioner = make_unique<KMeansTreePartitioner<float>>(
        make_unique<SquaredL2Distance>(), base.query_tokenization_distance());
    KMeansTreeTrainingOptions opts;
    opts.center_initialization_type = GmmUtils::Options::RANDOM_INITIALIZATION;
    SCANN_RETURN_IF_ERROR(top_partitioner->CreatePartitioning(
        base.LeafCenters(), SquaredL2Distance(), config.num_centroids(),
        &opts));
    top_partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);
    if (config.soar().enabled()) {
      top_partitioner->set_orthogonality_amplification_lambda(
          config.soar().lambda());
    }
    DCHECK(top_partitioner != nullptr);
    const float avq_eta = config.avq();
    token_to_datapoints =
        top_partitioner
            ->TokenizeDatabase(
                base.LeafCenters(), nullptr,
                {.avq_after_primary = !std::isnan(avq_eta), .avq_eta = avq_eta})
            .value();
  }
  top_partitioner->set_query_spilling_type(
      QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS);
  top_partitioner->set_query_spilling_max_centers(
      config.num_centroids_to_search());
  SCANN_ASSIGN_OR_RETURN(
      unique_ptr<TreeXHybridSMMD<float>> result,
      CreateTopLevelSearcher(base, config, std::move(token_to_datapoints)));

  unique_ptr<KMeansTreeLikePartitioner<float>> maybe_recursed_partitioner;
  if constexpr (std::is_same_v<T, float>) {
    if (config.next_higher_level().enabled()) {
      auto next_level = make_unique<TreeBruteForceSecondLevelWrapper<T>>(
          std::move(top_partitioner));
      SCANN_RETURN_IF_ERROR(
          next_level->CreatePartitioning(config.next_higher_level()));
      maybe_recursed_partitioner = std::move(next_level);
    }
  }
  if (!maybe_recursed_partitioner) {
    SCANN_RET_CHECK(top_partitioner);
    SCANN_RET_CHECK(!config.next_higher_level().enabled())
        << "N-level recursion is not supported for non-float types.";
    maybe_recursed_partitioner = std::move(top_partitioner);
  }
  maybe_recursed_partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  result->set_query_tokenizer(std::move(maybe_recursed_partitioner));
  return result;
}

SCANN_INSTANTIATE_TYPED_CLASS(, TreeBruteForceSecondLevelWrapper);

}  // namespace research_scann
