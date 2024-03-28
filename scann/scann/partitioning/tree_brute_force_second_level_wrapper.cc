// Copyright 2024 The Google Research Authors.
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

#include "scann/utils/datapoint_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

template <typename T>
TreeBruteForceSecondLevelWrapper<T>::TreeBruteForceSecondLevelWrapper(
    unique_ptr<KMeansTreeLikePartitioner<T>> base,
    DatapointIndex top_level_centroids,
    DatapointIndex top_level_centroids_to_search, float avq_eta,
    float orthogonality_amplification_lambda,
    float spilling_overretrieve_factor)
    : base_(std::move(base)),
      top_level_(CreateTopLevel(*base_, top_level_centroids,
                                top_level_centroids_to_search, avq_eta,
                                orthogonality_amplification_lambda,
                                spilling_overretrieve_factor)
                     .value()) {}

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
    params.set_pre_reordering_epsilon(numeric_limits<float>::infinity());
    if constexpr (std::is_same_v<T, float>) {
      return top_level_->FindNeighbors(dptr, params, result);
    } else {
      Datapoint<float> float_dptr;
      return top_level_->FindNeighbors(MaybeConvertDatapoint(dptr, &float_dptr),
                                       params, result);
    }
  }
}

template <typename T>
Status
TreeBruteForceSecondLevelWrapper<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<std::vector<pair<DatapointIndex, float>>> results) const {
  if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    return base_->TokensForDatapointWithSpillingBatched(
        queries, max_centers_override, results);
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
    params[i].set_pre_reordering_epsilon(numeric_limits<float>::infinity());
  }
  if constexpr (std::is_same_v<T, float>) {
    return top_level_->FindNeighborsBatchedNoSortNoExactReorder(queries, params,
                                                                results);
  } else {
    DenseDataset<float> float_queries;
    down_cast<const DenseDataset<T>*>(&queries)->ConvertType(&float_queries);
    return top_level_->FindNeighborsBatchedNoSortNoExactReorder(
        float_queries, params, results);
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
StatusOrPtr<TreeXHybridSMMD<float>>
TreeBruteForceSecondLevelWrapper<T>::CreateTopLevel(
    const KMeansTreeLikePartitioner<T>& base,
    DatapointIndex top_level_centroids,
    DatapointIndex top_level_centroids_to_search, float avq_eta,
    float orthogonality_amplification_lambda,
    float spilling_overretrieve_factor) {
  SCANN_RET_CHECK_LT(top_level_centroids_to_search, top_level_centroids);
  auto top_partitioner = make_unique<KMeansTreePartitioner<float>>(
      base.query_tokenization_distance(), make_unique<SquaredL2Distance>());
  KMeansTreeTrainingOptions opts;
  opts.center_initialization_type = GmmUtils::Options::RANDOM_INITIALIZATION;
  SCANN_RETURN_IF_ERROR(top_partitioner->CreatePartitioning(
      base.LeafCenters(), SquaredL2Distance(), top_level_centroids, &opts));
  top_partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);
  top_partitioner->set_query_spilling_type(
      QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS);
  top_partitioner->set_query_spilling_max_centers(
      top_level_centroids_to_search);
  top_partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);
  top_partitioner->set_orthogonality_amplification_lambda(
      orthogonality_amplification_lambda);
  DCHECK(top_partitioner != nullptr);
  top_partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);
  auto token_to_datapoints =
      top_partitioner
          ->TokenizeDatabase(
              base.LeafCenters(), nullptr,
              {.avq_after_primary = !std::isnan(avq_eta), .avq_eta = avq_eta})
          .value();
  auto result = make_unique<TreeXHybridSMMD<float>>(
      MakeDummyShared(&base.LeafCenters()), nullptr, top_level_centroids,
      numeric_limits<float>::infinity());
  SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
      std::move(token_to_datapoints),
      [&](shared_ptr<TypedDataset<float>> dataset_partition,
          shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
          int32_t) -> StatusOr<unique_ptr<SingleMachineSearcherBase<float>>> {
        SCANN_RET_CHECK(!hashed_dataset_partition);
        return make_unique<BruteForceSearcher<float>>(
            base.query_tokenization_distance(), dataset_partition,
            base.query_spilling_max_centers(),
            numeric_limits<float>::infinity());
      }));
  top_partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  result->set_query_tokenizer(std::move(top_partitioner));
  result->set_spilling_overretrieve_factor(spilling_overretrieve_factor);
  return result;
}

SCANN_INSTANTIATE_TYPED_CLASS(, TreeBruteForceSecondLevelWrapper);

}  // namespace research_scann
