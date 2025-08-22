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



#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/tree_x_hybrid/internal/batching.h"
#include "scann/tree_x_hybrid/internal/utils.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/hash_leaf_helpers.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
TreeXHybridSMMD<T>::TreeXHybridSMMD(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase<T>(dataset, hashed_dataset,
                                   default_pre_reordering_num_neighbors,
                                   default_pre_reordering_epsilon) {}

template <typename T>
TreeXHybridSMMD<T>::TreeXHybridSMMD(
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase<T>(default_pre_reordering_num_neighbors,
                                   default_pre_reordering_epsilon),
      is_streaming_input_data_(true) {}

template <typename T>
DatapointIndex TreeXHybridSMMD<T>::optimal_batch_size() const {
  auto kmeans_partitioner =
      dynamic_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
  if (!kmeans_partitioner) return 1;
  return (kmeans_partitioner->SupportsLowLevelQueryBatching()) ? 256 : 1;
}

namespace {

template <typename T>
TypedDataset<T>* PartitionDataset(const TypedDataset<T>& original,
                                  ConstSpan<DatapointIndex> subset) {
  TypedDataset<T>* result =
      (original.IsSparse())
          ? absl::implicit_cast<TypedDataset<T>*>(new SparseDataset<T>)
          : absl::implicit_cast<TypedDataset<T>*>(new DenseDataset<T>);
  result->set_packing_strategy(original.packing_strategy());
  result->set_dimensionality(original.dimensionality());
  result->Reserve(subset.size());
  for (const DatapointIndex i : subset) {
    result->AppendOrDie(original[i], "");
  }

  result->set_normalization_tag(original.normalization());
  return result;
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::BuildLeafSearchers(
    const Partitioner<T>& database_tokenizer,
    std::function<StatusOrSearcher(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder,
    shared_ptr<ThreadPool> thread_pool) {
  if (!leaf_searchers_.empty()) {
    return FailedPreconditionError(
        "BuildLeafSearchers must not be called more than once per instance.");
  }
  if (this->dataset() == nullptr) {
    return FailedPreconditionError(
        "this->dataset() must be non-null if calling the overload of "
        "TreeXHybridSMMD::BuildLeafSearchers where datapoints_by_token is "
        "computed on-the-fly.");
  }

  VLOG(1) << "Tokenizing database...";
  const absl::Time tokenization_start = absl::Now();
  SCANN_ASSIGN_OR_RETURN(
      auto datapoints_by_token,
      database_tokenizer.TokenizeDatabase(*this->dataset(), thread_pool.get()));
  VLOG(1) << "Done tokenizing database in " << absl::Now() - tokenization_start
          << ".";
  return BuildLeafSearchers(std::move(datapoints_by_token),
                            leaf_searcher_builder);
}

template <typename T>
void TreeXHybridSMMD<T>::set_leaf_searcher_optional_parameter_creator(
    shared_ptr<const LeafSearcherOptionalParameterCreator<T>> x) {
  leaf_searcher_optional_parameter_creator_ = std::move(x);
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildLeafSearchers(
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    std::function<StatusOrSearcher(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder) {
  leaf_searcher_builder_ = leaf_searcher_builder;
  for (std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
    if (!dp_list.empty()) {
      num_datapoints_ =
          std::max(num_datapoints_, *absl::c_max_element(dp_list) + 1);
    }
  }

  SCANN_ASSIGN_OR_RETURN(const DatapointIndex dataset_size,
                         this->DatasetSize());
  SCANN_ASSIGN_OR_RETURN(
      datapoints_by_token_disjoint_,
      ValidateDatapointsByToken(datapoints_by_token, dataset_size));

  const TypedDataset<T>* dataset = this->dataset();
  const DenseDataset<uint8_t>* hashed_dataset = this->hashed_dataset();
  const DatapointIndex n_tokens = datapoints_by_token.size();
  leaf_searchers_.resize(n_tokens);
  for (int32_t token = 0; token < n_tokens; ++token) {
    const absl::Time token_start = absl::Now();
    if (!hashed_dataset) {
      shared_ptr<TypedDataset<T>> dataset_partition(
          PartitionDataset(*dataset, datapoints_by_token[token]));
      SCANN_ASSIGN_OR_RETURN(
          unique_ptr<SingleMachineSearcherBase<T>> leaf_searcher,
          leaf_searcher_builder(dataset_partition, nullptr, token));

      if (!leaf_searcher->needs_dataset()) {
        leaf_searcher->ReleaseDatasetAndDocids();
      }

      leaf_searchers_[token] = std::move(leaf_searcher);
    } else {
      shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition(
          down_cast<DenseDataset<uint8_t>*>(
              PartitionDataset(*hashed_dataset, datapoints_by_token[token])));
      SCANN_ASSIGN_OR_RETURN(
          unique_ptr<SingleMachineSearcherBase<T>> leaf_searcher,
          leaf_searcher_builder(nullptr, hashed_dataset_partition, token));
      if (!leaf_searcher->needs_hashed_dataset()) {
        leaf_searcher->ReleaseHashedDataset();
      }
      leaf_searchers_[token] = std::move(leaf_searcher);
    }

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
  }

  datapoints_by_token_ = std::move(datapoints_by_token);
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                              this->crowding_dimension_names());
  }
  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildBFloat16BruteForceLeafSearchers(
    const DenseDataset<int16_t>& bfloat16_database,
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    std::function<unique_ptr<SingleMachineSearcherBase<float>>(
        shared_ptr<const DenseDataset<int16_t>> dataset_partition,
        int32_t token)>
        leaf_searcher_builder) {
  if constexpr (!std::is_same_v<T, float>) {
    return UnimplementedError("Tree-brute force only works with float.");
  } else {
    for (std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
      if (!dp_list.empty()) {
        num_datapoints_ =
            std::max(num_datapoints_, *absl::c_max_element(dp_list) + 1);
      }
    }

    const DatapointIndex dataset_size = bfloat16_database.size();
    SCANN_ASSIGN_OR_RETURN(
        datapoints_by_token_disjoint_,
        ValidateDatapointsByToken(datapoints_by_token, dataset_size));

    const DatapointIndex n_tokens = datapoints_by_token.size();
    leaf_searchers_.resize(n_tokens);
    for (int32_t token = 0; token < n_tokens; ++token) {
      const absl::Time token_start = absl::Now();
      auto partitioned =
          PartitionDataset(bfloat16_database, datapoints_by_token[token]);
      auto dense = dynamic_cast<const DenseDataset<int16_t>*>(partitioned);
      SCANN_RET_CHECK(dense != nullptr);
      shared_ptr<const DenseDataset<int16_t>> dataset_partition(dense);
      unique_ptr<SingleMachineSearcherBase<float>> leaf_searcher =
          leaf_searcher_builder(dataset_partition, token);

      leaf_searchers_[token] = std::move(leaf_searcher);
      VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
              << " (size = " << datapoints_by_token[token].size() << " DPs) in "
              << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
    }

    datapoints_by_token_ = std::move(datapoints_by_token);
    if (this->crowding_enabled()) {
      return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                                this->crowding_dimension_names());
    }
    return OkStatus();
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::AddLeafSearcher() {
  SCANN_RET_CHECK(!leaf_searchers_.empty())
      << "At least one leaf searcher must exist in the current tree searcher.";
  auto hashed_database = make_shared<DenseDataset<uint8_t>>();
  auto database = make_shared<DenseDataset<T>>();
  if (leaf_searchers_[0]->dataset())
    database->set_dimensionality(
        leaf_searchers_[0]->dataset()->dimensionality());
  else
    database = nullptr;
  if (leaf_searchers_[0]->hashed_dataset())
    hashed_database->set_dimensionality(
        leaf_searchers_[0]->hashed_dataset()->dimensionality());
  else
    hashed_database = nullptr;
  unique_ptr<SingleMachineSearcherBase<T>> leaf_searcher;
  if (leaf_searcher_builder_) {
    SCANN_ASSIGN_OR_RETURN(
        leaf_searcher, leaf_searcher_builder_(database, hashed_database, -1));
  } else if (sq_leaf_searcher_builder_) {
    DenseDataset<int8_t> sq_database;
    vector<float> squared_l2_norms;
    SCANN_ASSIGN_OR_RETURN(
        leaf_searcher,
        sq_leaf_searcher_builder_(std::move(sq_database), squared_l2_norms));
  } else {
    return InvalidArgumentError(
        "Either leaf_searcher_builder_ or sq_leaf_searcher_builder_ must be "
        "set for AddLeafSearcher.");
  }
  if (!leaf_searcher->needs_dataset()) leaf_searcher->ReleaseDatasetAndDocids();
  if (!leaf_searcher->needs_hashed_dataset())
    leaf_searcher->ReleaseHashedDataset();
  leaf_searchers_.push_back(std::move(leaf_searcher));
  datapoints_by_token_.push_back(std::vector<DatapointIndex>());

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildStreamingAsymmetricHashingLeafSearchers(
    size_t n_tokens, ConstSpan<int32_t> query_tokens,
    const internal::TrainedAsymmetricHashingResults<T>& training_results,
    bool streaming_result,
    std::function<StatusOrSearcher(
        int32_t token,
        const internal::TrainedAsymmetricHashingResults<T>& training_results)>
        leaf_searcher_builder) {
  leaf_searchers_.resize(n_tokens);
  for (int32_t token : query_tokens) {
    const absl::Time token_start_time = absl::Now();
    SCANN_RET_CHECK_LT(token, n_tokens);
    SCANN_ASSIGN_OR_RETURN(leaf_searchers_[token],
                           leaf_searcher_builder(token, training_results));

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " in " << absl::ToDoubleSeconds(absl::Now() - token_start_time)
            << " sec.";
  }
  is_streaming_result_ = streaming_result;

  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                              this->crowding_dimension_names());
  }

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildPretrainedScalarQuantizationLeafSearchers(
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    vector<DenseDataset<int8_t>> partitioned_datasets,
    vector<vector<float>> partitioned_squared_l2_norms,
    std::function<
        StatusOrSearcher(DenseDataset<int8_t> scalar_quantized_partition,
                         vector<float> squared_l2_norms)>
        leaf_searcher_builder) {
  sq_leaf_searcher_builder_ = leaf_searcher_builder;
  for (auto& dp_list : datapoints_by_token) {
    if (!dp_list.empty()) {
      num_datapoints_ =
          std::max(num_datapoints_, *absl::c_max_element(dp_list) + 1);
    }
  }
  SCANN_ASSIGN_OR_RETURN(
      datapoints_by_token_disjoint_,
      ValidateDatapointsByToken(datapoints_by_token, num_datapoints_));

  const auto n_tokens = datapoints_by_token.size();
  leaf_searchers_.resize(n_tokens);
  for (int32_t token = 0; token < n_tokens; ++token) {
    const auto token_start = absl::Now();

    vector<float> squared_l2_norms;
    if (!partitioned_squared_l2_norms.empty())
      squared_l2_norms = std::move(partitioned_squared_l2_norms[token]);
    SCANN_ASSIGN_OR_RETURN(
        auto leaf_searcher,
        leaf_searcher_builder(std::move(partitioned_datasets[token]),
                              std::move(squared_l2_norms)));
    leaf_searchers_[token] = std::move(leaf_searcher);

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
  }

  datapoints_by_token_ = std::move(datapoints_by_token);
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                              this->crowding_dimension_names());
  }

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildStreamingScalarQuantizationLeafSearchers(
    size_t n_tokens, absl::Span<const int32_t> query_tokens,
    std::shared_ptr<const DistanceMeasure> distance,
    ConstSpan<float> inverse_multiplier_by_dimension, bool streaming_result,
    std::function<StatusOrSearcher(
        int token, std::shared_ptr<const DistanceMeasure> distance,
        ConstSpan<float> inverse_multiplier_by_dimension)>
        leaf_searcher_builder) {
  leaf_searchers_.resize(n_tokens);
  for (int32_t i = 0; i < query_tokens.size(); ++i) {
    const auto token_start = absl::Now();

    int32_t token = query_tokens[i];
    SCANN_RET_CHECK_LT(token, n_tokens);
    SCANN_ASSIGN_OR_RETURN(
        auto leaf_searcher,
        leaf_searcher_builder(token, distance,
                              inverse_multiplier_by_dimension));
    leaf_searchers_[token] = std::move(leaf_searcher);

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " in " << absl::ToDoubleSeconds(absl::Now() - token_start)
            << " sec.";
  }
  is_streaming_result_ = streaming_result;

  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                              this->crowding_dimension_names());
  }

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::BuildStreamingLeafSearchers(
    size_t n_tokens, absl::Span<const int32_t> query_tokens,
    std::shared_ptr<const DistanceMeasure> distance, bool streaming_result,
    std::function<StatusOrSearcher(
        int token, std::shared_ptr<const DistanceMeasure> distance)>
        leaf_searcher_builder) {
  leaf_searchers_.resize(n_tokens);
  for (int32_t i = 0; i < query_tokens.size(); ++i) {
    const auto token_start = absl::Now();

    int32_t token = query_tokens[i];
    SCANN_RET_CHECK_LT(token, n_tokens);
    SCANN_ASSIGN_OR_RETURN(auto leaf_searcher,
                           leaf_searcher_builder(token, distance));
    leaf_searchers_[token] = std::move(leaf_searcher);

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " in " << absl::ToDoubleSeconds(absl::Now() - token_start)
            << " sec.";
  }
  is_streaming_result_ = streaming_result;

  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute(),
                              this->crowding_dimension_names());
  }

  return OkStatus();
}

namespace {

void RemapToGlobalDatapointIndices(
    MutableSpan<pair<DatapointIndex, float>> partition_leaf_result,
    ConstSpan<DatapointIndex> local_to_global_datapoint_indices) {
  for (auto& r : partition_leaf_result) {
    r.first = local_to_global_datapoint_indices[r.first];
  }
}

inline bool PreTokenizationEnabled(
    const shared_ptr<const TreeXOptionalParameters>& params) {
  return params && params->pre_tokenization_enabled();
}

inline bool PreTokenizationEnabled(const SearchParameters& params) {
  return PreTokenizationEnabled(
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>());
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
    ConstSpan<std::string> crowding_dimension_names) {
  if (leaf_searchers_.empty()) return OkStatus();
  if (is_streaming_input_data_) return OkStatus();
  for (size_t token = 0; token < leaf_searchers_.size(); ++token) {
    ConstSpan<DatapointIndex> cur_leaf_datapoints = datapoints_by_token_[token];
    vector<int64_t> leaf_datapoint_index_to_crowding_attribute(
        cur_leaf_datapoints.size());
    for (size_t i = 0; i < cur_leaf_datapoints.size(); ++i) {
      leaf_datapoint_index_to_crowding_attribute[i] =
          datapoint_index_to_crowding_attribute[cur_leaf_datapoints[i]];
    }
    Status status = leaf_searchers_[token]->EnableCrowding(
        std::move(leaf_datapoint_index_to_crowding_attribute));
    if (!status.ok()) {
      for (size_t i = 0; i <= token; ++i) {
        leaf_searchers_[i]->DisableCrowding();
      }
    }
  }
  return OkStatus();
}

template <typename T>
void TreeXHybridSMMD<T>::DisableCrowdingImpl() {
  for (auto& ls : leaf_searchers_) {
    ls->DisableCrowding();
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsImpl(const DatapointPtr<T>& query,
                                             const SearchParameters& params,
                                             NNResultsVector* result) const {
  SCANN_RETURN_IF_ERROR(CheckReadyToQuery(params));
  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> query_tokens_storage;
  ConstSpan<int32_t> query_tokens;
  if (PreTokenizationEnabled(tree_x_params)) {
    query_tokens = tree_x_params->leaf_tokens_to_search();
  } else if (params.unlocked_query_preprocessing_results<CentersToSearch>()) {
    query_tokens =
        params.unlocked_query_preprocessing_results<CentersToSearch>()
            ->centers_to_search();
  } else {
    bool override = false;

    if (tree_x_params) {
      const auto num_partitions_to_search_override =
          tree_x_params->num_partitions_to_search_override();
      if (num_partitions_to_search_override > 0) {
        const auto* kmeans_tokenizer =
            down_cast<const KMeansTreeLikePartitioner<T>*>(
                query_tokenizer_.get());
        if (!kmeans_tokenizer)
          return InvalidArgumentError(
              "num_partitions_to_search_override is > 0, but the tokenizer is "
              "not a KMeansTreeLikePartitioner.");

        vector<pair<DatapointIndex, float>> pairs_storage;
        SCANN_RETURN_IF_ERROR(kmeans_tokenizer->TokensForDatapointWithSpilling(
            query, num_partitions_to_search_override, &pairs_storage));
        query_tokens_storage.resize(pairs_storage.size());
        for (const auto [i, pair] : Enumerate(pairs_storage))
          query_tokens_storage[i] = pair.first;
        override = true;
      }
    }

    if (!override) {
      SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
          query, &query_tokens_storage));
    }

    query_tokens = query_tokens_storage;
  }

  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    return FindNeighborsPreTokenizedImpl(
        query, params, query_tokens,
        TopNeighbors<float>(NumNeighborsWithSpillingMultiplier(
            params.pre_reordering_num_neighbors())),
        result);
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if (is_streaming_result_) {
    return FailedPreconditionError(
        "Find neighbors for a batch of queries is not supported "
        "for this instance of TreeXHybridSMMD with streaming result.");
  }
  if (params.empty()) {
    DCHECK_EQ(queries.size(), 0);
    DCHECK_EQ(results.size(), 0);
    return OkStatus();
  }

  SCANN_RETURN_IF_ERROR(CheckReadyToQuery(params[0]));

  const bool pre_tokenized = PreTokenizationEnabled(params[0]);
  for (const SearchParameters& p : params) {
    if (PreTokenizationEnabled(p) != pre_tokenized) {
      return InvalidArgumentError(
          "For tree-X hybrid FindNeighborsBatched, either all queries in a "
          "given batch must use pre-tokenization or no queries in this batch "
          "must use pre-tokenization.");
    }
  }

  const bool has_unlocked_preprocessing =
      params[0].unlocked_query_preprocessing_results<CentersToSearch>();
  if (!pre_tokenized) {
    for (const SearchParameters& p : params) {
      if (static_cast<bool>(
              p.unlocked_query_preprocessing_results<CentersToSearch>()) !=
          has_unlocked_preprocessing) {
        return InvalidArgumentError(
            "For tree-X hybrid FindNeighborsBatched, either all queries in a "
            "given batch must use unlocked preprocessing or no queries in "
            "this batch must use unlocked preprocessing.");
      }
    }
  }

  vector<vector<int32_t>> query_tokens_storage;
  vector<ConstSpan<int32_t>> query_tokens(queries.size());
  if (pre_tokenized) {
    for (size_t i = 0; i < queries.size(); ++i) {
      auto tree_x_params =
          params[i]
              .searcher_specific_optional_parameters<TreeXOptionalParameters>();
      query_tokens[i] = tree_x_params->leaf_tokens_to_search();
    }
  } else if (has_unlocked_preprocessing) {
    for (size_t i : IndicesOf(queries)) {
      auto unlocked_preprocessing =
          params[i].unlocked_query_preprocessing_results<CentersToSearch>();
      query_tokens[i] = unlocked_preprocessing->centers_to_search();
    }
  } else {
    query_tokens_storage.resize(queries.size());
    vector<int32_t> max_centers_override(queries.size(), 0);
    bool override = false;

    for (int i = 0; i < queries.size(); ++i) {
      auto tree_x_params =
          params[i]
              .searcher_specific_optional_parameters<TreeXOptionalParameters>();
      if (tree_x_params)
        max_centers_override[i] =
            tree_x_params->num_partitions_to_search_override();
      if (max_centers_override[i] > 0) override = true;
    }

    if (override) {
      const auto* kmeans_tokenizer =
          down_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
      if (!kmeans_tokenizer)
        return InvalidArgumentError(
            "num_partitions_to_search_override is > 0, but the tokenizer is "
            "not a KMeansTreePartitioner.");

      SCANN_RETURN_IF_ERROR(
          kmeans_tokenizer->TokensForDatapointWithSpillingBatchedAndOverride(
              queries, max_centers_override,
              MakeMutableSpan(query_tokens_storage)))
          << typeid(*kmeans_tokenizer).name();
    } else {
      SCANN_RETURN_IF_ERROR(
          query_tokenizer_->TokensForDatapointWithSpillingBatched(
              queries, MakeMutableSpan(query_tokens_storage)))
          << typeid(*query_tokenizer_).name();
    }

    for (size_t i = 0; i < queries.size(); ++i) {
      query_tokens[i] = query_tokens_storage[i];
    }
  }

  if (!tree_x_internal::SupportsLowLevelBatching(queries, params) ||
      tree_x_internal::RecursiveSize(query_tokens) < leaf_searchers_.size()) {
    return FindNeighborsPreTokenizedBatchedGenericImpl(queries, params,
                                                       query_tokens, results);
  } else {
    return FindNeighborsPreTokenizedBatchedOptimizedImpl(queries, params,
                                                         query_tokens, results);
  }
}

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedBatchedGenericImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    ConstSpan<ConstSpan<int32_t>> query_tokens,
    MutableSpan<NNResultsVector> results) const {
  DCHECK_EQ(queries.size(), params.size());
  DCHECK_EQ(queries.size(), query_tokens.size());
  DCHECK_EQ(queries.size(), results.size());
  for (DatapointIndex i : IndicesOf(queries)) {
    if (params[i].pre_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      SCANN_RETURN_IF_ERROR(FindNeighborsPreTokenizedImpl(
          queries[i], params[i], query_tokens[i],
          TopNeighbors<float>(NumNeighborsWithSpillingMultiplier(
              params[i].pre_reordering_num_neighbors())),
          &results[i]));
    }
  }
  return OkStatus();
}

namespace {

vector<std::vector<DatapointIndex>> InvertQueryTokens(
    ConstSpan<ConstSpan<int32_t>> query_tokens, size_t num_tokens) {
  vector<std::vector<DatapointIndex>> result(num_tokens);
  for (DatapointIndex query_index : IndicesOf(query_tokens)) {
    ConstSpan<int32_t> cur_query_tokens = query_tokens[query_index];
    for (int32_t qt : cur_query_tokens) {
      result[qt].push_back(query_index);
    }
  }
  return result;
}

size_t MaxQueriesPerPartition(
    ConstSpan<std::vector<DatapointIndex>> queries_by_partition) {
  size_t result = 0;
  for (auto& vec : queries_by_partition) {
    result = std::max(result, vec.size());
  }
  return result;
}

}  // namespace

template <typename T>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedBatchedOptimizedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    ConstSpan<ConstSpan<int32_t>> query_tokens,
    MutableSpan<NNResultsVector> results) const {
  DCHECK(queries.IsDense());

  vector<std::vector<DatapointIndex>> queries_by_partition =
      InvertQueryTokens(query_tokens, leaf_searchers_.size());
  const size_t max_queries_per_partition =
      MaxQueriesPerPartition(queries_by_partition);
  vector<T> backing_storage;
  backing_storage.reserve(queries.dimensionality() * max_queries_per_partition);

  auto make_leaf_query_dataset =
      [&backing_storage, &queries](ConstSpan<DatapointIndex> query_idxs) {
        backing_storage.resize(0);
        for (DatapointIndex qi : query_idxs) {
          ConstSpan<T> values = queries[qi].values_span();
          backing_storage.insert(backing_storage.end(), values.begin(),
                                 values.end());
        }
        return DenseDataset<T>(std::move(backing_storage), query_idxs.size());
      };

  vector<FastTopNeighbors<float>> top_ns;
  top_ns.reserve(params.size());
  vector<shared_ptr<const SearcherSpecificOptionalParameters>>
      leaf_optional_params(queries.size());

  bool leaf_optional_params_all_null = true;
  for (const auto& [query_idx, p] : Enumerate(params)) {
    top_ns.emplace_back(
        NumNeighborsWithSpillingMultiplier(p.pre_reordering_num_neighbors()),
        p.pre_reordering_epsilon());
    SCANN_ASSIGN_OR_RETURN(
        leaf_optional_params[query_idx],
        CreateLeafOptionalParameters(queries[query_idx], params[query_idx]));
    leaf_optional_params_all_null &= leaf_optional_params[query_idx] == nullptr;
  }

  vector<FastTopNeighbors<float>*> leaf_topns;
  leaf_topns.reserve(max_queries_per_partition);
  for (auto [leaf_idx, query_idxs] : Enumerate(queries_by_partition)) {
    if (query_idxs.empty()) continue;
    DenseDataset<T> leaf_dataset = make_leaf_query_dataset(query_idxs);
    vector<SearchParameters> leaf_params;
    if (!leaf_optional_params_all_null) {
      leaf_params = tree_x_internal::CreateParamsSubsetForLeaf<DatapointIndex>(
          top_ns, leaf_optional_params, query_idxs);
    }
    leaf_topns.resize(leaf_dataset.size());
    for (auto [i, query_idx] : Enumerate(query_idxs)) {
      leaf_topns[i] = &top_ns[query_idx];
    }
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[leaf_idx]->FindNeighborsBatchedNoSortNoExactReorder(
            leaf_dataset, leaf_params, MakeMutableSpan(leaf_topns),
            is_streaming_input_data_ ? ConstSpan<DatapointIndex>()
                                     : datapoints_by_token_[leaf_idx]));
    backing_storage = leaf_dataset.ClearRecyclingDataVector();
  }

  for (size_t query_index : IndicesOf(top_ns)) {
    top_ns[query_index].FinishUnsorted(&results[query_index]);
    if (!datapoints_by_token_disjoint_) {
      DeduplicateDatabaseSpilledResults(
          &results[query_index],
          params[query_index].pre_reordering_num_neighbors());
    }
  }
  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::CheckReadyToQuery(
    const SearchParameters& params) const {
  if (leaf_searchers_.empty()) {
    return FailedPreconditionError("BuildLeafSearchers not called yet.");
  }

  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> query_tokens_storage;
  if (!PreTokenizationEnabled(tree_x_params) && query_tokenizer_ == nullptr) {
    return FailedPreconditionError(
        "Query tokenizer not set and pre-tokenization "
        "not enabled.");
  }

  return OkStatus();
}

template <typename T>
Status TreeXHybridSMMD<T>::ValidateTokenList(ConstSpan<int32_t> token_list,
                                             bool check_oob) const {
  absl::flat_hash_set<int32_t> duplicate_checker;

  for (int32_t token : token_list) {
    if (!duplicate_checker.insert(token).second) {
      return InvalidArgumentError(
          absl::StrCat("Duplicate token:  ", token, "."));
    }

    if (token < 0) {
      return InvalidArgumentError(StrCat(
          "Tree-X hybrid tokens may not be negative.  (Got: ", token, ")."));
    }

    if (check_oob) {
      if (!is_streaming_input_data_) {
        if (token >= datapoints_by_token_.size()) {
          return InvalidArgumentError(
              absl::StrCat("Token out of bounds (", token, " vs. ",
                           datapoints_by_token_.size()));
        }
      }
      if (token >= leaf_searchers_.size()) {
        return InvalidArgumentError(
            "Query token out of range of database tokens (got %d, "
            "expected in the range [0, %d).",
            static_cast<int>(token), static_cast<int>(leaf_searchers_.size()));
      }
    }
  }

  return OkStatus();
}

template <typename T>
StatusOr<shared_ptr<const SearcherSpecificOptionalParameters>>
TreeXHybridSMMD<T>::CreateLeafOptionalParameters(
    const DatapointPtr<T>& query,
    const SearchParameters& top_level_params) const {
  auto tree_x_params =
      top_level_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  const bool have_external_leaf_optional_params =
      tree_x_params && tree_x_params->all_leaf_optional_params();
  if (leaf_searcher_optional_parameter_creator_ &&
      have_external_leaf_optional_params) {
    return InvalidArgumentError(
        "Conflicting leaf searcher optional parameters.  Cannot have both "
        "external parameters from TreeXOptionalParameters and a "
        "LeafSearcherOptionalParameterCreator.");
  } else if (leaf_searcher_optional_parameter_creator_) {
    return leaf_searcher_optional_parameter_creator_
        ->CreateLeafSearcherOptionalParameters(query);
  } else if (have_external_leaf_optional_params) {
    return tree_x_params->all_leaf_optional_params();
  } else {
    return shared_ptr<const SearcherSpecificOptionalParameters>(nullptr);
  }
}

template <typename T>
template <typename TopN>
Status TreeXHybridSMMD<T>::FindNeighborsPreTokenizedImpl(
    const DatapointPtr<T>& query, const SearchParameters& params,
    ConstSpan<int32_t> query_tokens, TopN top_n,
    NNResultsVector* result) const {
  DCHECK(result);
  DCHECK(top_n.empty());

  if (query_tokens.empty()) {
    result->clear();
    return OkStatus();
  }

  if (params.restrict_whitelist() && !PreTokenizationEnabled(params) &&
      this->reordering_enabled() &&
      std::isinf(params.pre_reordering_epsilon()) &&
      params.restrict_whitelist()->NumPointsAllowlisted() <=
          params.pre_reordering_num_neighbors()) {
    auto it = params.restrict_whitelist()->AllowlistedPointIterator();
    result->clear();
    for (; !it.Done(); it.Next()) {
      result->emplace_back(it.value(), params.pre_reordering_epsilon());
    }
    return OkStatus();
  }

  SCANN_RETURN_IF_ERROR(
      ValidateTokenList(query_tokens, query_tokenizer_ != nullptr));

  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  if (PreTokenizationEnabled(tree_x_params)) {
    DCHECK_EQ(tree_x_params->leaf_tokens_to_search().size(),
              query_tokens.size());
    DCHECK(std::equal(query_tokens.begin(), query_tokens.end(),
                      tree_x_params->leaf_tokens_to_search().begin()));
  }

  shared_ptr<const SearcherSpecificOptionalParameters> leaf_optional_params;
  SCANN_ASSIGN_OR_RETURN(leaf_optional_params,
                         CreateLeafOptionalParameters(query, params));
  auto create_leaf_params = [&]() -> SearchParameters {
    SearchParameters leaf_params;
    leaf_params.set_pre_reordering_num_neighbors(
        params.pre_reordering_num_neighbors());
    leaf_params.set_pre_reordering_epsilon(params.pre_reordering_epsilon());
    leaf_params.set_per_crowding_attribute_pre_reordering_num_neighbors(
        params.per_crowding_attribute_pre_reordering_num_neighbors());
    leaf_params.set_searcher_specific_optional_parameters(leaf_optional_params);
    return leaf_params;
  };

  if (query_tokens.size() == 1) {
    SearchParameters leaf_params = create_leaf_params();
    leaf_params.set_pre_reordering_num_neighbors(
        params.pre_reordering_num_neighbors());
    const auto token = query_tokens.front();

    if (!is_streaming_input_data_) {
      if (token >= datapoints_by_token_.size()) {
        LOG_FIRST_N(INFO, 10)
            << "With an empty partitioner, ignored a token of " << token
            << " that is >=" << datapoints_by_token_.size();
        return OkStatus();
      }
    }
    Status status = leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
        query, leaf_params, result);
    if (!status.ok()) return status;

    if (!is_streaming_input_data_) {
      RemapToGlobalDatapointIndices(MakeMutableSpan(*result),
                                    datapoints_by_token_[token]);
    }
    return status;
  }

  std::shared_ptr<SearchParameters> shared_leaf_param =
      std::make_shared<SearchParameters>(create_leaf_params());
  size_t num_neighbors_with_spilling_multiplier =
      NumNeighborsWithSpillingMultiplier(params.pre_reordering_num_neighbors());

  float top_n_forwarded_epsilon = params.pre_reordering_epsilon();

  absl::Mutex top_n_mutex;

  bool require_top_n_locking = false;

  auto pf_task = [&](size_t i) -> Status {
    SearchParameters* leaf_params = shared_leaf_param.get();
    shared_ptr<SearchParameters> local_leaf_params;
    if (require_top_n_locking) {
      local_leaf_params =
          std::make_shared<SearchParameters>(create_leaf_params());
      leaf_params = local_leaf_params.get();
    }
    leaf_params->set_pre_reordering_num_neighbors(
        num_neighbors_with_spilling_multiplier);

    const int32_t token = query_tokens[i];

    if (!is_streaming_input_data_) {
      if (token >= datapoints_by_token_.size()) {
        LOG_FIRST_N(INFO, 10)
            << "With an empty partitioner, ignored a token of " << token
            << " that is >=" << datapoints_by_token_.size();
        return OkStatus();
      }

      {
        std::unique_ptr<absl::MutexLock> lock;
        if (require_top_n_locking) {
          lock = std::make_unique<absl::MutexLock>(&top_n_mutex);
        }
        leaf_params->set_pre_reordering_epsilon(top_n_forwarded_epsilon);
      }
    }
    NNResultsVector leaf_results;
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
            query, *leaf_params, &leaf_results));

    if (!is_streaming_input_data_) {
      RemapToGlobalDatapointIndices(MakeMutableSpan(leaf_results),
                                    datapoints_by_token_[token]);
    }
    if (!is_streaming_result_) {
      std::unique_ptr<absl::MutexLock> lock;
      if (require_top_n_locking) {
        lock = std::make_unique<absl::MutexLock>(&top_n_mutex);
      }
      for (const auto& result : leaf_results) {
        top_n.push(result);
      }
      if (top_n.full()) {
        top_n_forwarded_epsilon = top_n.approx_bottom().second;
      }
    }
    return OkStatus();
  };

  absl::Status pf_status = OkStatus();
  pf_status = ParallelForWithStatus(Seq(query_tokens.size()), nullptr,
                                    std::move(pf_task));
  SCANN_RETURN_IF_ERROR(pf_status);

  if (!is_streaming_result_) {
    *result = top_n.TakeUnsorted();
    if (!datapoints_by_token_disjoint_) {
      DeduplicateDatabaseSpilledResults(result,
                                        params.pre_reordering_num_neighbors());
    }
  }

  return OkStatus();
}

template <typename T>
StatusOr<typename SingleMachineSearcherBase<T>::Mutator*>
TreeXHybridSMMD<T>::GetMutator() const {
  if (!mutator_) {
    SCANN_RET_CHECK(!this->hashed_dataset())
        << "Must release hashed dataset before calling "
           "TreeXHybridSMMD::GetMutator since the hashed dataset is not used "
           "once the tree-X hybrid is built and can't be easily updated.";
    auto mutable_this = const_cast<TreeXHybridSMMD<T>*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_, TreeXHybridMutator<TreeXHybridSMMD<T>>::Create(mutable_this));
  }
  return static_cast<typename SingleMachineSearcherBase<T>::Mutator*>(
      mutator_.get());
}

template <typename T>
StatusOr<typename TreeXHybridSMMD<T>::MutationArtifacts>
TreeXHybridSMMD<T>::TokenizeAndMaybeResidualize(const DatapointPtr<T>& dptr) {
  DCHECK(database_tokenizer_);
  MutationArtifacts result;
  SCANN_RETURN_IF_ERROR(database_tokenizer_->TokensForDatapointWithSpilling(
      dptr, &result.tokens));
  return result;
}

template <typename T>
StatusOr<vector<typename TreeXHybridSMMD<T>::MutationArtifacts>>
TreeXHybridSMMD<T>::TokenizeAndMaybeResidualize(const TypedDataset<T>& dps) {
  vector<vector<int32_t>> token_storage(dps.size());
  SCANN_RETURN_IF_ERROR(
      database_tokenizer_->TokensForDatapointWithSpillingBatched(
          dps, MakeMutableSpan(token_storage)));
  vector<MutationArtifacts> result(dps.size());
  for (size_t dp_idx : IndicesOf(dps)) {
    result[dp_idx].tokens = std::move(token_storage[dp_idx]);
  }
  return result;
}

template <typename T>
StatusOr<SingleMachineFactoryOptions>
TreeXHybridSMMD<T>::ExtractSingleMachineFactoryOptions() {
  SCANN_ASSIGN_OR_RETURN(const int dataset_size,
                         UntypedSingleMachineSearcherBase::DatasetSize());
  auto int8_query_processor = std::dynamic_pointer_cast<
      const TreeScalarQuantizationPreprocessedQueryCreator>(
      leaf_searcher_optional_parameter_creator_);
  ConstSpan<float> int8_multipliers;
  if (int8_query_processor)
    int8_multipliers = int8_query_processor->inverse_multipliers();

  SCANN_ASSIGN_OR_RETURN(
      SingleMachineFactoryOptions leaf_opts,
      MergeAHLeafOptions(leaf_searchers_, datapoints_by_token_, dataset_size,
                         1));

  SCANN_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<T>::ExtractSingleMachineFactoryOptions());
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          datapoints_by_token_);
  opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
  query_tokenizer_->CopyToProto(opts.serialized_partitioner.get());

  if (leaf_opts.ah_codebook != nullptr) {
    opts.ah_codebook = leaf_opts.ah_codebook;
    opts.hashed_dataset = leaf_opts.hashed_dataset;
  }
  if (leaf_opts.pre_quantized_fixed_point && !int8_multipliers.empty()) {
    opts.pre_quantized_fixed_point = make_shared<PreQuantizedFixedPoint>();
    opts.pre_quantized_fixed_point = leaf_opts.pre_quantized_fixed_point;
    opts.pre_quantized_fixed_point->multiplier_by_dimension =
        make_shared<vector<float>>(int8_multipliers.begin(),
                                   int8_multipliers.end());

    for (float& mult : *opts.pre_quantized_fixed_point->multiplier_by_dimension)
      mult = 1 / mult;
  }
  return opts;
}

template <typename T>
StatusOr<shared_ptr<const DenseDataset<float>>>
TreeXHybridSMMD<T>::SharedFloatDatasetIfNeeded() {
  SCANN_ASSIGN_OR_RETURN(
      shared_ptr<const DenseDataset<float>> inherited_res,
      SingleMachineSearcherBase<T>::SharedFloatDatasetIfNeeded());
  if (inherited_res != nullptr) return inherited_res;

  vector<const DenseDataset<float>*> datasets(datapoints_by_token_.size());
  for (int i = 0; i < datasets.size(); i++) {
    auto ptr_or = leaf_searchers_[i]->SharedFloatDatasetIfNeeded();
    SCANN_RETURN_IF_ERROR(ptr_or.status());
    datasets[i] = ptr_or->get();
  }
  SCANN_ASSIGN_OR_RETURN(const int dataset_size,
                         UntypedSingleMachineSearcherBase::DatasetSize());
  const auto get_dataset = [&](int leaf_idx) { return datasets[leaf_idx]; };

  SCANN_ASSIGN_OR_RETURN(
      vector<float> storage,
      CombineLeafDatasets<float>(dataset_size, "float32", datapoints_by_token_,
                                 get_dataset));
  if (storage.empty()) return shared_ptr<const DenseDataset<float>>(nullptr);
  return std::make_shared<const DenseDataset<float>>(std::move(storage),
                                                     dataset_size);
}

template <typename T>
Status TreeXHybridSMMD<T>::PreprocessQueryIntoParamsUnlocked(
    const DatapointPtr<T>& query, SearchParameters& search_params) const {
  const auto& params =
      search_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  vector<int32_t> centers_to_search;
  if (params) {
    if (params->pre_tokenization_enabled()) {
      centers_to_search.assign(params->leaf_tokens_to_search().begin(),
                               params->leaf_tokens_to_search().end());
    } else {
      const auto* kmeans_tokenizer =
          dynamic_cast<const KMeansTreePartitioner<T>*>(query_tokenizer_.get());
      if (!kmeans_tokenizer)
        return InvalidArgumentError(
            "num_partitions_to_search_override is > 0, but the tokenizer is "
            "not a KMeansTreePartitioner.");

      int max_centers_override = params->num_partitions_to_search_override();
      SCANN_RETURN_IF_ERROR(
          kmeans_tokenizer->TokensForDatapointWithSpillingAndOverride(
              query, max_centers_override, &centers_to_search));
    }
  } else {
    SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
        query, &centers_to_search));
  }

  search_params.set_unlocked_query_preprocessing_results(
      {make_unique<CentersToSearch>(std::move(centers_to_search))});
  return OkStatus();
}

template <typename T>
vector<uint32_t> TreeXHybridSMMD<T>::SizeByPartition() const {
  return ::research_scann::SizeByPartition(datapoints_by_token_);
}

template <typename T>
Status TreeXHybridSMMD<T>::InitializeHealthStats() {
  return stats_collector_.Initialize(*this);
}

template <typename T>
StatusOr<typename TreeXHybridSMMD<T>::HealthStats>
TreeXHybridSMMD<T>::GetHealthStats() const {
  return stats_collector_.GetHealthStats();
}

SCANN_INSTANTIATE_TREE_X_HYBRID_SMMD();

}  // namespace research_scann
