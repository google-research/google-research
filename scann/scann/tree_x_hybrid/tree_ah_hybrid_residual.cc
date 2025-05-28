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



#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/serialization.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_serialize.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_status_builder.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/tree_x_hybrid/internal/batching.h"
#include "scann/tree_x_hybrid/internal/utils.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

namespace research_scann {

using asymmetric_hashing2::AsymmetricHashingOptionalParameters;

Status TreeAHHybridResidual::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (leaf_searchers_.empty()) return OkStatus();
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

void TreeAHHybridResidual::DisableCrowdingImpl() {
  for (auto& ls : leaf_searchers_) {
    ls->DisableCrowding();
  }
}

Status TreeAHHybridResidual::CheckBuildLeafSearchersPreconditions(
    const AsymmetricHasherConfig& config,
    const KMeansTreeLikePartitioner<float>& partitioner) const {
  if (!leaf_searchers_.empty()) {
    return FailedPreconditionErrorBuilder().LogError()
           << "BuildLeafSearchers must not be called more than once per "
              "instance.";
  }
  if (partitioner.query_tokenization_distance()
          ->specially_optimized_distance_tag() !=
      DistanceMeasure::DOT_PRODUCT) {
    return InvalidArgumentErrorBuilder().LogError()
           << "For TreeAHHybridResidual, partitioner must use "
              "DotProductDistance for query tokenization.";
  }
  if (config.partition_level_confidence_interval_stdevs() != 0.0) {
    LOG(WARNING) << "partition_level_confidence_interval_stdevs has no effect.";
  }
  return OkStatus();
}

namespace {
vector<uint32_t> OrderLeafTokensByCenterNorm(
    const KMeansTreeLikePartitioner<float>& partitioner) {
  vector<float> norms(partitioner.n_tokens());
  std::function<void(const KMeansTreeNode&)> impl =
      [&](const KMeansTreeNode& node) {
        if (node.IsLeaf()) {
          const int32_t leaf_id = node.LeafId();
          DCHECK_LT(leaf_id, norms.size());
          norms[leaf_id] = SquaredL2Norm(node.cur_node_center());
        } else {
          for (const KMeansTreeNode& child : node.Children()) {
            impl(child);
          }
        }
      };

  impl(*partitioner.kmeans_tree()->root());
  vector<uint32_t> perm(norms.size());
  std::iota(perm.begin(), perm.end(), 0U);
  ZipSortBranchOptimized(std::greater<float>(), norms.begin(), norms.end(),
                         perm.begin(), perm.end());
  return perm;
}

template <typename GetResidual>
StatusOr<DenseDataset<float>> ComputeResidualsImpl(
    const DenseDataset<float>& dataset, GetResidual get_residual,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    ThreadPool* parallelization_pool) {
  vector<uint32_t> tokens_by_datapoint(dataset.size());
  for (uint32_t token : Seq(datapoints_by_token.size())) {
    for (DatapointIndex dp_idx : datapoints_by_token[token]) {
      tokens_by_datapoint[dp_idx] = token;
    }
  }

  vector<float> residuals_storage;
  auto loop_body = [&](size_t dp_idx, bool is_first_dp)
                       SCANN_INLINE_LAMBDA -> Status {
    const uint32_t token = tokens_by_datapoint[dp_idx];
    SCANN_ASSIGN_OR_RETURN(auto residual, get_residual(dataset[dp_idx], token));

    if (is_first_dp) {
      residuals_storage =
          vector<float>(dataset.size() * residual.dimensionality());
    } else {
      DCHECK_EQ(residuals_storage.size(),
                dataset.size() * residual.dimensionality());
    }
    std::copy(residual.values().begin(), residual.values().end(),
              residuals_storage.begin() + dp_idx * residual.dimensionality());
    return OkStatus();
  };

  if (dataset.empty()) return DenseDataset<float>();
  SCANN_RETURN_IF_ERROR(loop_body(0, true));
  SCANN_RETURN_IF_ERROR(ParallelForWithStatus<1>(
      Seq(1, dataset.size()), parallelization_pool,
      [&](size_t dp_idx) { return loop_body(dp_idx, false); }));
  return DenseDataset<float>(std::move(residuals_storage), dataset.size());
}

}  // namespace

StatusOr<DenseDataset<float>> TreeAHHybridResidual::ComputeResiduals(
    const DenseDataset<float>& dataset,
    const DenseDataset<float>& kmeans_centers,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    ThreadPool* parallelization_pool) {
  DCHECK(!kmeans_centers.empty());
  DCHECK_EQ(kmeans_centers.size(), datapoints_by_token.size());
  DCHECK_EQ(kmeans_centers.dimensionality(), dataset.dimensionality());
  const size_t dimensionality = dataset.dimensionality();
  auto get_residual = [&](const DatapointPtr<float>& dptr,
                          const int32_t token) -> StatusOr<Datapoint<float>> {
    ConstSpan<float> datapoint = dptr.values_span();
    ConstSpan<float> center = kmeans_centers[token].values_span();
    DCHECK_EQ(center.size(), dimensionality);
    DCHECK_EQ(datapoint.size(), dimensionality);
    Datapoint<float> result;
    result.mutable_values()->resize(dimensionality);
    MutableSpan<float> result_values(*result.mutable_values());
    for (size_t d : Seq(dimensionality)) {
      result_values[d] = datapoint[d] - center[d];
    }
    return result;
  };

  return ComputeResidualsImpl(dataset, get_residual, datapoints_by_token,
                              parallelization_pool);
}

StatusOr<DenseDataset<float>> TreeAHHybridResidual::ComputeResiduals(
    const DenseDataset<float>& dataset,
    const KMeansTreeLikePartitioner<float>* partitioner,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    ThreadPool* parallelization_pool) {
  auto get_residual = [&](const DatapointPtr<float>& dptr,
                          const int32_t token) -> StatusOr<Datapoint<float>> {
    return partitioner->ResidualizeToFloat(dptr, token);
  };
  return ComputeResidualsImpl(dataset, get_residual, datapoints_by_token,
                              parallelization_pool);
}

Status TreeAHHybridResidual::PreprocessQueryIntoParamsUnlocked(
    const DatapointPtr<float>& query, SearchParameters& search_params) const {
  Datapoint<float> projected_query_storage;
  SCANN_ASSIGN_OR_RETURN(DatapointPtr<float> maybe_projected_query,
                         MaybeProjectQuery(query, &projected_query_storage));
  const auto& params =
      search_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  int32_t max_centers_override = 0;

  if (params && params->num_partitions_to_search_override()) {
    max_centers_override = params->num_partitions_to_search_override();
  }

  vector<pair<DatapointIndex, float>> centers_to_search;
  if (params && params->pre_tokenization_with_distances_enabled()) {
    centers_to_search.assign(params->centers_to_search().begin(),
                             params->centers_to_search().end());
  } else {
    SCANN_RETURN_IF_ERROR(
        maybe_projected_query_tokenizer_->TokensForDatapointWithSpilling(
            maybe_projected_query, max_centers_override, &centers_to_search));
  }

  SCANN_ASSIGN_OR_RETURN(auto shared_lookup_table,
                         asymmetric_queryer_->CreateLookupTable(
                             maybe_projected_query, lookup_type_tag_,
                             fixed_point_lut_conversion_options_));
  search_params.set_unlocked_query_preprocessing_results(
      {make_unique<UnlockedTreeAHHybridResidualPreprocessingResults>(
          std::move(projected_query_storage), std::move(centers_to_search),
          std::move(shared_lookup_table))});
  return OkStatus();
}

Status TreeAHHybridResidual::AddLeafSearcher() {
  auto hashed_partition = make_unique<DenseDataset<uint8_t>>();
  leaf_searchers_.push_back(make_unique<asymmetric_hashing2::Searcher<float>>(
      nullptr, std::move(hashed_partition), *searcher_options_,
      default_pre_reordering_num_neighbors(),
      default_pre_reordering_epsilon()));
  datapoints_by_token_.push_back(std::vector<DatapointIndex>());

  return OkStatus();
}

void TreeAHHybridResidual::MaybeInitializeProjection() {
  auto projecting_decorator =
      dynamic_cast<const KMeansTreeProjectingDecorator<float>*>(
          query_tokenizer_.get());
  if (projecting_decorator) {
    projection_ = projecting_decorator->projection().get();
    maybe_projected_query_tokenizer_ =
        projecting_decorator->base_kmeans_tree_partitioner();
  } else {
    maybe_projected_query_tokenizer_ = query_tokenizer_.get();
  }
}

StatusOr<DatapointPtr<float>> TreeAHHybridResidual::MaybeProjectQuery(
    const DatapointPtr<float>& query,
    Datapoint<float>* projected_query_storage) const {
  if (!projection_) return query;
  SCANN_RETURN_IF_ERROR(
      projection_->ProjectInput(query, projected_query_storage));
  return projected_query_storage->ToPtr();
}

StatusOr<const TypedDataset<float>*>
TreeAHHybridResidual::MaybeProjectQueriesBatched(
    const TypedDataset<float>* queries,
    DenseDataset<float>* projected_query_storage) const {
  if (!projection_) return queries;
  Datapoint<float> tmp;
  projected_query_storage->clear();
  for (const DatapointPtr<float>& query : *queries) {
    SCANN_RETURN_IF_ERROR(projection_->ProjectInput(query, &tmp));
    projected_query_storage->AppendOrDie(tmp.ToPtr(), "");
  }
  return projected_query_storage;
}

Status TreeAHHybridResidual::PreprocessQueryIntoParamsUnlocked(
    const DatapointPtr<float>& query,
    vector<pair<DatapointIndex, float>>& tokens_to_search,
    SearchParameters& search_params) const {
  Datapoint<float> projected_query_storage;
  SCANN_ASSIGN_OR_RETURN(DatapointPtr<float> maybe_projected_query,
                         MaybeProjectQuery(query, &projected_query_storage));
  SCANN_ASSIGN_OR_RETURN(asymmetric_hashing2::LookupTable shared_lookup_table,
                         asymmetric_queryer_->CreateLookupTable(
                             maybe_projected_query, lookup_type_tag_));
  search_params.set_unlocked_query_preprocessing_results(
      {make_unique<UnlockedTreeAHHybridResidualPreprocessingResults>(
          std::move(projected_query_storage), std::move(tokens_to_search),
          std::move(shared_lookup_table))});
  return OkStatus();
}

Status TreeAHHybridResidual::BuildLeafSearchers(
    const AsymmetricHasherConfig& config,
    unique_ptr<KMeansTreeLikePartitioner<float>> partitioner,
    shared_ptr<const asymmetric_hashing2::Model<float>> ah_model,
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    BuildLeafSearchersOptions opts) {
  DCHECK(partitioner);
  SCANN_RETURN_IF_ERROR(
      CheckBuildLeafSearchersPreconditions(config, *partitioner));
  if (config.projection().has_ckmeans_config() &&
      config.projection().ckmeans_config().need_learning()) {
    return FailedPreconditionError(
        "Cannot learn ckmeans when building a TreeAHHybridResidual with "
        "pre-training.");
  }
  SCANN_ASSIGN_OR_RETURN(shared_ptr<const ChunkingProjection<float>> projector,
                         ah_model->GetProjection(config.projection()));
  SCANN_ASSIGN_OR_RETURN(auto quantization_distance,
                         GetDistanceMeasure(config.quantization_distance()));
  lookup_type_tag_ = config.lookup_type();
  std::function<StatusOr<DatapointPtr<uint8_t>>(DatapointIndex, int32_t,
                                                Datapoint<uint8_t>*)>
      get_hashed_datapoint;
  Datapoint<uint8_t> hashed_dp_storage;
  auto indexer = make_shared<asymmetric_hashing2::Indexer<float>>(
      projector, quantization_distance, ah_model);
  const auto dataset =
      dynamic_cast<const DenseDataset<float>*>(this->dataset());

  for (auto& vec : datapoints_by_token) {
    for (DatapointIndex token : vec) {
      num_datapoints_ = std::max(token + 1, num_datapoints_);
    }
    leaf_size_upper_bound_ =
        std::max<uint32_t>(leaf_size_upper_bound_, vec.size());
  }
  SCANN_ASSIGN_OR_RETURN(
      datapoints_by_token_disjoint_,
      ValidateDatapointsByToken(datapoints_by_token, num_datapoints_));

  const DenseDataset<uint8_t>* hashed_dataset = opts.hashed_dataset;
  const DenseDataset<uint8_t>* soar_hashed_dataset = opts.soar_hashed_dataset;
  if (hashed_dataset && !soar_hashed_dataset &&
      !datapoints_by_token_disjoint_) {
    return InvalidArgumentError(
        "If hashed_dataset is set and SOAR database spilling is enabled, "
        "soar_hashed_dataset must be set.");
  }
  shared_ptr<const Projection<float>> partitioning_projection;
  if (auto projecting_decorator =
          dynamic_cast<const KMeansTreeProjectingDecorator<float>*>(
              partitioner.get())) {
    partitioning_projection = projecting_decorator->projection();
  }
  if (hashed_dataset) {
    if (datapoints_by_token_disjoint_) {
      get_hashed_datapoint = [hashed_dataset](DatapointIndex i, int32_t token,
                                              Datapoint<uint8_t>* storage)
          -> StatusOr<DatapointPtr<uint8_t>> { return (*hashed_dataset)[i]; };
    } else {
      get_hashed_datapoint =
          [hashed_dataset, soar_hashed_dataset](
              DatapointIndex i, int32_t token,
              Datapoint<uint8_t>* storage) -> StatusOr<DatapointPtr<uint8_t>> {
        const bool token_is_secondary =
            strings::KeyToInt32(soar_hashed_dataset->GetDocid(i)) == token;
        if (token_is_secondary) {
          return (*soar_hashed_dataset)[i];
        } else {
          return (*hashed_dataset)[i];
        }
      };
    }
  } else {
    if (!this->dataset()) {
      return InvalidArgumentError(
          "At least one of dataset/hashed_dataset must be non-null in "
          "TreeAHHybridResidual::BuildLeafSearchersPreTrained.  If database "
          "spilling is used, dataset must be non-null and hashed_dataset is "
          "ignored.");
    }
    get_hashed_datapoint =
        [&](DatapointIndex i, int32_t token,
            Datapoint<uint8_t>* storage) -> StatusOr<DatapointPtr<uint8_t>> {
      DCHECK(dataset);
      DatapointPtr<float> original = (*dataset)[i];
      DatapointPtr<float> maybe_projected_original = original;
      Datapoint<float> projected_original_storage;
      if (partitioning_projection) {
        SCANN_RETURN_IF_ERROR(partitioning_projection->ProjectInput(
            original, &projected_original_storage));
        maybe_projected_original = projected_original_storage.ToPtr();
      }
      SCANN_ASSIGN_OR_RETURN(Datapoint<float> residual,
                             partitioner->ResidualizeToFloat(original, token));
      if (std::isnan(config.noise_shaping_threshold())) {
        SCANN_RETURN_IF_ERROR(indexer->Hash(residual.ToPtr(), storage));
      } else {
        SCANN_RETURN_IF_ERROR(indexer->HashWithNoiseShaping(
            residual.ToPtr(), maybe_projected_original, storage,
            {.threshold = config.noise_shaping_threshold()}));
      }
      return storage->ToPtr();
    };
  }

  shared_ptr<DistanceMeasure> lookup_distance =
      std::make_shared<DotProductDistance>();
  asymmetric_queryer_ =
      std::make_shared<asymmetric_hashing2::AsymmetricQueryer<float>>(
          projector, lookup_distance, ah_model);
  leaf_searchers_ =
      vector<unique_ptr<asymmetric_hashing2::SearcherBase<float>>>(
          datapoints_by_token.size());

  searcher_options_ = make_unique<asymmetric_hashing2::SearcherOptions<float>>(
      asymmetric_queryer_, indexer);
  searcher_options_->set_asymmetric_lookup_type(lookup_type_tag_);
  searcher_options_->set_noise_shaping_threshold(
      config.noise_shaping_threshold());

  auto build_leaf_for_token = [&](size_t token) -> Status {
    const absl::Time token_start = absl::Now();
    auto hashed_partition = make_unique<DenseDataset<uint8_t>>();
    if (asymmetric_queryer_->quantization_scheme() ==
        AsymmetricHasherConfig::PRODUCT_AND_PACK) {
      hashed_partition->set_packing_strategy(HashedItem::NIBBLE);
    }
    Datapoint<uint8_t> dp;
    Datapoint<uint8_t> hashed_storage;
    for (DatapointIndex dp_index : datapoints_by_token[token]) {
      auto status_or_hashed_dptr =
          get_hashed_datapoint(dp_index, token, &hashed_storage);
      SCANN_ASSIGN_OR_RETURN(auto hashed_dptr, status_or_hashed_dptr);
      auto local_status = hashed_partition->Append(hashed_dptr, "");
      SCANN_RETURN_IF_ERROR(local_status);
    }

    leaf_searchers_[token] = make_unique<asymmetric_hashing2::Searcher<float>>(
        nullptr, std::move(hashed_partition), *searcher_options_,
        default_pre_reordering_num_neighbors(),
        default_pre_reordering_epsilon());
    if (!leaf_searchers_[token]->needs_hashed_dataset()) {
      leaf_searchers_[token]->ReleaseHashedDataset();
    }
    VLOG(1) << "Built leaf searcher " << token + 1 << " of "
            << datapoints_by_token.size()
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
    return OkStatus();
  };

  SCANN_RETURN_IF_ERROR(ParallelForWithStatus<1>(
      IndicesOf(datapoints_by_token), opts.pool, build_leaf_for_token));

  datapoints_by_token_ = std::move(datapoints_by_token);
  leaf_tokens_by_norm_ = OrderLeafTokensByCenterNorm(*partitioner);
  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  query_tokenizer_ = std::move(partitioner);
  MaybeInitializeProjection();
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute());
  }
  if (config.use_global_topn() &&
      config.lookup_type() == AsymmetricHasherConfig::INT8_LUT16 &&
      !config.use_normalized_residual_quantization()) {
    enable_global_topn_ = true;
  }
  return OkStatus();
}

Status TreeAHHybridResidual::FindNeighborsImpl(const DatapointPtr<float>& query,
                                               const SearchParameters& params,
                                               NNResultsVector* result) const {
  auto query_preprocessing_results =
      params.unlocked_query_preprocessing_results<
          UnlockedTreeAHHybridResidualPreprocessingResults>();
  if (query_preprocessing_results) {
    SCANN_RETURN_IF_ERROR(
        ValidateTokenList(query_preprocessing_results->centers_to_search(),
                          query_tokenizer_ != nullptr));
    const DatapointPtr<float> maybe_projected_query =
        projection_ ? (query_preprocessing_results->projected_query()) : query;
    return FindNeighborsInternal1(
        maybe_projected_query, params,
        query_preprocessing_results->centers_to_search(), result);
  }

  Datapoint<float> projected_query_storage;
  SCANN_ASSIGN_OR_RETURN(DatapointPtr<float> maybe_projected_query,
                         MaybeProjectQuery(query, &projected_query_storage));
  int num_centers = 0;
  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  if (tree_x_params) {
    int center_override = tree_x_params->num_partitions_to_search_override();
    if (center_override > 0) num_centers = center_override;
  }
  vector<pair<DatapointIndex, float>> centers_to_search_storage;
  ConstSpan<pair<DatapointIndex, float>> centers_to_search;
  if (tree_x_params &&
      tree_x_params->pre_tokenization_with_distances_enabled()) {
    centers_to_search = tree_x_params->centers_to_search();
    SCANN_RETURN_IF_ERROR(
        ValidateTokenList(centers_to_search, query_tokenizer_ != nullptr));
  } else {
    SCANN_RETURN_IF_ERROR(
        maybe_projected_query_tokenizer_->TokensForDatapointWithSpilling(
            maybe_projected_query, num_centers, &centers_to_search_storage));
    centers_to_search = centers_to_search_storage;
  }
  return FindNeighborsInternal1(maybe_projected_query, params,
                                centers_to_search, result);
}

Status TreeAHHybridResidual::BuildStreamingLeafSearchers(
    const AsymmetricHasherConfig& config, size_t n_tokens,
    ConstSpan<pair<DatapointIndex, float>> query_tokens,
    shared_ptr<KMeansTreeLikePartitioner<float>> partitioner,
    shared_ptr<const asymmetric_hashing2::Model<float>> ah_model,
    bool streaming_result,
    std::function<
        StatusOr<std::unique_ptr<asymmetric_hashing2::SearcherBase<float>>>(
            int token, float distance_to_center,
            asymmetric_hashing2::SearcherOptions<float> opts)>
        leaf_searcher_builder) {
  DCHECK(partitioner);
  SCANN_RETURN_IF_ERROR(
      CheckBuildLeafSearchersPreconditions(config, *partitioner));
  if (config.projection().has_ckmeans_config() &&
      config.projection().ckmeans_config().need_learning()) {
    return InvalidArgumentError(
        "Cannot learn ckmeans when building a TreeAHHybridResidual with "
        "pre-training.");
  }

  SCANN_ASSIGN_OR_RETURN(shared_ptr<const ChunkingProjection<float>> projector,
                         ah_model->GetProjection(config.projection()));
  lookup_type_tag_ = config.lookup_type();
  SCANN_ASSIGN_OR_RETURN(shared_ptr<DistanceMeasure> quantization_distance,
                         GetDistanceMeasure(config.quantization_distance()));
  auto indexer = make_shared<asymmetric_hashing2::Indexer<float>>(
      projector, quantization_distance, ah_model);
  shared_ptr<DistanceMeasure> lookup_distance =
      std::make_shared<DotProductDistance>();
  asymmetric_queryer_ =
      std::make_shared<asymmetric_hashing2::AsymmetricQueryer<float>>(
          projector, lookup_distance, ah_model);

  searcher_options_ = make_unique<asymmetric_hashing2::SearcherOptions<float>>(
      asymmetric_queryer_, indexer);
  searcher_options_->set_asymmetric_lookup_type(lookup_type_tag_);
  searcher_options_->set_noise_shaping_threshold(
      config.noise_shaping_threshold());

  leaf_searchers_.resize(n_tokens);
  for (auto [token, distance_to_center] : query_tokens) {
    const auto token_start_time = absl::Now();

    SCANN_RET_CHECK_LT(token, n_tokens);
    SCANN_ASSIGN_OR_RETURN(
        leaf_searchers_[token],
        leaf_searcher_builder(token, distance_to_center, *searcher_options_));

    VLOG(1) << "Built leaf searcher " << token + 1 << " of " << n_tokens
            << " in " << absl::ToDoubleSeconds(absl::Now() - token_start_time)
            << " sec.";
  }
  is_streaming_result_ = streaming_result;

  query_tokenizer_ = std::move(partitioner);
  MaybeInitializeProjection();
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute());
  }

  return OkStatus();
}

namespace {
using QueryForLeaf = tree_x_internal::QueryForResidualLeaf;
using BatchedGlobalTopNData = tree_x_internal::BatchedGlobalTopNData;

vector<std::vector<QueryForLeaf>> InvertCentersToSearch(
    ConstSpan<vector<pair<DatapointIndex, float>>> centers_to_search,
    size_t num_centers) {
  vector<std::vector<QueryForLeaf>> result(num_centers);
  for (DatapointIndex query_index : IndicesOf(centers_to_search)) {
    ConstSpan<pair<DatapointIndex, float>> cur_query_centers =
        centers_to_search[query_index];
    for (const auto& center : cur_query_centers) {
      result[center.first].emplace_back(query_index, center.second);
    }
  }
  return result;
}

template <typename TopN>
inline void AssignResults(TopN* top_n, NNResultsVector* results) {
  top_n->FinishUnsorted(results);
}

}  // namespace

Status TreeAHHybridResidual::FindNeighborsBatchedImpl(
    const TypedDataset<float>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  vector<int32_t> centers_override(queries.size());
  bool centers_overridden = false;
  for (int i = 0; i < queries.size(); i++) {
    auto tree_x_params =
        params[i]
            .searcher_specific_optional_parameters<TreeXOptionalParameters>();
    if (tree_x_params) {
      int center_override = tree_x_params->num_partitions_to_search_override();
      if (center_override > 0) {
        centers_override[i] = center_override;
        centers_overridden = true;
      }
    }
  }

  DenseDataset<float> projected_query_storage;
  SCANN_ASSIGN_OR_RETURN(
      const TypedDataset<float>* maybe_projected_queries_ptr,
      MaybeProjectQueriesBatched(&queries, &projected_query_storage));
  const TypedDataset<float>& maybe_projected_queries =
      *maybe_projected_queries_ptr;

  vector<vector<pair<DatapointIndex, float>>> centers_to_search(queries.size());
  if (centers_overridden)
    SCANN_RETURN_IF_ERROR(
        maybe_projected_query_tokenizer_->TokensForDatapointWithSpillingBatched(
            maybe_projected_queries, centers_override,
            MakeMutableSpan(centers_to_search)));
  else
    SCANN_RETURN_IF_ERROR(
        maybe_projected_query_tokenizer_->TokensForDatapointWithSpillingBatched(
            maybe_projected_queries, vector<int32_t>(),
            MakeMutableSpan(centers_to_search)));
  if (!tree_x_internal::SupportsLowLevelBatching(maybe_projected_queries,
                                                 params) ||
      !leaf_searchers_[0]->lut16_ ||
      leaf_searchers_[0]->opts_.quantization_scheme() ==
          AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    for (size_t i = 0; i < centers_to_search.size(); ++i) {
      SCANN_RETURN_IF_ERROR(
          FindNeighborsInternal1(maybe_projected_queries[i], params[i],
                                 centers_to_search[i], &results[i]));
    }
    return OkStatus();
  }
  const uint8_t global_topn_shift = GlobalTopNShift();
  auto queries_by_leaf = InvertCentersToSearch(
      centers_to_search, maybe_projected_query_tokenizer_->n_tokens());
  if (global_topn_shift > 0) {
    const size_t nq = maybe_projected_queries.size();
    vector<FastTopNeighbors<float>> top_ns;
    top_ns.reserve(nq);
    ConstSpan<FastTopNeighbors<float>> topn_span(top_ns);

    vector<vector<uint8_t>> lookup_tables(nq);
    vector<const uint8_t*> lookup_addrs(nq);
    vector<float> multipliers(nq);
    bool int16_accum_ok = true;

    for (const auto& [i, p] : Enumerate(params)) {
      SCANN_ASSIGN_OR_RETURN(auto lut,
                             asymmetric_queryer_->CreateLookupTable(
                                 maybe_projected_queries[i], lookup_type_tag_,
                                 fixed_point_lut_conversion_options_));
      lookup_tables[i] = std::move(lut.int8_lookup_table);
      lookup_addrs[i] = lookup_tables[i].data();
      multipliers[i] = lut.fixed_point_multiplier;
      int16_accum_ok = int16_accum_ok && lut.can_use_int16_accumulator;

      top_ns.emplace_back(
          NumNeighborsWithSpillingMultiplier(p.pre_reordering_num_neighbors()),
          p.pre_reordering_epsilon());
    }
    vector<BatchedGlobalTopNData> data;
    data.reserve(datapoints_by_token_.size() + 1);
    size_t num_blocks = 0;
    for (size_t leaf_token : IndicesOf(datapoints_by_token_)) {
      if (!leaf_tokens_by_norm_.empty())
        leaf_token = leaf_tokens_by_norm_[leaf_token];
      if (queries_by_leaf[leaf_token].empty()) continue;

      const asymmetric_hashing2::PackedDataset& packed =
          leaf_searchers_[leaf_token]->packed_dataset();
      data.emplace_back(leaf_token, packed.num_datapoints,
                        packed.bit_packed_data.data(),
                        queries_by_leaf[leaf_token]);
      DCHECK(packed.num_blocks == 0 || num_blocks == 0 ||
             num_blocks == packed.num_blocks);
      num_blocks = std::max<size_t>(num_blocks, packed.num_blocks);
    }
    data.push_back(data.back());

    constexpr const int kMaxBatch = 3;
    std::array<FastTopNeighbors<float>*, kMaxBatch> top_arr;
    std::array<const uint8_t*, kMaxBatch> lut_arr;
    std::array<float, kMaxBatch> mult_arr;
    std::array<float, kMaxBatch> bias_arr;

    for (size_t i = 0; i + 1 < data.size(); i++) {
      BatchedGlobalTopNData cur_data = data[i];

      asymmetric_hashing_internal::LUT16ArgsTopN<float> args;
      args.packed_dataset = cur_data.ah_data;
      args.next_partition = data[i + 1].ah_data;
      args.prefetch_strategy =
          asymmetric_hashing_internal::PrefetchStrategy::kSmart;
      args.num_32dp_simd_iters = DivRoundUp(cur_data.leaf_size, 32);
      args.num_blocks = num_blocks;
      args.first_dp_index = cur_data.leaf_index << global_topn_shift;
      args.num_datapoints = cur_data.leaf_size;

      size_t cur_numq = cur_data.queries.size();
      for (size_t batch_start = 0; batch_start < cur_numq;) {
        const size_t cur_batch_size = [](size_t left) -> size_t {
          if (left <= kMaxBatch) return left;
          if (left >= 2 * kMaxBatch) return kMaxBatch;
          return left / 2;
        }(cur_numq - batch_start);
        for (size_t j = 0; j < cur_batch_size; j++) {
          uint32_t query_idx = cur_data.queries[batch_start + j].query_index;
          top_arr[j] = &top_ns[query_idx];
          lut_arr[j] = lookup_addrs[query_idx];
          mult_arr[j] = multipliers[query_idx];
          bias_arr[j] = cur_data.queries[batch_start + j].distance_to_center;
        }
        args.fast_topns = {top_arr.data(), cur_batch_size};
        args.lookups = {lut_arr.data(), cur_batch_size};
        args.fixed_point_multipliers = {mult_arr.data(), cur_batch_size};
        args.biases = {bias_arr.data(), cur_batch_size};
        asymmetric_hashing_internal::LUT16Interface::GetTopFloatDistances(args);
        batch_start += cur_batch_size;
      }
    }
    const uint32_t local_idx_mask = (1u << global_topn_shift) - 1;
    for (size_t query_index : IndicesOf(results)) {
      top_ns[query_index].FinishUnsorted(&results[query_index]);
      for (pair<DatapointIndex, float>& idx_dis : results[query_index]) {
        uint32_t partition_idx = idx_dis.first >> global_topn_shift;
        uint32_t local_idx = idx_dis.first & local_idx_mask;
        idx_dis.first = datapoints_by_token_[partition_idx][local_idx];
      }
      if (!datapoints_by_token_disjoint_) {
        DeduplicateDatabaseSpilledResults(
            &results[query_index],
            params[query_index].pre_reordering_num_neighbors());
      }
    }
    return OkStatus();
  }

  vector<shared_ptr<const SearcherSpecificOptionalParameters>> lookup_tables(
      maybe_projected_queries.size());
  for (size_t i : IndicesOf(maybe_projected_queries)) {
    SCANN_ASSIGN_OR_RETURN(auto lut,
                           asymmetric_queryer_->CreateLookupTable(
                               maybe_projected_queries[i], lookup_type_tag_,
                               fixed_point_lut_conversion_options_));
    lookup_tables[i] =
        make_shared<AsymmetricHashingOptionalParameters>(std::move(lut));
  }
  vector<FastTopNeighbors<float>> top_ns;
  vector<FastTopNeighbors<float>::Mutator> mutators(params.size());
  top_ns.reserve(params.size());
  for (const auto& [idx, p] : Enumerate(params)) {
    top_ns.emplace_back(
        NumNeighborsWithSpillingMultiplier(p.pre_reordering_num_neighbors()),
        p.pre_reordering_epsilon());
    top_ns[idx].AcquireMutator(&mutators[idx]);
  }
  vector<NNResultsVector> leaf_results;
  for (size_t leaf_token : Seq(datapoints_by_token_.size())) {
    if (!leaf_tokens_by_norm_.empty())
      leaf_token = leaf_tokens_by_norm_[leaf_token];
    ConstSpan<QueryForLeaf> queries_for_cur_leaf = queries_by_leaf[leaf_token];
    if (queries_for_cur_leaf.empty()) continue;
    vector<SearchParameters> leaf_params =
        tree_x_internal::CreateParamsSubsetForLeaf<QueryForLeaf>(
            top_ns, lookup_tables, queries_for_cur_leaf);
    auto get_query = [&maybe_projected_queries,
                      &queries_for_cur_leaf](DatapointIndex i) {
      return maybe_projected_queries[queries_for_cur_leaf[i].query_index];
    };
    leaf_results.clear();
    leaf_results.resize(leaf_params.size());
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[leaf_token]->FindNeighborsBatchedInternal(
            get_query, leaf_params, MakeMutableSpan(leaf_results)));

    ConstSpan<DatapointIndex> local_to_global_index =
        datapoints_by_token_[leaf_token];
    for (size_t j = 0; j < queries_for_cur_leaf.size(); ++j) {
      const DatapointIndex cur_query_index =
          queries_for_cur_leaf[j].query_index;
      tree_x_internal::AddLeafResultsToTopN(
          local_to_global_index, queries_for_cur_leaf[j].distance_to_center,
          leaf_results[j], &mutators[cur_query_index]);
    }
  }
  for (size_t query_index : IndicesOf(results)) {
    mutators[query_index].Release();
    top_ns[query_index].FinishUnsorted(&results[query_index]);
    if (!datapoints_by_token_disjoint_) {
      DeduplicateDatabaseSpilledResults(
          &results[query_index],
          params[query_index].pre_reordering_num_neighbors());
    }
  }
  return OkStatus();
}

Status TreeAHHybridResidual::ValidateTokenList(
    ConstSpan<pair<DatapointIndex, float>> centers_to_search,
    bool check_oob) const {
  absl::flat_hash_set<int32_t> duplicate_checker;

  for (const auto& [token, distance] : centers_to_search) {
    if (!duplicate_checker.insert(token).second) {
      return InvalidArgumentError(
          absl::StrCat("Duplicate token:  ", token, "."));
    }

    if (check_oob && token >= leaf_searchers_.size()) {
      return InvalidArgumentError(
          "Query token out of range of database tokens (got %d, "
          "expected in the range [0, %d).",
          static_cast<int>(token), static_cast<int>(leaf_searchers_.size()));
    }
  }

  return OkStatus();
}

Status TreeAHHybridResidual::FindNeighborsInternal1(
    const DatapointPtr<float>& maybe_projected_query,
    const SearchParameters& params,
    ConstSpan<pair<DatapointIndex, float>> centers_to_search,
    NNResultsVector* result) const {
  auto query_preprocessing_results =
      params.unlocked_query_preprocessing_results<
          UnlockedTreeAHHybridResidualPreprocessingResults>();
  shared_ptr<AsymmetricHashingOptionalParameters> lookup_table;
  if (query_preprocessing_results) {
    DCHECK(query_preprocessing_results->lookup_table());
    lookup_table = query_preprocessing_results->lookup_table();
  } else {
    lookup_table = make_shared<AsymmetricHashingOptionalParameters>(
        asymmetric_queryer_
            ->CreateLookupTable(maybe_projected_query, lookup_type_tag_,
                                fixed_point_lut_conversion_options_)
            .value());
  }

  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  }

  const uint8_t global_topn_shift = GlobalTopNShift();
  if (global_topn_shift > 0 &&
      lookup_table->precomputed_lookup_table().can_use_int16_accumulator) {
    FastTopNeighbors<float> top_n(NumNeighborsWithSpillingMultiplier(
                                      params.pre_reordering_num_neighbors()),
                                  params.pre_reordering_epsilon());
    DCHECK(result);

    std::array<FastTopNeighbors<float>*, 1> tops = {&top_n};
    std::array<const uint8_t*, 1> lookups = {
        lookup_table->precomputed_lookup_table().int8_lookup_table.data()};
    std::array<float, 1> multipliers = {
        lookup_table->precomputed_lookup_table().fixed_point_multiplier};

    SearchParameters leaf_params;
    std::array<RestrictAllowlistConstView, 1> allowlists = {
        RestrictAllowlistConstView()};

    size_t num_blocks = 0;

    vector<pair<const uint8_t*, uint32_t>> center_data(
        centers_to_search.size() + 1);
    for (size_t i : IndicesOf(centers_to_search)) {
      const asymmetric_hashing2::PackedDataset& packed =
          leaf_searchers_[centers_to_search[i].first]->packed_dataset();
      center_data[i] = {packed.bit_packed_data.data(), packed.num_datapoints};

      DCHECK(packed.num_blocks == 0 || num_blocks == 0 ||
             num_blocks == packed.num_blocks);
      num_blocks = std::max<size_t>(num_blocks, packed.num_blocks);
    }

    if (!centers_to_search.empty()) {
      center_data.back() = center_data[center_data.size() - 2];
    }

    for (size_t i = 0; i < centers_to_search.size(); ++i) {
      const uint32_t token = centers_to_search[i].first;

      asymmetric_hashing_internal::LUT16ArgsTopN<float> args;
      args.packed_dataset = center_data[i].first;
      args.next_partition = center_data[i + 1].first;
      args.num_32dp_simd_iters = DivRoundUp(center_data[i].second, 32);
      args.num_blocks = num_blocks;
      args.lookups = lookups;
      args.fixed_point_multipliers = multipliers;
      args.biases = {&centers_to_search[i].second, 1};
      args.first_dp_index = token << global_topn_shift;
      args.num_datapoints = center_data[i].second;
      args.fast_topns = tops;
      args.prefetch_strategy =
          asymmetric_hashing_internal::PrefetchStrategy::kSmart;

      args.restrict_whitelists = allowlists;
      asymmetric_hashing_internal::LUT16Interface::GetTopFloatDistances(
          std::move(args));
    }

    AssignResults(&top_n, result);

    const uint32_t local_idx_mask = (1u << global_topn_shift) - 1;
    for (pair<DatapointIndex, float>& idx_dis : *result) {
      uint32_t partition_idx = idx_dis.first >> global_topn_shift;
      uint32_t local_idx = idx_dis.first & local_idx_mask;
      idx_dis.first = datapoints_by_token_[partition_idx][local_idx];
    }
    if (!datapoints_by_token_disjoint_) {
      DeduplicateDatabaseSpilledResults(result,
                                        params.pre_reordering_num_neighbors());
    }
    return OkStatus();
  } else {
    FastTopNeighbors<float> top_n(NumNeighborsWithSpillingMultiplier(
                                      params.pre_reordering_num_neighbors()),
                                  params.pre_reordering_epsilon());
    SCANN_RETURN_IF_ERROR(FindNeighborsInternal2(
        maybe_projected_query, params, centers_to_search, std::move(top_n),
        result, std::move(lookup_table)));
    if (!datapoints_by_token_disjoint_) {
      DeduplicateDatabaseSpilledResults(result,
                                        params.pre_reordering_num_neighbors());
    }
    return OkStatus();
  }
}

template <typename TopN>
Status TreeAHHybridResidual::FindNeighborsInternal2(
    const DatapointPtr<float>& maybe_projected_query,
    const SearchParameters& params,
    ConstSpan<pair<DatapointIndex, float>> centers_to_search, TopN top_n,
    NNResultsVector* result,
    shared_ptr<AsymmetricHashingOptionalParameters> lookup_table) const {
  DCHECK(result);
  DCHECK(!params.pre_reordering_crowding_enabled() ||
         datapoints_by_token_disjoint_);
  SearchParameters leaf_params;
  leaf_params.set_pre_reordering_num_neighbors(
      NumNeighborsWithSpillingMultiplier(
          params.pre_reordering_num_neighbors()));
  if (params.pre_reordering_crowding_enabled()) {
    SCANN_RET_CHECK_EQ(params.pre_reordering_num_neighbors(),
                       leaf_params.pre_reordering_num_neighbors());
    leaf_params.set_per_crowding_attribute_pre_reordering_num_neighbors(
        params.per_crowding_attribute_pre_reordering_num_neighbors());
  }
  leaf_params.set_searcher_specific_optional_parameters(lookup_table);

  typename TopN::Mutator mutator;
  top_n.AcquireMutator(&mutator);
  for (size_t center_idx : IndicesOf(centers_to_search)) {
    const int32_t token = centers_to_search[center_idx].first;
    const float distance_to_center = centers_to_search[center_idx].second;
    NNResultsVector leaf_results;
    leaf_params.set_pre_reordering_epsilon(mutator.epsilon() -
                                           distance_to_center);
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
            maybe_projected_query, leaf_params, &leaf_results));

    if (!is_streaming_result_) {
      tree_x_internal::AddLeafResultsToTopN(datapoints_by_token_[token],
                                            distance_to_center, leaf_results,
                                            &mutator);
    }
  }
  mutator.Release();

  if (!is_streaming_result_) {
    AssignResults(&top_n, result);
  }
  return OkStatus();
}

StatusOr<typename SingleMachineSearcherBase<float>::Mutator*>
TreeAHHybridResidual::GetMutator() const {
  if (!mutator_) {
    SCANN_RET_CHECK(!this->hashed_dataset())
        << "Must release hashed dataset before calling "
           "TreeAHHybridResidual::GetMutator since the hashed dataset is not "
           "used once the tree-X hybrid is built and can't be easily updated.";
    auto mutable_this = const_cast<TreeAHHybridResidual*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_,
        TreeXHybridMutator<TreeAHHybridResidual>::Create(mutable_this));
  }
  return static_cast<typename SingleMachineSearcherBase<float>::Mutator*>(
      mutator_.get());
}

StatusOr<TreeAHHybridResidual::MutationArtifacts>
TreeAHHybridResidual::TokenizeAndMaybeResidualize(
    const DatapointPtr<float>& dptr) {
  vector<pair<DatapointIndex, float>> token_storage;
  SCANN_RET_CHECK(database_tokenizer_);
  SCANN_RETURN_IF_ERROR(database_tokenizer_->TokensForDatapointWithSpilling(
      dptr, 2, &token_storage));
  MutationArtifacts result;
  result.tokens.reserve(token_storage.size());
  result.residual_storage[0] = nullptr;
  result.residual_dimensionality[0] = 0;
  for (auto [token_idx, token] : Enumerate(token_storage)) {
    SCANN_ASSIGN_OR_RETURN(
        Datapoint<float> residual,
        database_tokenizer_->ResidualizeToFloat(dptr, token.first));
    if (result.residual_storage[0] == nullptr) {
      result.residual_storage[0] = make_unique<float[]>(
          residual.dimensionality() * token_storage.size());
      result.residual_dimensionality[0] = residual.dimensionality();
    }
    MutableSpan<float> residual_span(result.residual_storage[0].get() +
                                         token_idx * residual.dimensionality(),
                                     residual.dimensionality());
    std::copy(residual.values().begin(), residual.values().end(),
              residual_span.begin());
    result.tokens.push_back(token.first);
  }
  return result;
}

StatusOr<vector<TreeAHHybridResidual::MutationArtifacts>>
TreeAHHybridResidual::TokenizeAndMaybeResidualize(
    const TypedDataset<float>& dps) {
  vector<vector<pair<DatapointIndex, float>>> token_storage(dps.size());
  SCANN_RET_CHECK(database_tokenizer_);
  SCANN_RETURN_IF_ERROR(
      database_tokenizer_->TokensForDatapointWithSpillingBatched(
          dps, {}, MakeMutableSpan(token_storage)));
  vector<TreeAHHybridResidual::MutationArtifacts> results(dps.size());
  for (size_t dp_idx : IndicesOf(token_storage)) {
    DatapointPtr<float> dptr = dps[dp_idx];
    auto tokens = MakeConstSpan(token_storage[dp_idx]);
    const size_t num_tokens = tokens.size();
    auto& result = results[dp_idx];
    result.residual_storage[0] = nullptr;
    result.residual_dimensionality[0] = 0;
    for (auto [token_idx, token_and_dist] : Enumerate(tokens)) {
      SCANN_ASSIGN_OR_RETURN(
          Datapoint<float> residual,
          database_tokenizer_->ResidualizeToFloat(dptr, token_and_dist.first));
      if (result.residual_storage[0] == nullptr) {
        result.residual_storage[0] =
            make_unique<float[]>(residual.dimensionality() * num_tokens);
        result.residual_dimensionality[0] = residual.dimensionality();
      }
      MutableSpan<float> residual_span(
          result.residual_storage[0].get() +
              token_idx * residual.dimensionality(),
          residual.dimensionality());
      std::copy(residual.values().begin(), residual.values().end(),
                residual_span.begin());
      result.tokens.push_back(token_and_dist.first);
    }
  }
  return results;
}

StatusOr<SingleMachineFactoryOptions>
TreeAHHybridResidual::ExtractSingleMachineFactoryOptions() {
  SCANN_ASSIGN_OR_RETURN(const int dataset_size,
                         UntypedSingleMachineSearcherBase::DatasetSize());
  SCANN_ASSIGN_OR_RETURN(
      SingleMachineFactoryOptions leaf_opts,
      MergeAHLeafOptions(leaf_searchers_, datapoints_by_token_, dataset_size,
                         datapoints_by_token_disjoint_
                             ? 1.0f
                             : spilling_overretrieve_factor_));
  SCANN_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<float>::ExtractSingleMachineFactoryOptions());
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          datapoints_by_token_);
  opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
  query_tokenizer_->CopyToProto(opts.serialized_partitioner.get());

  if (leaf_opts.ah_codebook != nullptr) {
    opts.ah_codebook = leaf_opts.ah_codebook;
    opts.hashed_dataset = leaf_opts.hashed_dataset;
    opts.soar_hashed_dataset = leaf_opts.soar_hashed_dataset;
  }
  return opts;
}

Status TreeAHHybridResidual::InitializeHealthStats() {
  return stats_collector_.Initialize(*this);
}

StatusOr<TreeAHHybridResidual::HealthStats>
TreeAHHybridResidual::GetHealthStats() const {
  return stats_collector_.GetHealthStats();
}

vector<uint32_t> TreeAHHybridResidual::SizeByPartition() const {
  return ::research_scann::SizeByPartition(datapoints_by_token_);
}

}  // namespace research_scann
