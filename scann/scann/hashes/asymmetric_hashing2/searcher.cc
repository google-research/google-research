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



#include "scann/hashes/asymmetric_hashing2/searcher.h"

#include <math.h>

#include <memory>
#include <typeinfo>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/serialization.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/oss_wrappers/scann_serialize.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace research_scann {
namespace asymmetric_hashing2 {
namespace {

shared_ptr<DenseDataset<uint8_t>> PreprocessHashedDataset(
    shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
    const AsymmetricHasherConfig::QuantizationScheme quantization_scheme,
    const size_t num_blocks) {
  if (quantization_scheme == AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    auto dataset_no_bias = std::make_shared<DenseDataset<uint8_t>>();

    if (hashed_dataset->empty()) {
      return dataset_no_bias;
    }
    dataset_no_bias->set_dimensionality((*hashed_dataset)[0].nonzero_entries() -
                                        sizeof(float));
    dataset_no_bias->Reserve(hashed_dataset->size());
    for (const auto& dp : *hashed_dataset) {
      auto dptr =
          MakeDatapointPtr(dp.values(), dp.nonzero_entries() - sizeof(float));
      TF_CHECK_OK(dataset_no_bias->Append(dptr, ""));
    }
    return dataset_no_bias;
  } else if (quantization_scheme == AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    auto dataset_unpacked = std::make_shared<DenseDataset<uint8_t>>();
    dataset_unpacked->set_dimensionality(num_blocks);
    dataset_unpacked->Reserve(hashed_dataset->size());
    Datapoint<uint8_t> unpacked_dp;
    for (const auto& dptr : *hashed_dataset) {
      UnpackNibblesDatapoint(dptr, &unpacked_dp);
      TF_CHECK_OK(dataset_unpacked->Append(unpacked_dp.ToPtr(), ""));
    }
    return dataset_unpacked;
  }
  return hashed_dataset;
}
}  // namespace

template <typename T>
Searcher<T>::Searcher(shared_ptr<TypedDataset<T>> dataset,
                      shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
                      SearcherOptions<T> opts,
                      int32_t default_pre_reordering_num_neighbors,
                      float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase<T>(
          dataset,
          PreprocessHashedDataset(hashed_dataset, opts.quantization_scheme(),
                                  opts.num_blocks()),
          default_pre_reordering_num_neighbors, default_pre_reordering_epsilon),
      opts_(std::move(opts)),
      limited_inner_product_(
          (opts_.asymmetric_queryer_ &&
           typeid(*opts_.asymmetric_queryer_->lookup_distance()) ==
               typeid(const LimitedInnerProductDistance))),
      lut16_(opts_.asymmetric_lookup_type_ ==
                 AsymmetricHasherConfig::INT8_LUT16 &&
             opts_.asymmetric_queryer_) {
  DCHECK(hashed_dataset);

  if (lut16_) {
    packed_dataset_ =
        ::research_scann::asymmetric_hashing2::CreatePackedDataset(
            *this->hashed_dataset());

    const size_t l2_cache_bytes = 256 * 1024;
    if (packed_dataset_.bit_packed_data.size() <= l2_cache_bytes / 2) {
      optimal_low_level_batch_size_ = 3;
      max_low_level_batch_size_ = 3;
    } else {
      if (RuntimeSupportsAvx2()) {
        if (packed_dataset_.num_blocks <= 300) {
          optimal_low_level_batch_size_ = 7;
        } else {
          optimal_low_level_batch_size_ = 5;
        }
      } else {
        if (packed_dataset_.num_blocks <= 300) {
          optimal_low_level_batch_size_ = 6;
        } else {
          optimal_low_level_batch_size_ = 5;
        }
      }
    }
  }

  if (opts_.quantization_scheme() == AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    bias_.reserve(hashed_dataset->size());
    if (!hashed_dataset->empty()) {
      const int dim = hashed_dataset->at(0).nonzero_entries();
      for (int i = 0; i < hashed_dataset->size(); i++) {
        const float bias = strings::KeyToFloat(string_view(
            reinterpret_cast<const char*>((*hashed_dataset)[i].values() + dim -
                                          sizeof(float)),
            sizeof(float)));

        bias_.push_back(-bias);
      }
    }
  }

  if (limited_inner_product_) {
    CHECK(opts_.indexer_) << "Indexer must be non-null if "
                             "limited inner product searcher is being used.";
    for (DatapointIndex dp_idx : Seq(hashed_dataset->size())) {
      Datapoint<FloatingTypeFor<T>> dp;
      TF_CHECK_OK(opts_.indexer_->Reconstruct((*hashed_dataset)[dp_idx], &dp));
      double norm = SquaredL2Norm(dp.ToPtr());
      norm_inv_.push_back(static_cast<float>(norm == 0 ? 0 : 1 / sqrt(norm)));
    }
  }
}

template <typename T>
Searcher<T>::~Searcher() {}

template <typename T>
Status Searcher<T>::FindNeighborsImpl(const DatapointPtr<T>& query,
                                      const SearchParameters& params,
                                      NNResultsVector* result) const {
  if (limited_inner_product_) {
    if (opts_.symmetric_queryer_) {
      return FailedPreconditionError(
          "LimitedInnerProduct does not work with symmetric queryer.");
    }
    float query_norm = static_cast<float>(sqrt(SquaredL2Norm(query)));
    asymmetric_hashing_internal::LimitedInnerFunctor functor(query_norm,
                                                             norm_inv_);
    return FindNeighborsTopNDispatcher<
        asymmetric_hashing_internal::LimitedInnerFunctor>(query, params,
                                                          functor, result);
  } else if (opts_.quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    asymmetric_hashing_internal::AddBiasFunctor functor(
        bias_, query.values_slice().back());
    return FindNeighborsTopNDispatcher<
        asymmetric_hashing_internal::AddBiasFunctor>(query, params, functor,
                                                     result);
  } else {
    return FindNeighborsTopNDispatcher<
        asymmetric_hashing_internal::IdentityPostprocessFunctor>(
        query, params,
        asymmetric_hashing_internal::IdentityPostprocessFunctor(), result);
  }
}

template <typename T>
Status Searcher<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  bool crowding_enabled_for_any_query = false;
  for (const auto& p : params) {
    if (p.pre_reordering_crowding_enabled()) {
      crowding_enabled_for_any_query = true;
      break;
    }
  }
  if (opts_.symmetric_queryer_ || !lut16_ || limited_inner_product_ ||
      crowding_enabled_for_any_query ||
      opts_.quantization_scheme() == AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    return SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
        queries, params, results);
  }
  return FindNeighborsBatchedInternal<
      asymmetric_hashing_internal::IdentityPostprocessFunctor>(
      [&queries](DatapointIndex i) { return queries[i]; }, params,
      asymmetric_hashing_internal::IdentityPostprocessFunctor(), results);
}

template <typename T>
StatusOr<const LookupTable*> Searcher<T>::GetOrCreateLookupTable(
    const DatapointPtr<T>& query, const SearchParameters& params,
    LookupTable* created_lookup_table_storage) const {
  DCHECK(created_lookup_table_storage);
  auto per_query_opts =
      dynamic_cast<const AsymmetricHashingOptionalParameters*>(
          params.searcher_specific_optional_parameters());
  if (per_query_opts && !per_query_opts->precomputed_lookup_table_.empty()) {
    return &per_query_opts->precomputed_lookup_table_;
  } else {
    TF_ASSIGN_OR_RETURN(*created_lookup_table_storage,
                        opts_.asymmetric_queryer_->CreateLookupTable(
                            query, opts_.asymmetric_lookup_type_,
                            opts_.fixed_point_lut_conversion_options_));
    return created_lookup_table_storage;
  }
}

template <typename T>
template <typename PostprocessFunctor>
Status Searcher<T>::FindNeighborsTopNDispatcher(
    const DatapointPtr<T>& query, const SearchParameters& params,
    PostprocessFunctor postprocessing_functor, NNResultsVector* result) const {
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    auto ah_optional_params = params.searcher_specific_optional_parameters<
        AsymmetricHashingOptionalParameters>();
    if (ah_optional_params && ah_optional_params->top_n() &&
        !opts_.symmetric_queryer_) {
      auto queryer_opts = GetQueryerOptions(postprocessing_functor);
      queryer_opts.first_dp_index = ah_optional_params->starting_dp_idx_;
      queryer_opts.lut16_bias = ah_optional_params->lut16_bias_;
      LookupTable lookup_table_storage;
      TF_ASSIGN_OR_RETURN(
          const LookupTable* lookup_table,
          GetOrCreateLookupTable(query, params, &lookup_table_storage));
      SCANN_RETURN_IF_ERROR(AsymmetricQueryer<T>::FindApproximateNeighbors(
          *lookup_table, params, std::move(queryer_opts),
          ah_optional_params->top_n_));
    } else {
      TopNeighbors<float> top_n(params.pre_reordering_num_neighbors());
      SCANN_RETURN_IF_ERROR(FindNeighborsQueryerDispatcher(
          query, params, postprocessing_functor, &top_n));
      *result = top_n.TakeUnsorted();
    }
  }
  return OkStatus();
}

template <typename T>
template <typename PostprocessFunctor>
QueryerOptions<PostprocessFunctor> Searcher<T>::GetQueryerOptions(
    PostprocessFunctor postprocessing_functor) const {
  QueryerOptions<PostprocessFunctor> queryer_options;
  std::shared_ptr<DefaultDenseDatasetView<uint8_t>> hashed_dataset_view;

  if (this->hashed_dataset()) {
    hashed_dataset_view = std::make_shared<DefaultDenseDatasetView<uint8_t>>(
        *this->hashed_dataset());
  }
  queryer_options.hashed_dataset = hashed_dataset_view;
  queryer_options.postprocessing_functor = std::move(postprocessing_functor);
  if (lut16_) queryer_options.lut16_packed_dataset = &packed_dataset_;
  return queryer_options;
}

template <typename T>
template <typename PostprocessFunctor, typename TopN>
Status Searcher<T>::FindNeighborsQueryerDispatcher(
    const DatapointPtr<T>& query, const SearchParameters& params,
    PostprocessFunctor postprocessing_functor, TopN* result) const {
  auto queryer_options = GetQueryerOptions(postprocessing_functor);
  if (opts_.symmetric_queryer_) {
    auto view = queryer_options.hashed_dataset.get();
    Datapoint<uint8_t> hashed_query;
    SCANN_RETURN_IF_ERROR(opts_.indexer_->Hash(query, &hashed_query));
    SCANN_RETURN_IF_ERROR(opts_.symmetric_queryer_->FindApproximateNeighbors(
        hashed_query.ToPtr(), view, params, std::move(queryer_options),
        result));
  } else {
    LookupTable lookup_table_storage;
    TF_ASSIGN_OR_RETURN(
        const LookupTable* lookup_table,
        GetOrCreateLookupTable(query, params, &lookup_table_storage));
    SCANN_RETURN_IF_ERROR(AsymmetricQueryer<T>::FindApproximateNeighbors(
        *lookup_table, params, std::move(queryer_options), result));
  }
  return OkStatus();
}

template <typename T>
template <typename PostprocessFunctor>
Status Searcher<T>::FindNeighborsBatchedInternal(
    std::function<DatapointPtr<T>(DatapointIndex)> get_query,
    ConstSpan<SearchParameters> params,
    PostprocessFunctor postprocessing_functor,
    MutableSpan<NNResultsVector> results) const {
  using QueryerOptionsT = QueryerOptions<PostprocessFunctor>;
  QueryerOptionsT queryer_options;

  if (this->hashed_dataset()) {
    queryer_options.hashed_dataset =
        std::make_shared<DefaultDenseDatasetView<uint8_t>>(
            *this->hashed_dataset());
  }
  queryer_options.postprocessing_functor = std::move(postprocessing_functor);
  queryer_options.lut16_packed_dataset = &packed_dataset_;
  const size_t num_queries = params.size();
  size_t low_level_batch_start = 0;

  while (low_level_batch_start < num_queries) {
    const size_t low_level_batch_size = [&] {
      const size_t queries_left = num_queries - low_level_batch_start;

      if (queries_left <= max_low_level_batch_size_) return queries_left;

      if (queries_left >= 2 * max_low_level_batch_size_) {
        return optimal_low_level_batch_size_;
      }

      return queries_left / 2;
    }();
    switch (low_level_batch_size) {
      case 9:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<9>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 8:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<8>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 7:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<7>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 6:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<6>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 5:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<5>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 4:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<4>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 3:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<3>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 2:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<2>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      case 1:
        SCANN_RETURN_IF_ERROR(FindOneLowLevelBatchOfNeighbors<1>(
            low_level_batch_start, get_query, params, queryer_options,
            results));
        break;
      default:
        LOG(FATAL) << "Can't happen";
    }
    low_level_batch_start += low_level_batch_size;
  }
  return OkStatus();
}

template <typename T>
template <size_t kNumQueries, typename PostprocessFunctor>
Status Searcher<T>::FindOneLowLevelBatchOfNeighbors(
    size_t low_level_batch_start,
    std::function<DatapointPtr<T>(DatapointIndex)> get_query,
    ConstSpan<SearchParameters> params,
    const QueryerOptions<PostprocessFunctor>& queryer_options,
    MutableSpan<NNResultsVector> results) const {
  std::array<LookupTable, kNumQueries> lookup_storages;
  std::array<const LookupTable*, kNumQueries> lookup_ptrs;
  std::array<TopNeighbors<float>, kNumQueries> top_ns_storage;
  std::array<TopNeighbors<float>*, kNumQueries> top_ns;
  std::array<const SearchParameters*, kNumQueries> cur_batch_params;
  for (size_t batch_idx = 0; batch_idx < kNumQueries; ++batch_idx) {
    TF_ASSIGN_OR_RETURN(
        lookup_ptrs[batch_idx],
        GetOrCreateLookupTable(get_query(low_level_batch_start + batch_idx),
                               params[low_level_batch_start + batch_idx],
                               &lookup_storages[batch_idx]));
    top_ns_storage[batch_idx] =
        TopNeighbors<float>(params[low_level_batch_start + batch_idx]
                                .pre_reordering_num_neighbors());
    top_ns[batch_idx] = &top_ns_storage[batch_idx];
    cur_batch_params[batch_idx] = &params[low_level_batch_start + batch_idx];
  }
  SCANN_RETURN_IF_ERROR(AsymmetricQueryer<T>::FindApproximateNeighborsBatched(
      lookup_ptrs, cur_batch_params, queryer_options, top_ns));
  for (size_t batch_idx = 0; batch_idx < kNumQueries; ++batch_idx) {
    results[low_level_batch_start + batch_idx] =
        top_ns_storage[batch_idx].TakeUnsorted();
  }
  return OkStatus();
}

template <typename T>
StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
PrecomputedAsymmetricLookupTableCreator<T>::
    CreateLeafSearcherOptionalParameters(const DatapointPtr<T>& query) const {
  TF_ASSIGN_OR_RETURN(auto lookup_table,
                      queryer_->CreateLookupTable(query, lookup_type_));
  return unique_ptr<SearcherSpecificOptionalParameters>(
      new AsymmetricHashingOptionalParameters(std::move(lookup_table)));
}

template <typename T>
StatusOr<SingleMachineFactoryOptions>
Searcher<T>::ExtractSingleMachineFactoryOptions() {
  TF_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<T>::ExtractSingleMachineFactoryOptions());
  if (opts_.asymmetric_queryer_) {
    auto centers = opts_.asymmetric_queryer_->model()->centers();
    opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
    *opts.ah_codebook =
        DatasetSpanToCentersProto(centers, opts_.quantization_scheme());
    if (opts_.asymmetric_lookup_type_ == AsymmetricHasherConfig::INT8_LUT16)
      opts.hashed_dataset =
          make_shared<DenseDataset<uint8_t>>(UnpackDataset(packed_dataset_));
  }
  return opts;
}

template Status Searcher<float>::FindNeighborsBatchedInternal<
    asymmetric_hashing_internal::IdentityPostprocessFunctor>(
    std::function<DatapointPtr<float>(DatapointIndex)> get_query,
    ConstSpan<SearchParameters> params,
    asymmetric_hashing_internal::IdentityPostprocessFunctor
        postprocessing_functor,
    MutableSpan<NNResultsVector> results) const;

SCANN_INSTANTIATE_TYPED_CLASS(, SearcherOptions);
SCANN_INSTANTIATE_TYPED_CLASS(, Searcher);
SCANN_INSTANTIATE_TYPED_CLASS(, PrecomputedAsymmetricLookupTableCreator);

}  // namespace asymmetric_hashing2
}  // namespace research_scann
