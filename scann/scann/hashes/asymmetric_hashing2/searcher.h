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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_SEARCHER_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_SEARCHER_H_

#include <cstdint>
#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/proto/hash.pb.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/platform/macros.h"

namespace research_scann {

class TreeAHHybridResidual;

namespace asymmetric_hashing2 {

template <typename T>
class SearcherOptions {
 public:
  explicit SearcherOptions(shared_ptr<const AsymmetricQueryer<T>> queryer,
                           shared_ptr<const Indexer<T>> indexer = nullptr)
      : asymmetric_queryer_(std::move(queryer)), indexer_(std::move(indexer)) {}

  void set_asymmetric_lookup_type(
      AsymmetricHasherConfig::LookupType lookup_type) {
    asymmetric_lookup_type_ = lookup_type;
  }

  AsymmetricHasherConfig::QuantizationScheme quantization_scheme() const {
    return asymmetric_queryer_ ? asymmetric_queryer_->quantization_scheme()
                               : AsymmetricHasherConfig::PRODUCT;
  }

  size_t num_blocks() const {
    if (asymmetric_queryer_) {
      return asymmetric_queryer_->num_blocks();
    } else {
      return 0;
    }
  }

  using FixedPointLUTConversionOptions =
      AsymmetricHasherConfig::FixedPointLUTConversionOptions;

  void set_fixed_point_lut_conversion_options(
      FixedPointLUTConversionOptions x) {
    fixed_point_lut_conversion_options_ = x;
  }

  void set_noise_shaping_threshold(double t) { noise_shaping_threshold_ = t; }

 private:
  shared_ptr<const AsymmetricQueryer<T>> asymmetric_queryer_ = nullptr;

  shared_ptr<const Indexer<T>> indexer_ = nullptr;

  AsymmetricHasherConfig::LookupType asymmetric_lookup_type_ =
      AsymmetricHasherConfig::FLOAT;

  FixedPointLUTConversionOptions fixed_point_lut_conversion_options_;

  double noise_shaping_threshold_ = NAN;

  template <typename U>
  friend class Searcher;
};

template <typename T>
class Searcher final : public SingleMachineSearcherBase<T> {
 public:
  Searcher(shared_ptr<TypedDataset<T>> dataset,
           shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
           SearcherOptions<T> opts,
           int32_t default_pre_reordering_num_neighbors,
           float default_pre_reordering_epsilon);

  ~Searcher() final;

  Searcher(Searcher&& rhs) = default;
  Searcher& operator=(Searcher&& rhs) = default;

  bool supports_crowding() const final { return true; }

  DatapointIndex optimal_batch_size() const final {
    return optimal_low_level_batch_size_;
  }

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

 protected:
  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

 private:
  bool impl_needs_dataset() const final { return false; }

  bool impl_needs_hashed_dataset() const final {
    return !(RuntimeSupportsSse4() && lut16_ &&
             opts_.asymmetric_queryer_->num_clusters_per_block() == 16);
  }

  StatusOr<const LookupTable*> GetOrCreateLookupTable(
      const DatapointPtr<T>& query, const SearchParameters& params,
      LookupTable* created_lookup_table_storage) const;

  template <typename PostprocessFunctor>
  QueryerOptions<PostprocessFunctor> GetQueryerOptions(
      PostprocessFunctor postprocessing_functor) const;

  template <typename PostprocessFunctor>
  Status FindNeighborsTopNDispatcher(const DatapointPtr<T>& query,
                                     const SearchParameters& params,
                                     PostprocessFunctor postprocessing_functor,
                                     NNResultsVector* result) const;

  template <typename PostprocessFunctor>
  Status FindNeighborsBatchedInternal(
      std::function<DatapointPtr<T>(DatapointIndex)> get_query,
      ConstSpan<SearchParameters> params,
      PostprocessFunctor postprocessing_functor,
      MutableSpan<NNResultsVector> results) const;

  template <size_t kNumQueries, typename PostprocessFunctor>
  Status FindOneLowLevelBatchOfNeighbors(
      size_t low_level_batch_start,
      std::function<DatapointPtr<T>(DatapointIndex)> get_query,
      ConstSpan<SearchParameters> params,
      const QueryerOptions<PostprocessFunctor>& queryer_options,
      MutableSpan<NNResultsVector> results) const;

  SearcherOptions<T> opts_;

  PackedDataset packed_dataset_;

  vector<float> norm_inv_ = {};

  const bool limited_inner_product_;

  vector<float> bias_ = {};

  const bool lut16_;

  size_t max_low_level_batch_size_ = 9;

  size_t optimal_low_level_batch_size_ = 1;

  friend class ::research_scann::TreeAHHybridResidual;

  TF_DISALLOW_COPY_AND_ASSIGN(Searcher);
};

class AsymmetricHashingOptionalParameters
    : public SearcherSpecificOptionalParameters {
 public:
  explicit AsymmetricHashingOptionalParameters(
      LookupTable precomputed_lookup_table)
      : precomputed_lookup_table_(std::move(precomputed_lookup_table)) {}

  void SetIndexAndBias(DatapointIndex index, float bias) {
    starting_dp_idx_ = index;
    lut16_bias_ = bias;
  }

  void SetFastTopNeighbors(FastTopNeighbors<float>* top_n) { top_n_ = top_n; }

  const FastTopNeighbors<float>* top_n() const { return top_n_; }

 private:
  LookupTable precomputed_lookup_table_;

  FastTopNeighbors<float>* top_n_ = nullptr;

  DatapointIndex starting_dp_idx_ = 0;
  float lut16_bias_ = 0;

  template <typename U>
  friend class Searcher;
};

template <typename T>
class PrecomputedAsymmetricLookupTableCreator final
    : public LeafSearcherOptionalParameterCreator<T> {
 public:
  PrecomputedAsymmetricLookupTableCreator(
      shared_ptr<const AsymmetricQueryer<T>> queryer,
      AsymmetricHasherConfig::LookupType lookup_type)
      : queryer_(std::move(queryer)), lookup_type_(lookup_type) {}

  StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
  CreateLeafSearcherOptionalParameters(
      const DatapointPtr<T>& query) const final;

 private:
  shared_ptr<const AsymmetricQueryer<T>> queryer_;
  AsymmetricHasherConfig::LookupType lookup_type_;
};

extern template Status Searcher<float>::FindNeighborsBatchedInternal<
    asymmetric_hashing_internal::IdentityPostprocessFunctor>(
    std::function<DatapointPtr<float>(DatapointIndex)> get_query,
    ConstSpan<SearchParameters> params,
    asymmetric_hashing_internal::IdentityPostprocessFunctor
        postprocessing_functor,
    MutableSpan<NNResultsVector> results) const;

SCANN_INSTANTIATE_TYPED_CLASS(extern, SearcherOptions);
SCANN_INSTANTIATE_TYPED_CLASS(extern, Searcher);
SCANN_INSTANTIATE_TYPED_CLASS(extern, PrecomputedAsymmetricLookupTableCreator);

}  // namespace asymmetric_hashing2
}  // namespace research_scann

#endif
