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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_SEARCHER_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_SEARCHER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/proto/hash.pb.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

class TreeAHHybridResidual;

namespace asymmetric_hashing2 {

template <typename T>
class SearcherOptions {
 public:
  explicit SearcherOptions(std::shared_ptr<const AsymmetricQueryer<T>> queryer,
                           std::shared_ptr<const Indexer<T>> indexer = nullptr)
      : indexer_(std::move(indexer)), asymmetric_queryer_(std::move(queryer)) {}

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

  std::shared_ptr<const AsymmetricQueryer<T>> asymmetric_queryer() const {
    return asymmetric_queryer_;
  }

  AsymmetricHasherConfig::LookupType asymmetric_lookup_type() const {
    return asymmetric_lookup_type_;
  }

  FixedPointLUTConversionOptions fixed_point_lut_conversion_options() const {
    return fixed_point_lut_conversion_options_;
  }

 private:
  std::shared_ptr<const Indexer<T>> indexer_ = nullptr;

  double noise_shaping_threshold_ = NAN;

  FixedPointLUTConversionOptions fixed_point_lut_conversion_options_;

  std::shared_ptr<const AsymmetricQueryer<T>> asymmetric_queryer_ = nullptr;

  AsymmetricHasherConfig::LookupType asymmetric_lookup_type_ =
      AsymmetricHasherConfig::FLOAT;

  template <typename U>
  friend class SearcherBase;
  template <typename U>
  friend class Searcher;
};

inline constexpr int kNumClustersPerBlockForLUT16 = 16;

template <typename T>
class SearcherBase : public SingleMachineSearcherBase<T> {
 public:
  explicit SearcherBase(
      absl_nullable std::shared_ptr<TypedDataset<T>> dataset,
      absl_nonnull std::shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
      SearcherOptions<T> opts, int32_t default_pre_reordering_num_neighbors,
      float default_pre_reordering_epsilon);

  explicit SearcherBase(SearcherOptions<T> opts,
                        int32_t default_pre_reordering_num_neighbors,
                        float default_pre_reordering_epsilon);

  ~SearcherBase() override = default;

  DatapointIndex optimal_batch_size() const final {
    return optimal_low_level_batch_size_;
  }

  const PackedDataset& packed_dataset() { return packed_dataset_; }

  bool impl_needs_dataset() const final { return false; }

  bool impl_needs_hashed_dataset() const final {
    return !(lut16_ && opts_.asymmetric_queryer_->num_clusters_per_block() ==
                           kNumClustersPerBlockForLUT16);
  }

 protected:
  virtual Status FindNeighborsBatchedInternal(
      std::function<DatapointPtr<T>(DatapointIndex)> get_query,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const = 0;

  StatusOr<const LookupTable*> GetOrCreateLookupTable(
      const DatapointPtr<T>& query, const SearchParameters& params,
      LookupTable* created_lookup_table_storage) const;

  SearcherOptions<T> opts_;

  PackedDataset packed_dataset_;

  const bool limited_inner_product_ : 1;

  const bool lut16_ : 1;

  uint8_t max_low_level_batch_size_ = 9;

  uint8_t optimal_low_level_batch_size_ = 1;

  std::vector<float> norm_inv_or_bias_ = {};

  friend class ::research_scann::TreeAHHybridResidual;
};

template <typename T>
class Searcher final : public SearcherBase<T> {
 public:
  Searcher(std::shared_ptr<TypedDataset<T>> dataset,
           std::shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
           SearcherOptions<T> opts,
           int32_t default_pre_reordering_num_neighbors,
           float default_pre_reordering_epsilon);

  ~Searcher() override;

  SCANN_DECLARE_IMMOBILE_CLASS(Searcher);

  bool supports_crowding() const final { return true; }

  using PrecomputedMutationArtifacts =
      UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
  using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;

  class Mutator : public SingleMachineSearcherBase<T>::Mutator {
   public:
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    static StatusOr<std::unique_ptr<typename Searcher<T>::Mutator>> Create(
        Searcher<T>* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final {}

    using SingleMachineSearcherBase<
        T>::Mutator::ComputePrecomputedMutationArtifacts;
    std::unique_ptr<PrecomputedMutationArtifacts>
    ComputePrecomputedMutationArtifacts(
        const DatapointPtr<T>& dptr) const final;
    StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex i) const final;
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                          string_view docid,
                                          const MutationOptions& mo) final;
    Status RemoveDatapoint(string_view docid) final;
    void Reserve(size_t size) final;
    Status RemoveDatapoint(DatapointIndex index) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             string_view docid,
                                             const MutationOptions& mo) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions& mo) final;

    std::unique_ptr<PrecomputedMutationArtifacts>
    ComputePrecomputedMutationArtifacts(const DatapointPtr<T>& maybe_residual,
                                        const DatapointPtr<T>& original) const;

   private:
    Mutator(Searcher<T>* searcher, const Indexer<T>* indexer,
            PackedDataset* packed_dataset)
        : searcher_(searcher),
          indexer_(indexer),
          packed_dataset_(packed_dataset) {}

    Status Hash(const DatapointPtr<T>& maybe_residual,
                const DatapointPtr<T>& original,
                Datapoint<uint8_t>* result) const;

    Datapoint<uint8_t> EnsureDatapointUnpacked(const Datapoint<uint8_t>& dp);

    Searcher<T>* searcher_ = nullptr;
    const Indexer<T>* indexer_ = nullptr;
    PackedDataset* packed_dataset_ = nullptr;
  };

  StatusOr<typename SingleMachineSearcherBase<T>::Mutator*> GetMutator()
      const final;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

 protected:
  explicit Searcher(SearcherOptions<T> opts,
                    int32_t default_pre_reordering_num_neighbors,
                    float default_pre_reordering_epsilon);

  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const override;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const override;

  Status VerifyLimitedInnerProductNormsSize() const;

  template <typename PostprocessFunctor>
  QueryerOptions<PostprocessFunctor> GetQueryerOptions(
      PostprocessFunctor postprocessing_functor) const;

  template <typename PostprocessFunctor>
  Status FindNeighborsTopNDispatcher(const DatapointPtr<T>& query,
                                     const SearchParameters& params,
                                     PostprocessFunctor postprocessing_functor,
                                     NNResultsVector* result) const;

  Status FindNeighborsBatchedInternal(
      std::function<DatapointPtr<T>(DatapointIndex)> get_query,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const override;

  template <size_t kNumQueries, typename PostprocessFunctor>
  Status FindOneLowLevelBatchOfNeighbors(
      size_t low_level_batch_start,
      std::function<DatapointPtr<T>(DatapointIndex)> get_query,
      ConstSpan<SearchParameters> params,
      const QueryerOptions<PostprocessFunctor>& queryer_options,
      MutableSpan<NNResultsVector> results) const;

  mutable std::unique_ptr<typename Searcher<T>::Mutator> mutator_ = nullptr;

  friend class ::research_scann::TreeAHHybridResidual;
};

class AsymmetricHashingOptionalParameters
    : public SearcherSpecificOptionalParameters {
 public:
  explicit AsymmetricHashingOptionalParameters(
      LookupTable precomputed_lookup_table)
      : precomputed_lookup_table_(std::move(precomputed_lookup_table)) {}

  const LookupTable& precomputed_lookup_table() const {
    return precomputed_lookup_table_;
  }

 private:
  LookupTable precomputed_lookup_table_;

  template <typename U>
  friend class SearcherBase;
  template <typename U>
  friend class Searcher;
};

template <typename T>
class PrecomputedAsymmetricLookupTableCreator final
    : public LeafSearcherOptionalParameterCreator<T> {
 public:
  explicit PrecomputedAsymmetricLookupTableCreator(
      std::shared_ptr<const AsymmetricQueryer<T>> queryer,
      AsymmetricHasherConfig::LookupType lookup_type,
      AsymmetricHasherConfig::FixedPointLUTConversionOptions
          fixed_point_lut_conversion_options =
              AsymmetricHasherConfig::FixedPointLUTConversionOptions())
      : queryer_(std::move(queryer)),
        lookup_type_(lookup_type),
        fixed_point_lut_conversion_options_(
            fixed_point_lut_conversion_options) {}

  StatusOr<std::unique_ptr<SearcherSpecificOptionalParameters>>
  CreateLeafSearcherOptionalParameters(
      const DatapointPtr<T>& query) const final;

 private:
  std::shared_ptr<const AsymmetricQueryer<T>> queryer_;
  AsymmetricHasherConfig::LookupType lookup_type_;
  AsymmetricHasherConfig::FixedPointLUTConversionOptions
      fixed_point_lut_conversion_options_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, SearcherOptions);
SCANN_INSTANTIATE_TYPED_CLASS(extern, SearcherBase);
SCANN_INSTANTIATE_TYPED_CLASS(extern, Searcher);
SCANN_INSTANTIATE_TYPED_CLASS(extern, PrecomputedAsymmetricLookupTableCreator);

}  // namespace asymmetric_hashing2
}  // namespace research_scann

#endif
