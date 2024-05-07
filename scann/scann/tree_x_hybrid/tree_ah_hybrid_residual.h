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



#ifndef SCANN_TREE_X_HYBRID_TREE_AH_HYBRID_RESIDUAL_H_
#define SCANN_TREE_X_HYBRID_TREE_AH_HYBRID_RESIDUAL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/proto/hash.pb.h"
#include "scann/tree_x_hybrid/internal/utils.h"
#include "scann/tree_x_hybrid/mutator.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/types.h"

namespace research_scann {

class TreeAHHybridResidual final : public SingleMachineSearcherBase<float> {
 public:
  TreeAHHybridResidual(shared_ptr<const DenseDataset<float>> dataset,
                       int32_t default_pre_reordering_num_neighbors,
                       float default_pre_reordering_epsilon)
      : SingleMachineSearcherBase<float>(std::move(dataset),
                                         default_pre_reordering_num_neighbors,
                                         default_pre_reordering_epsilon) {}

  struct BuildLeafSearchersOptions {
    const DenseDataset<uint8_t>* hashed_dataset = nullptr;

    const DenseDataset<uint8_t>* soar_hashed_dataset = nullptr;

    ThreadPool* pool = nullptr;
  };

  Status BuildLeafSearchers(
      const AsymmetricHasherConfig& config,
      unique_ptr<KMeansTreeLikePartitioner<float>> partitioner,
      shared_ptr<const asymmetric_hashing2::Model<float>> ah_model,
      vector<std::vector<DatapointIndex>> datapoints_by_token,
      BuildLeafSearchersOptions opts);

  void set_database_tokenizer(
      shared_ptr<const KMeansTreeLikePartitioner<float>> database_tokenizer) {
    database_tokenizer_ = database_tokenizer;
  }

  bool supports_crowding() const final { return true; }

  StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
      const final;

  static StatusOr<DenseDataset<float>> ComputeResiduals(
      const DenseDataset<float>& dataset,
      const KMeansTreeLikePartitioner<float>* partitioner,
      ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

  static StatusOr<DenseDataset<float>> ComputeResiduals(
      const DenseDataset<float>& dataset,
      const DenseDataset<float>& kmeans_centers,
      ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

  Status PreprocessQueryIntoParamsUnlocked(
      const DatapointPtr<float>& query,
      SearchParameters& search_params) const final;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

  vector<uint32_t> SizeByPartition() const final;

  float spilling_overretrieve_factor() const {
    return spilling_overretrieve_factor_;
  }
  void set_spilling_overretrieve_factor(float factor) {
    DCHECK_GE(factor, 1.0);
    DCHECK_LE(factor, 2.0);
    spilling_overretrieve_factor_ = factor;
  }

  void set_fixed_point_lut_conversion_options(
      AsymmetricHasherConfig::FixedPointLUTConversionOptions opts) {
    fixed_point_lut_conversion_options_ = std::move(opts);
  }

 protected:
  bool impl_needs_dataset() const final { return leaf_searchers_.empty(); }

  bool impl_needs_hashed_dataset() const final {
    return leaf_searchers_.empty();
  }

  Status FindNeighborsImpl(const DatapointPtr<float>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<float>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute) final;
  void DisableCrowdingImpl() final;

 private:
  class UnlockedTreeAHHybridResidualPreprocessingResults
      : public SearchParameters::UnlockedQueryPreprocessingResults {
   public:
    UnlockedTreeAHHybridResidualPreprocessingResults(
        vector<pair<DatapointIndex, float>> centers_to_search,
        asymmetric_hashing2::LookupTable lookup_table)
        : centers_to_search_(std::move(centers_to_search)),
          lookup_table_(
              make_unique<
                  asymmetric_hashing2::AsymmetricHashingOptionalParameters>(
                  std::move(lookup_table))) {}

    ConstSpan<pair<DatapointIndex, float>> centers_to_search() const {
      return centers_to_search_;
    }

    shared_ptr<asymmetric_hashing2::AsymmetricHashingOptionalParameters>
    lookup_table() const {
      return lookup_table_;
    }

   private:
    vector<pair<DatapointIndex, float>> centers_to_search_;
    shared_ptr<asymmetric_hashing2::AsymmetricHashingOptionalParameters>
        lookup_table_;
  };

  Status FindNeighborsInternal1(
      const DatapointPtr<float>& query, const SearchParameters& params,
      ConstSpan<pair<DatapointIndex, float>> centers_to_search,
      NNResultsVector* result) const;

  template <typename TopN>
  Status FindNeighborsInternal2(
      const DatapointPtr<float>& query, const SearchParameters& params,
      ConstSpan<pair<DatapointIndex, float>> centers_to_search, TopN top_n,
      NNResultsVector* result,
      shared_ptr<asymmetric_hashing2::AsymmetricHashingOptionalParameters>
          lookup_table) const;

  SCANN_INLINE uint8_t GlobalTopNShift() const {
    if (!enable_global_topn_) return 0;

    int inner_leaf_bits = 32 - bits::Log2Ceiling(datapoints_by_token_.size());
    if (leaf_size_upper_bound_ <= (1ull << inner_leaf_bits)) {
      SCANN_LOG_NOOP(3) << "Global top-N enabled for query.";
      return inner_leaf_bits;
    } else {
      SCANN_LOG_NOOP(3) << "Leaf too big, global top-N disabled for query.";
      return 0;
    }
  }

  Status CheckBuildLeafSearchersPreconditions(
      const AsymmetricHasherConfig& config,
      const KMeansTreeLikePartitioner<float>& partitioner) const;

  using MutationArtifacts = TreeXHybridMutator<
      TreeAHHybridResidual>::TreeXPrecomputedMutationArtifacts;

  StatusOr<MutationArtifacts> TokenizeAndMaybeResidualize(
      const DatapointPtr<float>& dptr);

  StatusOr<vector<MutationArtifacts>> TokenizeAndMaybeResidualize(
      const TypedDataset<float>& dps);

  int32_t NumNeighborsWithSpillingMultiplier(int32_t num_neighbors) const {
    return datapoints_by_token_disjoint_
               ? num_neighbors
               : SafeIntFloatMul(num_neighbors, spilling_overretrieve_factor_);
  }

  Status AddLeafSearcher();

  vector<unique_ptr<asymmetric_hashing2::Searcher<float>>> leaf_searchers_;

  shared_ptr<const asymmetric_hashing2::AsymmetricQueryer<float>>
      asymmetric_queryer_;

  shared_ptr<KMeansTreeLikePartitioner<float>> query_tokenizer_;
  shared_ptr<const KMeansTreeLikePartitioner<float>> database_tokenizer_;

  vector<std::vector<DatapointIndex>> datapoints_by_token_;

  vector<uint32_t> leaf_tokens_by_norm_;

  DatapointIndex num_datapoints_ = 0;

  AsymmetricHasherConfig::LookupType lookup_type_tag_ =
      AsymmetricHasherConfig::FLOAT;

  AsymmetricHasherConfig::FixedPointLUTConversionOptions
      fixed_point_lut_conversion_options_;

  mutable unique_ptr<TreeXHybridMutator<TreeAHHybridResidual>> mutator_;
  friend class TreeXHybridMutator<TreeAHHybridResidual>;

  unique_ptr<asymmetric_hashing2::SearcherOptions<float>> searcher_options_ =
      nullptr;

  uint32_t leaf_size_upper_bound_ = 0;

  bool datapoints_by_token_disjoint_ = true;

  bool enable_global_topn_ = false;

  float spilling_overretrieve_factor_ = 2.0f;

  FRIEND_TEST(TreeAHHybridResidualTest, CrowdingMutation);

  FRIEND_TEST(TreeAHHybridResidualTest, GlobalTopNManyUpdates);
};

}  // namespace research_scann

#endif
