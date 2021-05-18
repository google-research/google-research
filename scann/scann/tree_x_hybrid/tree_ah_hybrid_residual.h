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



#ifndef SCANN_TREE_X_HYBRID_TREE_AH_HYBRID_RESIDUAL_H_
#define SCANN_TREE_X_HYBRID_TREE_AH_HYBRID_RESIDUAL_H_

#include <cstdint>
#include <functional>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/proto/hash.pb.h"
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

  Status BuildLeafSearchers(
      const AsymmetricHasherConfig& config,
      unique_ptr<KMeansTreeLikePartitioner<float>> partitioner,
      shared_ptr<const asymmetric_hashing2::Model<float>> ah_model,
      vector<std::vector<DatapointIndex>> datapoints_by_token,
      const DenseDataset<uint8_t>* hashed_dataset, ThreadPool* pool = nullptr);

  void set_database_tokenizer(
      shared_ptr<const KMeansTreeLikePartitioner<float>> database_tokenizer) {
    database_tokenizer_ = database_tokenizer;
  }

  bool supports_crowding() const final { return true; }

  static StatusOr<DenseDataset<float>> ComputeResiduals(
      const DenseDataset<float>& dataset,
      const KMeansTreeLikePartitioner<float>* partitioner,
      ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
      bool normalize_residual_by_cluster_stdev = false);

  static StatusOr<DenseDataset<float>> ComputeResiduals(
      const DenseDataset<float>& dataset,
      const DenseDataset<float>& kmeans_centers,
      ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

  static StatusOr<uint8_t> ComputeGlobalTopNShift(
      ConstSpan<std::vector<DatapointIndex>> datapoints_by_token);

  Status PreprocessQueryIntoParamsUnlocked(
      const DatapointPtr<float>& query,
      SearchParameters& search_params) const final;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

  void AttemptEnableGlobalTopN();

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
        vector<KMeansTreeSearchResult> centers_to_search,
        asymmetric_hashing2::LookupTable lookup_table)
        : centers_to_search_(std::move(centers_to_search)),
          lookup_table_(
              make_unique<
                  asymmetric_hashing2::AsymmetricHashingOptionalParameters>(
                  std::move(lookup_table))) {}

    ConstSpan<KMeansTreeSearchResult> centers_to_search() const {
      return centers_to_search_;
    }

    shared_ptr<asymmetric_hashing2::AsymmetricHashingOptionalParameters>
    lookup_table() const {
      return lookup_table_;
    }

   private:
    vector<KMeansTreeSearchResult> centers_to_search_;
    shared_ptr<asymmetric_hashing2::AsymmetricHashingOptionalParameters>
        lookup_table_;
  };

  Status FindNeighborsInternal1(
      const DatapointPtr<float>& query, const SearchParameters& params,
      ConstSpan<KMeansTreeSearchResult> centers_to_search,
      NNResultsVector* result) const;

  template <typename TopN>
  Status FindNeighborsInternal2(
      const DatapointPtr<float>& query, const SearchParameters& params,
      ConstSpan<KMeansTreeSearchResult> centers_to_search, TopN top_n,
      NNResultsVector* result) const;

  Status CheckBuildLeafSearchersPreconditions(
      const AsymmetricHasherConfig& config,
      const KMeansTreeLikePartitioner<float>& partitioner) const;

  StatusOr<pair<int32_t, DatapointPtr<float>>> TokenizeAndMaybeResidualize(
      const DatapointPtr<float>& dptr, Datapoint<float>* residual_storage);

  StatusOr<vector<pair<int32_t, DatapointPtr<float>>>>
  TokenizeAndMaybeResidualize(const TypedDataset<float>& dps,
                              MutableSpan<Datapoint<float>*> residual_storage);

  vector<unique_ptr<asymmetric_hashing2::Searcher<float>>> leaf_searchers_;

  shared_ptr<const asymmetric_hashing2::AsymmetricQueryer<float>>
      asymmetric_queryer_;

  unique_ptr<KMeansTreeLikePartitioner<float>> query_tokenizer_;
  shared_ptr<const KMeansTreeLikePartitioner<float>> database_tokenizer_;

  vector<std::vector<DatapointIndex>> datapoints_by_token_;

  DatapointIndex num_datapoints_ = 0;

  vector<uint32_t> leaf_tokens_by_norm_;

  AsymmetricHasherConfig::LookupType lookup_type_tag_ =
      AsymmetricHasherConfig::FLOAT;

  bool disjoint_leaf_partitions_ = true;

  bool enable_global_topn_ = false;

  uint8_t global_topn_shift_ = 0;

  FRIEND_TEST(TreeAHHybridResidualTest, CrowdingMutation);
};

}  // namespace research_scann

#endif
