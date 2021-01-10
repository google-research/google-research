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



#ifndef SCANN_PARTITIONING_KMEANS_TREE_PARTITIONER_H_
#define SCANN_PARTITIONING_KMEANS_TREE_PARTITIONER_H_

#include <limits>

#include "absl/synchronization/mutex.h"
#include "scann/base/single_machine_base.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/trees/kmeans_tree/kmeans_tree_node.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class KMeansTreePartitioner final : public KMeansTreeLikePartitioner<T> {
 public:
  KMeansTreePartitioner(
      shared_ptr<const DistanceMeasure> database_tokenization_dist,
      shared_ptr<const DistanceMeasure> query_tokenization_dist);

  KMeansTreePartitioner(
      shared_ptr<const DistanceMeasure> database_tokenization_dist,
      shared_ptr<const DistanceMeasure> query_tokenization_dist,
      const SerializedKMeansTreePartitioner& proto);

  KMeansTreePartitioner(
      shared_ptr<const DistanceMeasure> database_tokenization_dist,
      shared_ptr<const DistanceMeasure> query_tokenization_dist,
      shared_ptr<const KMeansTree> pretrained_tree);

  unique_ptr<Partitioner<T>> Clone() const override;

  ~KMeansTreePartitioner() final;

  Status CreatePartitioning(const Dataset& training_dataset,
                            const DistanceMeasure& training_dist,
                            int32_t k_per_level,
                            KMeansTreeTrainingOptions* opts);

  void set_query_spilling_type(QuerySpillingConfig::SpillingType val) {
    query_spilling_type_ = val;
  }

  void set_query_spilling_threshold(double val);

  void set_query_spilling_max_centers(uint32_t val) {
    query_spilling_max_centers_ = val;
  }

  void set_database_spilling_fixed_number_of_centers(uint32_t val) {
    database_spilling_fixed_number_of_centers_ = val;
  }

  QuerySpillingConfig::SpillingType query_spilling_type() const {
    return query_spilling_type_;
  }

  double query_spilling_threshold() const { return query_spilling_threshold_; }

  uint32_t query_spilling_max_centers() const {
    return query_spilling_max_centers_;
  }

  uint32_t database_spilling_fixed_number_of_centers() const {
    return database_spilling_fixed_number_of_centers_;
  }

  enum TokenizationType {
    FLOAT = 1,

    FIXED_POINT_INT8 = 2,

    ASYMMETRIC_HASHING = 3
  };

  void SetQueryTokenizationType(TokenizationType type) {
    query_tokenization_type_ = type;
  }

  void SetDatabaseTokenizationType(TokenizationType type) {
    database_tokenization_type_ = type;
  }

  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           int32_t* result) const final;
  Status TokenForDatapointBatched(const TypedDataset<T>& queries,
                                  std::vector<int32_t>* results,
                                  ThreadPool* pool = nullptr) const final;
  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, std::vector<int32_t>* result) const final {
    return TokensForDatapointWithSpillingAndOverride(dptr, 0, result);
  }

  Status TokensForDatapointWithSpillingAndOverride(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      std::vector<int32_t>* result) const;

  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           KMeansTreeSearchResult* result) const;
  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      std::vector<KMeansTreeSearchResult>* result) const final;

  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries,
      MutableSpan<std::vector<int32_t>> results) const final {
    return TokensForDatapointWithSpillingBatchedAndOverride(
        queries, vector<int32_t>(), results);
  }
  Status TokensForDatapointWithSpillingBatchedAndOverride(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<int32_t>> results) const;
  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<KMeansTreeSearchResult>> results) const final;

  StatusOr<vector<std::vector<DatapointIndex>>> TokenizeDatabase(
      const TypedDataset<T>& database, ThreadPool* pool_or_null) const override;

  StatusOr<Datapoint<float>> ResidualizeToFloat(
      const DatapointPtr<T>& dptr, int32_t token,
      bool normalize_residual_by_cluster_stdev) const final;

  StatusOr<double> ResidualStdevForToken(int32_t token) const override {
    return kmeans_tree()->ResidualStdevForToken(token);
  }

  const DenseDataset<float>& LeafCenters() const;

  void CopyToProto(SerializedPartitioner* result) const final;

  int32_t n_tokens() const final;

  Normalization NormalizationRequired() const final;

  const shared_ptr<const DistanceMeasure>& database_tokenization_distance()
      const final {
    return database_tokenization_dist_;
  }

  const shared_ptr<const DistanceMeasure>& query_tokenization_distance()
      const final {
    return query_tokenization_dist_;
  }

  Status CreateAsymmetricHashingSearcherForDatabaseTokenization();

  Status CreateAsymmetricHashingSearcherForQueryTokenization(
      bool with_exact_reordering = true);

  const SingleMachineSearcherBase<float>* TokenizationSearcher() const;

  const shared_ptr<const KMeansTree>& kmeans_tree() const final {
    return kmeans_tree_;
  }

  bool populate_residual_stdev() const { return populate_residual_stdev_; }
  void set_populate_residual_stdev(bool val) { populate_residual_stdev_ = val; }

  bool SupportsLowLevelQueryBatching() const {
    return query_tokenization_type_ == FLOAT && is_one_level_tree_ &&
           ((typeid(*query_tokenization_dist_) ==
                 typeid(const DotProductDistance) ||
             typeid(*query_tokenization_dist_) ==
                 typeid(const SquaredL2Distance)));
  }

 private:
  Status TokenForDatapointUseSearcher(
      const DatapointPtr<T>& dptr, KMeansTreeSearchResult* result,
      int32_t pre_reordering_num_neighbors) const;
  Status TokensForDatapointWithSpillingUseSearcher(
      const DatapointPtr<T>& dptr, std::vector<KMeansTreeSearchResult>* result,
      int32_t num_neighbors, int32_t pre_reordering_num_neighbors) const;

  void SetIsOneLevelTree();

  StatusOr<std::vector<KMeansTreeSearchResult>> TokenizeDatabaseImplFastPath(
      const DenseDataset<T>& database, ThreadPool* pool_or_null) const;

  template <typename CenterType>
  enable_if_t<!IsSame<T, CenterType>(),
              StatusOr<std::vector<KMeansTreeSearchResult>>>
  TokenizeDatabaseImplFastPath(const DenseDataset<T>& database,
                               const DenseDataset<CenterType>& centers,
                               ThreadPool* pool_or_null) const;

  template <typename CenterType>
  enable_if_t<IsSame<T, CenterType>(),
              StatusOr<std::vector<KMeansTreeSearchResult>>>
  TokenizeDatabaseImplFastPath(const DenseDataset<T>& database,
                               const DenseDataset<CenterType>& centers,
                               ThreadPool* pool_or_null) const;

  template <typename FloatT>
  std::vector<KMeansTreeSearchResult> PostprocessNearestCenters(
      ConstSpan<pair<DatapointIndex, FloatT>> nearest_centers) const;

  const DenseDataset<float>* ConvertToFloatIfNecessary(
      const DenseDataset<T>& dataset, DenseDataset<float>* storage) const {
    if (std::is_same<T, float>::value) {
      return reinterpret_cast<const DenseDataset<float>*>(&dataset);
    } else {
      dataset.ConvertType(storage);
      return storage;
    }
  }

  shared_ptr<const KMeansTree> kmeans_tree_;

  shared_ptr<const DistanceMeasure> database_tokenization_dist_;
  shared_ptr<const DistanceMeasure> query_tokenization_dist_;

  std::vector<float> center_squared_norms_ = {};

  mutable absl::Mutex leaf_centers_mutex_;
  mutable DenseDataset<float> leaf_centers_
      ABSL_GUARDED_BY(leaf_centers_mutex_);

  QuerySpillingConfig::SpillingType query_spilling_type_ =
      QuerySpillingConfig::NO_SPILLING;

  double query_spilling_threshold_ = 1.0;

  int32_t query_spilling_max_centers_ = numeric_limits<int32_t>::max();

  int32_t database_spilling_fixed_number_of_centers_ = 0;

  bool ready_to_tokenize_ = false;

  TokenizationType query_tokenization_type_ = FLOAT;

  TokenizationType database_tokenization_type_ = FLOAT;

  bool is_one_level_tree_ = false;

  bool populate_residual_stdev_ = false;

  shared_ptr<const SingleMachineSearcherBase<float>>
      database_tokenization_searcher_ = nullptr;

  shared_ptr<const SingleMachineSearcherBase<float>>
      query_tokenization_searcher_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(KMeansTreePartitioner);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, KMeansTreePartitioner);

}  // namespace research_scann

#endif
