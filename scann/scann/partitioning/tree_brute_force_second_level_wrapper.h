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



#ifndef SCANN_PARTITIONING_TREE_BRUTE_FORCE_SECOND_LEVEL_WRAPPER_H_
#define SCANN_PARTITIONING_TREE_BRUTE_FORCE_SECOND_LEVEL_WRAPPER_H_

#include <utility>

#include "scann/brute_force/brute_force.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class TreeBruteForceSecondLevelWrapper final
    : public KMeansTreeLikePartitioner<T> {
 public:
  TreeBruteForceSecondLevelWrapper(
      unique_ptr<KMeansTreeLikePartitioner<T>> base);

  Status CreatePartitioning(const BottomUpTopLevelPartitioner& config);

  const shared_ptr<const DistanceMeasure>& query_tokenization_distance()
      const final {
    return base_->query_tokenization_distance();
  }

  const shared_ptr<const KMeansTree>& kmeans_tree() const final {
    return base_->kmeans_tree();
  }

  using Partitioner<T>::TokensForDatapointWithSpilling;
  using Partitioner<T>::TokensForDatapointWithSpillingBatched;
  using Partitioner<T>::TokenForDatapoint;
  using Partitioner<T>::TokenForDatapointBatched;

  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      vector<pair<DatapointIndex, float>>* result) const final;

  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<pair<DatapointIndex, float>>> results,
      ThreadPool* pool = nullptr) const final;

  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           pair<DatapointIndex, float>* result) const final {
    SCANN_RET_CHECK_EQ(this->tokenization_mode(), UntypedPartitioner::DATABASE);
    return base_->TokenForDatapoint(dptr, result);
  }

  Status TokenForDatapointBatched(
      const TypedDataset<T>& queries,
      std::vector<pair<DatapointIndex, float>>* result,
      ThreadPool* pool) const final {
    SCANN_RET_CHECK_EQ(this->tokenization_mode(), UntypedPartitioner::DATABASE);
    return base_->TokenForDatapointBatched(queries, result, pool);
  }

  StatusOr<Datapoint<float>> ResidualizeToFloat(const DatapointPtr<T>& dptr,
                                                int32_t token) const final {
    return base_->ResidualizeToFloat(dptr, token);
  }

  const DenseDataset<float>& LeafCenters() const final {
    return base_->LeafCenters();
  }

  void CopyToProto(SerializedPartitioner* result) const final {
    base_->CopyToProto(result);
  }
  int32_t n_tokens() const final { return base_->n_tokens(); }
  unique_ptr<Partitioner<T>> Clone() const final {
    LOG(FATAL) << "Not implemented";
  }

  Status TokenForDatapoint(const DatapointPtr<T>& query,
                           int32_t* result) const final {
    SCANN_RET_CHECK_EQ(this->tokenization_mode(), UntypedPartitioner::DATABASE);
    return base_->TokenForDatapoint(query, result);
  }
  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& query, std::vector<int32_t>* result) const final;

  Status TokenForDatapointBatched(const TypedDataset<T>& queries,
                                  std::vector<int32_t>* results,
                                  ThreadPool* pool = nullptr) const final {
    SCANN_RET_CHECK_EQ(this->tokenization_mode(), UntypedPartitioner::DATABASE);
    return base_->TokenForDatapointBatched(queries, results, pool);
  }

  StatusOr<vector<std::vector<DatapointIndex>>> TokenizeDatabase(
      const TypedDataset<T>& database, ThreadPool* pool_or_null) const final {
    return base_->TokenizeDatabase(database, pool_or_null);
  }

  uint32_t query_spilling_max_centers() const final {
    return base_->query_spilling_max_centers();
  }

 private:
  TreeBruteForceSecondLevelWrapper() = default;
  static StatusOrPtr<TreeXHybridSMMD<float>> CreateTopLevel(
      const KMeansTreeLikePartitioner<T>& base,
      const BottomUpTopLevelPartitioner& config);

  unique_ptr<KMeansTreeLikePartitioner<T>> base_;
  unique_ptr<TreeXHybridSMMD<float>> top_level_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TreeBruteForceSecondLevelWrapper);

}  // namespace research_scann

#endif
