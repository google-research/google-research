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



#ifndef SCANN_PARTITIONING_KMEANS_TREE_LIKE_PARTITIONER_H_
#define SCANN_PARTITIONING_KMEANS_TREE_LIKE_PARTITIONER_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class KMeansTreeLikePartitioner : public Partitioner<T> {
 public:
  virtual const shared_ptr<const DistanceMeasure>& query_tokenization_distance()
      const = 0;

  virtual const shared_ptr<const KMeansTree>& kmeans_tree() const = 0;

  using Partitioner<T>::TokensForDatapointWithSpilling;
  using Partitioner<T>::TokensForDatapointWithSpillingBatched;
  using Partitioner<T>::TokenForDatapoint;
  using Partitioner<T>::TokenForDatapointBatched;

  virtual Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      vector<pair<DatapointIndex, float>>* result) const = 0;

  virtual Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<pair<DatapointIndex, float>>> results,
      ThreadPool* pool = nullptr) const = 0;

  virtual Status TokenForDatapoint(
      const DatapointPtr<T>& dptr,
      pair<DatapointIndex, float>* result) const = 0;

  virtual Status TokenForDatapointBatched(
      const TypedDataset<T>& queries,
      std::vector<pair<DatapointIndex, float>>* result,
      ThreadPool* pool) const = 0;
  Status TokenForDatapointBatched(
      const TypedDataset<T>& queries,
      std::vector<pair<DatapointIndex, float>>* result) const {
    return TokenForDatapointBatched(queries, result, nullptr);
  }

  virtual StatusOr<Datapoint<float>> ResidualizeToFloat(
      const DatapointPtr<T>& dptr, int32_t token) const = 0;

  StatusOr<Datapoint<float>> ResidualizeToFloat(const DatapointPtr<T>& dptr,
                                                int32_t token, bool) const {
    return ResidualizeToFloat(dptr, token);
  }

  virtual const DenseDataset<float>& LeafCenters() const = 0;

  virtual uint32_t query_spilling_max_centers() const = 0;
};

}  // namespace research_scann

#endif
