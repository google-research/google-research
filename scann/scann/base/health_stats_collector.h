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

#ifndef SCANN_BASE_HEALTH_STATS_COLLECTOR_H_
#define SCANN_BASE_HEALTH_STATS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename Searcher, typename InDataType,

          typename InAccamulationType = InDataType,
          typename Partitioner =
              KMeansTreePartitioner<typename Searcher::DataType>>
class HealthStatsCollector {
 public:
  using DataType = InDataType;
  using AccamulationType = InAccamulationType;
  using HealthStats = typename Searcher::HealthStats;

  Status Initialize(const Searcher& searcher);

  absl::StatusOr<HealthStats> GetHealthStats();

  bool IsEnabled() const { return is_enabled_; }

  uint32_t NumTokens() const { return sizes_by_token_.size(); }

  void AddPartition();

  void SwapPartitions(int32_t token1, int32_t token2);

  void RemoveLastPartition();

  void Resize(size_t n);

  void Reserve(size_t n);

  void AddStats(int32_t token, absl::Span<const DatapointIndex> datapoints) {
    AddStats(absl::MakeConstSpan({token}), datapoints);
  }
  template <typename Tokens>
  void AddStats(const Tokens& tokens,
                absl::Span<const DatapointIndex> datapoints);
  void SubtractStats(int32_t token,
                     absl::Span<const DatapointIndex> datapoints) {
    SubtractStats(absl::MakeConstSpan({token}), datapoints);
  }
  template <typename Tokens>
  void SubtractStats(const Tokens& tokens,
                     absl::Span<const DatapointIndex> datapoints);

  void SubtractPartition(int32_t token);

  void UpdatePartitionCentroid(int32_t token,
                               DatapointPtr<DataType> new_centroid,
                               DatapointPtr<DataType> old_centroid);

 private:
  Status InitializeCentroids(const Searcher& searcher);

  enum class Op {
    Add,
    Subtract,
  };
  template <typename Tokens>
  void StatsUpdate(const Tokens& tokens, Op op,
                   absl::Span<const DatapointIndex> datapoints);

  void Add(int32_t token, DatapointPtr<DataType> dp_ptr,
           DatapointPtr<DataType> center);
  void Add(int32_t token);

  void Subtract(int32_t token, DatapointPtr<DataType> dp_ptr,
                DatapointPtr<DataType> center);
  void Subtract(int32_t token);

  static void AddDelta(Datapoint<InAccamulationType>& dst,
                       DatapointPtr<DataType> new_dp,
                       DatapointPtr<DataType> old_dp, int times = 1);

  void ComputeAvgRelativeImbalance();

  DatapointPtr<DataType> GetDatapointPtr(
      DatapointIndex i, Datapoint<typename Searcher::DataType>* storage) const {
    return searcher_->GetDatapointPtr(i);
  }

  const Searcher* searcher_ = nullptr;
  InAccamulationType sum_squared_quantization_error_ = 0;
  double partition_avg_relative_imbalance_ = 0;
  uint64_t sum_partition_sizes_ = 0;

  std::vector<Datapoint<InAccamulationType>> sum_qe_by_token_;

  std::vector<uint32_t> sizes_by_token_;

  std::vector<InAccamulationType> squared_quantization_error_by_token_;
  std::shared_ptr<Partitioner> centroids_;
  bool is_enabled_ = false;

  static constexpr bool kCentroidAndDPAreSameType =
      std::is_same_v<DataType, typename Searcher::DataType>;
};

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
Status HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                            Partitioner>::Initialize(const Searcher& searcher) {
  *this = HealthStatsCollector();
  is_enabled_ = true;
  searcher_ = &searcher;
  SCANN_RETURN_IF_ERROR(InitializeCentroids(searcher));

  ConstSpan<std::vector<DatapointIndex>> datapoints_by_token =
      searcher.datapoints_by_token();
  Reserve(datapoints_by_token.size());
  for (const auto& dps : datapoints_by_token) {
    sum_partition_sizes_ += dps.size();
    sizes_by_token_.push_back(dps.size());
    sum_qe_by_token_.emplace_back();
    squared_quantization_error_by_token_.emplace_back();
  }

  const auto* dataset = searcher.dataset();
  if constexpr (kCentroidAndDPAreSameType) {
    if (dataset && !dataset->empty()) {
      const auto& ds = *dataset;
      InAccamulationType total_squared_qe = 0.0;
      const auto& centroids = centroids_->LeafCenters();
      for (const auto& [token, dps] : Enumerate(datapoints_by_token)) {
        Datapoint<InAccamulationType> sum_dims;
        sum_dims.ZeroFill(ds.dimensionality());
        SCANN_RET_CHECK_EQ(sum_dims.dimensionality(), ds.dimensionality());
        SCANN_RET_CHECK_EQ(sum_dims.values_span().size(), ds.dimensionality());
        DatapointPtr<DataType> centroid = centroids[token];
        InAccamulationType v = 0;
        for (auto dp_idx : dps) {
          v += SquaredL2DistanceBetween(ds[dp_idx], centroid);
          AddDelta(sum_dims, ds[dp_idx], centroids[token]);
        }
        squared_quantization_error_by_token_[token] = v;
        sum_qe_by_token_[token] = std::move(sum_dims);
        total_squared_qe += v;
      }

      sum_squared_quantization_error_ = total_squared_qe;
    }
  }

  ComputeAvgRelativeImbalance();
  return OkStatus();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
absl::StatusOr<typename HealthStatsCollector<
    Searcher, InDataType, InAccamulationType, Partitioner>::HealthStats>
HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                     Partitioner>::GetHealthStats() {
  HealthStats r;
  if (sum_partition_sizes_ > 0) {
    r.avg_quantization_error =
        sqrt(sum_squared_quantization_error_ / sum_partition_sizes_);
  }

  r.sum_partition_sizes = sum_partition_sizes_;

  ComputeAvgRelativeImbalance();
  r.partition_avg_relative_imbalance = partition_avg_relative_imbalance_;
  return r;
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::AddPartition() {
  if (!is_enabled_) return;
  sum_qe_by_token_.emplace_back();
  sizes_by_token_.push_back(0);
  squared_quantization_error_by_token_.push_back(0);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::SwapPartitions(int32_t token1,
                                                       int32_t token2) {
  if (!is_enabled_) return;
  std::swap(sum_qe_by_token_[token1], sum_qe_by_token_[token2]);
  std::swap(sizes_by_token_[token1], sizes_by_token_[token2]);
  std::swap(squared_quantization_error_by_token_[token1],
            squared_quantization_error_by_token_[token2]);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::RemoveLastPartition() {
  if (!is_enabled_) return;
  sum_qe_by_token_.pop_back();
  sizes_by_token_.pop_back();
  squared_quantization_error_by_token_.pop_back();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Resize(size_t n) {
  if (!is_enabled_) return;
  sum_qe_by_token_.resize(n);
  sizes_by_token_.resize(n);
  squared_quantization_error_by_token_.resize(n);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Reserve(size_t n) {
  if (!is_enabled_) return;
  sum_qe_by_token_.reserve(n);
  sizes_by_token_.reserve(n);
  squared_quantization_error_by_token_.reserve(n);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::AddStats(const Tokens& tokens,
                           absl::Span<const DatapointIndex> datapoints) {
  if (!is_enabled_) return;
  StatsUpdate(tokens, Op::Add, datapoints);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::SubtractStats(const Tokens& tokens,
                                absl::Span<const DatapointIndex> datapoints) {
  if (!is_enabled_) return;
  StatsUpdate(tokens, Op::Subtract, datapoints);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::SubtractPartition(int32_t token) {
  if (!is_enabled_) return;

  sum_squared_quantization_error_ -=
      squared_quantization_error_by_token_[token];
  sum_partition_sizes_ -= sizes_by_token_[token];

  sum_qe_by_token_[token].ZeroFill(sum_qe_by_token_[token].dimensionality());
  sizes_by_token_[token] = 0;
  squared_quantization_error_by_token_[token] = 0;
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::UpdatePartitionCentroid(int32_t token,
                                          DatapointPtr<DataType> new_centroid,
                                          DatapointPtr<DataType> old_centroid) {
  if (!is_enabled_) return;

  if constexpr (kCentroidAndDPAreSameType) {
    if (sizes_by_token_[token] == 0) return;

    if (sum_qe_by_token_[token].dimensionality() == 0) {
      sum_qe_by_token_[token].ZeroFill(new_centroid.dimensionality());
    }
    InAccamulationType delta = 0;
    for (int dim = 0; dim < new_centroid.dimensionality(); ++dim) {
      auto d =
          new_centroid.values_span()[dim] - old_centroid.values_span()[dim];
      InAccamulationType v = sizes_by_token_[token] * d * d -
                             2 * d * sum_qe_by_token_[token].values_span()[dim];
      delta += v;
    }
    sum_squared_quantization_error_ += delta;
    squared_quantization_error_by_token_[token] += delta;

    AddDelta(sum_qe_by_token_[token], old_centroid, new_centroid,
             sizes_by_token_[token]);
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
Status HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                            Partitioner>::InitializeCentroids(const Searcher&
                                                                  searcher) {
  auto pd = std::dynamic_pointer_cast<const Partitioner>(
      searcher_->database_tokenizer());
  auto pq = std::dynamic_pointer_cast<const Partitioner>(
      searcher_->query_tokenizer());
  SCANN_RET_CHECK(pd != nullptr);
  SCANN_RET_CHECK(pq != nullptr);
  SCANN_RET_CHECK_EQ(pd->kmeans_tree(), pq->kmeans_tree())
      << "Centroids in database partitioner and query partitioner must be "
      << "identical";
  SCANN_RET_CHECK(pq->kmeans_tree()->is_flat())
      << "The query/database partitioner must contain a single flat "
      << "KMeansTree.";

  centroids_ = std::const_pointer_cast<Partitioner>(pq);
  return OkStatus();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::StatsUpdate(const Tokens& tokens, Op op,
                              absl::Span<const DatapointIndex> datapoints) {
  if constexpr (kCentroidAndDPAreSameType) {
    const auto& centroids = centroids_->LeafCenters();
    Datapoint<DataType> dp;
    for (int32_t token : tokens) {
      DatapointPtr<DataType> centroid = centroids[token];
      for (DatapointIndex dp_idx : datapoints) {
        auto d_ptr = GetDatapointPtr(dp_idx, &dp);
        if (op == Op::Add) {
          Add(token, d_ptr, centroid);
        } else {
          Subtract(token, d_ptr, centroid);
        }
      }
    }
  } else {
    for (int32_t token : tokens) {
      if (op == Op::Add) {
        Add(token);
      } else {
        Subtract(token);
      }
    }
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Add(int32_t token,
                                            DatapointPtr<DataType> dp_ptr,
                                            DatapointPtr<DataType> center) {
  double quantize_err = SquaredL2DistanceBetween(dp_ptr, center);
  sum_squared_quantization_error_ += quantize_err;

  AddDelta(sum_qe_by_token_[token], dp_ptr, center);
  squared_quantization_error_by_token_[token] += quantize_err;
  Add(token);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Add(int32_t token) {
  ++sum_partition_sizes_;
  ++sizes_by_token_[token];
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Subtract(int32_t token,
                                                 DatapointPtr<DataType> dp_ptr,
                                                 DatapointPtr<DataType>
                                                     center) {
  double quantize_err = SquaredL2DistanceBetween(dp_ptr, center);
  sum_squared_quantization_error_ -= quantize_err;

  AddDelta(sum_qe_by_token_[token], center, dp_ptr);
  squared_quantization_error_by_token_[token] -= quantize_err;
  Subtract(token);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Subtract(int32_t token) {
  --sum_partition_sizes_;
  --sizes_by_token_[token];
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::AddDelta(Datapoint<InAccamulationType>&
                                                     dst,
                                                 DatapointPtr<DataType> new_dp,
                                                 DatapointPtr<DataType> old_dp,
                                                 int times) {
  if (dst.dimensionality() == 0) dst.ZeroFill(new_dp.dimensionality());
  for (int dim = 0; dim < dst.dimensionality(); ++dim) {
    dst.mutable_values_span()[dim] +=
        (new_dp.values_span()[dim] - old_dp.values_span()[dim]) * times;
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::ComputeAvgRelativeImbalance() {
  partition_avg_relative_imbalance_ = 0;
  if (sum_partition_sizes_ == 0) return;

  for (const auto& partition_size : sizes_by_token_) {
    partition_avg_relative_imbalance_ +=
        1.0 * partition_size / sum_partition_sizes_ * partition_size;
  }
  partition_avg_relative_imbalance_ /=
      1.0 * sum_partition_sizes_ / sizes_by_token_.size();
  partition_avg_relative_imbalance_ -= 1.0;
}

}  // namespace research_scann
#endif
