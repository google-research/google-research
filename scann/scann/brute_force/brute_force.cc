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

#include "scann/brute_force/brute_force.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/many_to_many/many_to_many_floating_point.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
BruteForceSearcher<T>::BruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const TypedDataset<T>> dataset,
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase<T>(dataset,
                                   default_pre_reordering_num_neighbors,
                                   default_pre_reordering_epsilon),
      distance_(distance),
      supports_low_level_batching_(
          (typeid(*distance) == typeid(DotProductDistance) ||
           typeid(*distance) == typeid(CosineDistance) ||
           typeid(*distance) == typeid(SquaredL2Distance)) &&
          dataset->IsDense() && IsFloatingType<T>()) {}

template <typename T>
BruteForceSearcher<T>::~BruteForceSearcher() {}

template <typename T>
Status BruteForceSearcher<T>::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (datapoint_index_to_crowding_attribute.size() != this->dataset()->size()) {
    return InvalidArgumentError(absl::StrCat(
        "datapoint_index_to_crowding_attribute must have size equal to "
        "number of datapoints.  (",
        datapoint_index_to_crowding_attribute.size(), " vs. ",
        this->dataset()->size(), "."));
  }
  return OkStatus();
}

namespace {

struct EmptyStruct {};

template <typename ResultType>
class TopNWrapperInterface {
 public:
  virtual ~TopNWrapperInterface() {}
  virtual void PushBatch(ConstSpan<ResultType> result_block,
                         size_t base_dp_idx) = 0;
  virtual NNResultsVector TakeUnsorted() = 0;
};

template <typename TopNBase, typename ResultType, bool kUseMinDistance>
class TopNWrapper final : public TopNWrapperInterface<ResultType> {
 public:
  TopNWrapper(TopNBase base, ResultType epsilon, ResultType min_distance)
      : base_(std::move(base)), epsilon_(epsilon) {
    if constexpr (kUseMinDistance) {
      min_distance_ = min_distance;
    }
  }

  void PushBatch(ConstSpan<ResultType> result_block, size_t base_dp_idx) final {
    for (size_t i : IndicesOf(result_block)) {
      const ResultType dist = result_block[i];
      if (ShouldPush(dist)) {
        base_.push(std::make_pair(base_dp_idx + i, dist));
        if (base_.full()) epsilon_ = base_.approx_bottom().second;
      }
    }
  }

  NNResultsVector TakeUnsorted() final { return base_.TakeUnsorted(); }

 private:
  bool ShouldPush(ResultType dist) const {
    if constexpr (kUseMinDistance) {
      return dist <= epsilon_ && dist >= min_distance_;
    } else {
      return dist <= epsilon_;
    }
  }

  TopNBase base_;
  ResultType epsilon_;
  std::conditional_t<kUseMinDistance, ResultType, EmptyStruct> min_distance_;
};

template <typename TopNBase, typename ResultType, bool kUseMinDistance>
class TopNWrapperThreadSafe final : public TopNWrapperInterface<ResultType> {
 public:
  TopNWrapperThreadSafe(TopNBase base, ResultType epsilon,
                        ResultType min_distance)
      : base_(std::move(base)), epsilon_(epsilon) {
    if constexpr (kUseMinDistance) {
      min_distance_ = min_distance;
    }
  }

  void PushBatch(ConstSpan<ResultType> result_block, size_t base_dp_idx) final {
    constexpr size_t kBufferSize = 16;
    pair<DatapointIndex, ResultType> buf[kBufferSize];
    auto eps = epsilon_.load(std::memory_order_relaxed);
    size_t num_to_push = 0;

    auto locked_push_results = [&] {
      absl::MutexLock lock(&mutex_);
      for (size_t i : Seq(num_to_push)) {
        base_.push(buf[i]);
        if (base_.full()) {
          eps = base_.approx_bottom().second;
          epsilon_.store(eps, std::memory_order_relaxed);
        }
      }
      num_to_push = 0;
    };

    for (size_t i : IndicesOf(result_block)) {
      auto dist = result_block[i];
      if (dist > eps || !AboveMinDist(dist)) continue;
      buf[num_to_push++] = {i + base_dp_idx, dist};
      if (num_to_push == kBufferSize) locked_push_results();
    }
    if (num_to_push) locked_push_results();
  }

  NNResultsVector TakeUnsorted() final { return base_.TakeUnsorted(); }

 private:
  bool AboveMinDist(ResultType dist) const {
    if constexpr (kUseMinDistance) {
      return dist >= min_distance_;
    } else {
      return true;
    }
  }

  TopNBase base_;
  std::atomic<ResultType> epsilon_;
  std::conditional_t<kUseMinDistance, ResultType, EmptyStruct> min_distance_;
  mutable absl::Mutex mutex_;
};

template <typename ResultType, typename TopNBase>
unique_ptr<TopNWrapperInterface<ResultType>> MakeTopNWrapper(
    TopNBase base, ResultType epsilon, ResultType min_distance,
    bool thread_safe) {
  if (thread_safe) {
    if (min_distance == -numeric_limits<float>::infinity()) {
      return unique_ptr<TopNWrapperInterface<ResultType>>(
          new TopNWrapperThreadSafe<TopNBase, ResultType, false>(
              std::move(base), epsilon, min_distance));
    } else {
      return unique_ptr<TopNWrapperInterface<ResultType>>(
          new TopNWrapperThreadSafe<TopNBase, ResultType, true>(
              std::move(base), epsilon, min_distance));
    }
  } else {
    if (min_distance == -numeric_limits<float>::infinity()) {
      return unique_ptr<TopNWrapperInterface<ResultType>>(
          new TopNWrapper<TopNBase, ResultType, false>(std::move(base), epsilon,
                                                       min_distance));
    } else {
      return unique_ptr<TopNWrapperInterface<ResultType>>(
          new TopNWrapper<TopNBase, ResultType, true>(std::move(base), epsilon,
                                                      min_distance));
    }
  }
}

class FastTopNeighborsWrapper : public TopNWrapperInterface<float> {
 public:
  FastTopNeighborsWrapper(int32_t num_neighbors, float epsilon)
      : fast_top_neighbors_(num_neighbors, epsilon) {}

  void PushBatch(ConstSpan<float> result_block, size_t base_dp_idx) final {
    return fast_top_neighbors_.PushBlock(result_block, base_dp_idx);
  }

  NNResultsVector TakeUnsorted() final {
    NNResultsVector result;
    fast_top_neighbors_.FinishUnsorted(&result);
    return result;
  }

 private:
  FastTopNeighbors<float> fast_top_neighbors_;
};

class FastTopNeighborsWrapperThreadSafe final
    : public TopNWrapperInterface<float> {
 public:
  FastTopNeighborsWrapperThreadSafe(int32_t num_neighbors, float epsilon)
      : fast_top_neighbors_(num_neighbors, epsilon), epsilon_(epsilon) {}

  void PushBatch(ConstSpan<float> result_block, size_t base_dp_idx) final {
    constexpr size_t kBufferSize = 16;
    pair<DatapointIndex, float> buf[kBufferSize];
    float eps = epsilon_.load(std::memory_order_relaxed);
    size_t num_to_push = 0;

    auto locked_push_results = [&] {
      absl::MutexLock lock(&mutex_);
      FastTopNeighbors<float>::Mutator mut;
      fast_top_neighbors_.AcquireMutator(&mut);
      eps = mut.epsilon();
      for (size_t i : Seq(num_to_push)) {
        const float dist = buf[i].second;
        DCHECK_EQ(eps, mut.epsilon());
        if (dist > eps) continue;
        const DatapointIndex dp_idx = buf[i].first;
        if (mut.Push(dp_idx, dist)) {
          mut.GarbageCollect();
          eps = mut.epsilon();
          epsilon_.store(eps, std::memory_order_relaxed);
        }
      }
      num_to_push = 0;
    };

    for (size_t i : IndicesOf(result_block)) {
      const float dist = result_block[i];
      if (dist > eps) continue;
      buf[num_to_push++] = {i + base_dp_idx, dist};
      if (num_to_push == kBufferSize) locked_push_results();
    }
    if (num_to_push) locked_push_results();
  }

  NNResultsVector TakeUnsorted() final {
    NNResultsVector result;
    fast_top_neighbors_.FinishUnsorted(&result);
    return result;
  }

 private:
  FastTopNeighbors<float> fast_top_neighbors_;
  std::atomic<float> epsilon_;
  mutable absl::Mutex mutex_;
};

template <typename ResultType>
unique_ptr<TopNWrapperInterface<ResultType>> MakeNonCrowdingTopN(
    const SearchParameters& params, float min_distance, bool thread_safe) {
  if (min_distance == -numeric_limits<float>::infinity()) {
    if constexpr (IsSame<ResultType, float>()) {
      if (thread_safe) {
        return make_unique<FastTopNeighborsWrapperThreadSafe>(
            params.pre_reordering_num_neighbors(),
            params.pre_reordering_epsilon());
      } else {
        return make_unique<FastTopNeighborsWrapper>(
            params.pre_reordering_num_neighbors(),
            params.pre_reordering_epsilon());
      }
    }
  }

  return MakeTopNWrapper<ResultType>(
      TopNeighbors<float>(params.pre_reordering_num_neighbors()),
      params.pre_reordering_epsilon(), min_distance, thread_safe);
}

}  // namespace

template <typename T>
template <typename Float>
enable_if_t<!IsSameAny<Float, float, double>(), void>
BruteForceSearcher<T>::FinishBatchedSearch(
    const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
    ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  LOG(FATAL) << "Low-level batching only works and should only be called with "
                "float/double types.  This codepath should be impossible.";
}

template <typename T>
template <typename Float>
enable_if_t<IsSameAny<Float, float, double>(), void>
BruteForceSearcher<T>::FinishBatchedSearch(
    const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
    ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if constexpr (std::is_same_v<Float, float>) {
    if (min_distance_ == -numeric_limits<float>::infinity() &&
        std::all_of(params.begin(), params.end(),
                    [](const SearchParameters& params) {
                      return !params.pre_reordering_crowding_enabled();
                    })) {
      return FinishBatchedSearchSimple(db, queries, params, results);
    }
  }

  vector<unique_ptr<TopNWrapperInterface<Float>>> top_ns(queries.size());
  for (size_t i : IndicesOf(params)) {
    if (params[i].pre_reordering_crowding_enabled()) {
    } else {
      top_ns[i] =
          MakeNonCrowdingTopN<Float>(params[i], min_distance_, pool_.get());
    }
  }

  auto write_to_top_n = [&](MutableSpan<Float> result_block,
                            DatapointIndex base_dp_idx,
                            DatapointIndex query_idx) {
    auto& top_n = top_ns[query_idx];
    top_n->PushBatch(result_block, base_dp_idx);
  };

  DenseDistanceManyToMany<Float>(*distance_, queries, db, pool_.get(),
                                 write_to_top_n);
  for (size_t i : IndicesOf(top_ns)) {
    results[i] = top_ns[i]->TakeUnsorted();
  }
}

template <typename T>
void BruteForceSearcher<T>::FinishBatchedSearchSimple(
    const DenseDataset<float>& db, const DenseDataset<float>& queries,
    ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  vector<FastTopNeighbors<float>> top_ns(queries.size());
  for (size_t i : IndicesOf(params)) {
    top_ns[i].Init(params[i].pre_reordering_num_neighbors(),
                   params[i].pre_reordering_epsilon());
  }
  DenseDistanceManyToManyTopK(*distance_, queries, db, MakeMutableSpan(top_ns),
                              pool_.get());
  for (size_t i : IndicesOf(top_ns)) {
    top_ns[i].FinishUnsorted(&results[i]);
  }
}

template <typename T>
Status BruteForceSearcher<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if (!supports_low_level_batching_ || !queries.IsDense()) {
    return SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
        queries, params, results);
  }

  for (const SearchParameters& p : params) {
    if (p.restricts_enabled()) {
      return SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
          queries, params, results);
    }
  }
  const DenseDataset<T>& database =
      *down_cast<const DenseDataset<T>*>(this->dataset());
  const DenseDataset<T>& queries_dense =
      *down_cast<const DenseDataset<T>*>(&queries);
  FinishBatchedSearch<T>(database, queries_dense, params, results);
  return OkStatus();
}

template <typename T>
Status BruteForceSearcher<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<FastTopNeighbors<float>*> results,
    ConstSpan<DatapointIndex> datapoint_index_mapping) const {
  auto fallback = [&] {
    return SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
        queries, params, results, datapoint_index_mapping);
  };
  if constexpr (!std::is_same_v<T, float>) return fallback();
  if (!supports_low_level_batching_) return fallback();
  if (!queries.IsDense()) return fallback();
  if (min_distance_ != -numeric_limits<float>::infinity()) return fallback();
  if (std::any_of(params.begin(), params.end(), [](const SearchParameters& p) {
        return p.restricts_enabled() || p.pre_reordering_crowding_enabled();
      })) {
    return fallback();
  }

  const DenseDataset<float>& database =
      *reinterpret_cast<const DenseDataset<float>*>(this->dataset());
  const DenseDataset<float>& queries_dense =
      *reinterpret_cast<const DenseDataset<float>*>(&queries);
  DenseDistanceManyToManyTopKRemapped(*distance_, queries_dense, database,
                                      results, datapoint_index_mapping,
                                      pool_.get());
  return OkStatus();
}

template <typename T>
Status BruteForceSearcher<T>::FindNeighborsImpl(const DatapointPtr<T>& query,
                                                const SearchParameters& params,
                                                NNResultsVector* result) const {
  DCHECK(result);
  const bool use_min_distance =
      min_distance_ != -numeric_limits<float>::infinity();
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    TopNeighbors<float> top_n(params.pre_reordering_num_neighbors());
    Status internal_status =
        use_min_distance ? FindNeighborsInternal<true>(query, params, &top_n)
                         : FindNeighborsInternal<false>(query, params, &top_n);
    SCANN_RETURN_IF_ERROR(internal_status);
    *result = top_n.TakeUnsorted();
  }
  return OkStatus();
}

template <typename T>
template <bool kUseMinDistance, typename TopN>
Status BruteForceSearcher<T>::FindNeighborsInternal(
    const DatapointPtr<T>& query, const SearchParameters& params,
    TopN* top_n_ptr) const {
  DCHECK(top_n_ptr);

  if (query.IsDense() && this->dataset()->IsDense()) {
    TopN top_n = std::move(*top_n_ptr);
    const float epsilon = params.pre_reordering_epsilon();
    float min_keep_distance = epsilon;

    const float min_distance = min_distance_;

    auto should_push = [&](float dist) SCANN_INLINE_LAMBDA {
      if constexpr (kUseMinDistance) {
        return dist <= min_keep_distance && dist >= min_distance;
      } else {
        return dist <= min_keep_distance;
      }
    };

    const DenseDataset<T>& dataset =
        *down_cast<const DenseDataset<T>*>(this->dataset());

    if (params.restricts_enabled()) {
    } else {
      unique_ptr<float[]> distances_storage(new float[dataset.size()]);
      MutableSpan<float> distances(distances_storage.get(), dataset.size());
      DenseDistanceOneToMany<T, float>(*distance_, query, dataset, distances);
      for (DatapointIndex i : IndicesOf(dataset)) {
        const float dist = distances[i];
        if (should_push(dist)) {
          top_n.push(std::make_pair(i, dist));
          if (top_n.full()) {
            min_keep_distance = top_n.approx_bottom().second;
          }
        }
      }
    }
    *top_n_ptr = std::move(top_n);
  } else if (params.restricts_enabled()) {
    auto it = params.restrict_whitelist()->AllowlistedPointIterator();
    FindNeighborsOneToOneInternal<kUseMinDistance>(query, params, &it,
                                                   top_n_ptr);
  } else {
    DummyAllowlist allowlist(this->dataset()->size());
    auto it = allowlist.AllowlistedPointIterator();
    FindNeighborsOneToOneInternal<kUseMinDistance>(query, params, &it,
                                                   top_n_ptr);
  }
  return OkStatus();
}

template <typename T>
template <bool kUseMinDistance, typename AllowlistIterator, typename TopN>
void BruteForceSearcher<T>::FindNeighborsOneToOneInternal(
    const DatapointPtr<T>& query, const SearchParameters& params,
    AllowlistIterator* allowlist_iterator, TopN* top_n_ptr) const {
  DCHECK(top_n_ptr);

  TopN top_n = std::move(*top_n_ptr);
  const float epsilon = params.pre_reordering_epsilon();
  float min_keep_distance = epsilon;

  const float min_distance = min_distance_;

  auto should_push = [&](float dist) SCANN_INLINE_LAMBDA {
    if constexpr (kUseMinDistance) {
      return dist <= min_keep_distance && dist >= min_distance;
    } else {
      return dist <= min_keep_distance;
    }
  };

  if (query.IsDense() && this->dataset()->IsDense()) {
    const DenseDataset<T>& dataset =
        *down_cast<const DenseDataset<T>*>(this->dataset());
    for (; !allowlist_iterator->Done(); allowlist_iterator->Next()) {
      const DatapointIndex i = allowlist_iterator->value();
      const DatapointPtr<T> dptr = dataset[i];
      const double dist = distance_->GetDistanceDense(query, dptr);
      if (should_push(dist)) {
        top_n.push(std::make_pair(i, dist));
        if (top_n.full()) {
          min_keep_distance = top_n.approx_bottom().second;
        }
      }
    }
  } else if (query.IsSparse() && this->dataset()->IsSparse()) {
    const SparseDataset<T>& dataset =
        *down_cast<const SparseDataset<T>*>(this->dataset());
    for (; !allowlist_iterator->Done(); allowlist_iterator->Next()) {
      const DatapointIndex i = allowlist_iterator->value();
      const DatapointPtr<T> dptr = dataset[i];
      const double dist = distance_->GetDistanceSparse(query, dptr);
      if (should_push(dist)) {
        top_n.push(std::make_pair(i, dist));
        if (top_n.full()) {
          min_keep_distance = top_n.approx_bottom().second;
        }
      }
    }
  } else {
    for (; !allowlist_iterator->Done(); allowlist_iterator->Next()) {
      const DatapointIndex i = allowlist_iterator->value();
      const DatapointPtr<T> dptr = (*this->dataset())[i];
      const double dist = distance_->GetDistanceHybrid(query, dptr);
      if (should_push(dist)) {
        top_n.push(std::make_pair(i, dist));
        if (top_n.full()) {
          min_keep_distance = top_n.approx_bottom().second;
        }
      }
    }
  }
  *top_n_ptr = std::move(top_n);
}

template <typename T>
StatusOr<typename SingleMachineSearcherBase<T>::Mutator*>
BruteForceSearcher<T>::GetMutator() const {
  if (is_immutable_) {
    DCHECK(!mutator_);
    return FailedPreconditionError(
        "Cannot GetMutator on an immutable BruteForceSearcher.");
  }
  if (!mutator_) {
    auto mutable_this = const_cast<BruteForceSearcher<T>*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_, BruteForceSearcher<T>::Mutator::Create(mutable_this));
    SCANN_RETURN_IF_ERROR(mutator_->PrepareForBaseMutation(mutable_this));
  }
  return static_cast<typename SingleMachineSearcherBase<T>::Mutator*>(
      mutator_.get());
}

SCANN_INSTANTIATE_TYPED_CLASS(, BruteForceSearcher);

}  // namespace research_scann
