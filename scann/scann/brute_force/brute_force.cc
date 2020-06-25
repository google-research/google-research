// Copyright 2020 The Google Research Authors.
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

#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/utils/common.h"

#include "absl/synchronization/mutex.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

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
          dataset->IsDense() && IsFloatingType<T>()),
      using_squared_db_norms_(supports_low_level_batching_ &&
                              typeid(*distance) == typeid(SquaredL2Distance)) {
  if (using_squared_db_norms_) {
    squared_db_norms_.reserve(dataset->size());
    for (auto dptr : *dataset) {
      squared_db_norms_.push_back(SquaredL2Norm(dptr));
    }
  }
}

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
  } else if (mutator_) {
    return FailedPreconditionError(
        "Mutation and crowding may not yet be used simultaneously on a single "
        "BruteForceSearcher instance.");
  }
  return OkStatus();
}

namespace {

template <typename ResultType>
class TopNWrapperInterface {
 public:
  virtual ~TopNWrapperInterface() {}
  virtual void PushBatch(ConstSpan<ResultType> result_block,
                         size_t base_dp_idx) = 0;
  virtual NNResultsVector TakeUnsorted() = 0;
};

template <typename TopNBase, typename ResultType>
class TopNWrapper final : public TopNWrapperInterface<ResultType> {
 public:
  TopNWrapper(TopNBase base, ResultType epsilon)
      : base_(std::move(base)), epsilon_(epsilon) {}

  void PushBatch(ConstSpan<ResultType> result_block, size_t base_dp_idx) final {
    for (size_t i : IndicesOf(result_block)) {
      if (result_block[i] <= epsilon_) {
        base_.push(std::make_pair(base_dp_idx + i, result_block[i]));
        if (base_.full()) epsilon_ = base_.approx_bottom().second;
      }
    }
  }

  NNResultsVector TakeUnsorted() final { return base_.TakeUnsorted(); }

 private:
  TopNBase base_;
  ResultType epsilon_;
};

template <typename TopNBase, typename ResultType>
class TopNWrapperThreadSafe final : public TopNWrapperInterface<ResultType> {
 public:
  TopNWrapperThreadSafe(TopNBase base, ResultType epsilon)
      : base_(std::move(base)), epsilon_(epsilon) {}

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
      if (dist > eps) continue;
      buf[num_to_push++] = {i + base_dp_idx, dist};
      if (num_to_push == kBufferSize) locked_push_results();
    }
    if (num_to_push) locked_push_results();
  }

  NNResultsVector TakeUnsorted() final { return base_.TakeUnsorted(); }

 private:
  TopNBase base_;
  std::atomic<ResultType> epsilon_;
  mutable absl::Mutex mutex_;
};

template <typename ResultType, typename TopNBase>
unique_ptr<TopNWrapperInterface<ResultType>> MakeTopNWrapper(TopNBase base,
                                                             ResultType epsilon,
                                                             bool thread_safe) {
  if (thread_safe) {
    return unique_ptr<TopNWrapperInterface<ResultType>>(
        new TopNWrapperThreadSafe<TopNBase, ResultType>(std::move(base),
                                                        epsilon));
  } else {
    return unique_ptr<TopNWrapperInterface<ResultType>>(
        new TopNWrapper<TopNBase, ResultType>(std::move(base), epsilon));
  }
}

class FastTopNeighborsWrapper : public TopNWrapperInterface<float> {
 public:
  FastTopNeighborsWrapper(int32_t num_neighbors, float epsilon)
      : fast_top_neighbors_(num_neighbors, epsilon) {
    fast_top_neighbors_.AcquireMutator(&mutator_);
  }

  void PushBatch(ConstSpan<float> result_block, size_t base_dp_idx) final {
    return mutator_.PushDistanceBlock(result_block, base_dp_idx);
  }

  NNResultsVector TakeUnsorted() final {
    mutator_.Release();
    NNResultsVector result;
    fast_top_neighbors_.FinishUnsorted(&result);
    return result;
  }

 private:
  FastTopNeighbors<float> fast_top_neighbors_;
  FastTopNeighborsMutator<float> mutator_;
};

template <typename ResultType>
unique_ptr<TopNWrapperInterface<ResultType>> MakeNonCrowdingTopN(
    const SearchParameters& params, bool thread_safe) {
  if constexpr (IsSame<ResultType, float>()) {
    if (!thread_safe) {
      return make_unique<FastTopNeighborsWrapper>(
          params.pre_reordering_num_neighbors(),
          params.pre_reordering_epsilon());
    }
  }

  return MakeTopNWrapper<ResultType>(
      TopNeighbors<float>(params.pre_reordering_num_neighbors()),
      params.pre_reordering_epsilon(), thread_safe);
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
  vector<unique_ptr<TopNWrapperInterface<Float>>> top_ns(queries.size());
  for (size_t i : IndicesOf(params)) {
    if (params[i].pre_reordering_crowding_enabled()) {
    } else {
      top_ns[i] = MakeNonCrowdingTopN<Float>(params[i], pool_.get());
    }
  }

  auto write_to_top_n = [&](MutableSpan<Float> result_block,
                            DatapointIndex base_dp_idx,
                            DatapointIndex query_idx) {
    auto& top_n = top_ns[query_idx];
    top_n->PushBatch(result_block, base_dp_idx);
  };

  if (squared_db_norms_.empty()) {
    DCHECK(typeid(*distance_) != typeid(SquaredL2Distance));
    DenseDistanceManyToMany<Float>(*distance_, queries, db, pool_.get(),
                                   write_to_top_n);
  } else {
    vector<Float> squared_query_norms;
    squared_query_norms.reserve(queries.size());
    for (auto dptr : queries) {
      squared_query_norms.push_back(SquaredL2Norm(dptr));
    }
    DenseSquaredL2DistanceManyToMany<Float, Float, Float>(
        queries, db, squared_query_norms, squared_db_norms_, pool_.get(),
        write_to_top_n);
  }
  for (size_t i : IndicesOf(top_ns)) {
    results[i] = top_ns[i]->TakeUnsorted();
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
Status BruteForceSearcher<T>::FindNeighborsImpl(const DatapointPtr<T>& query,
                                                const SearchParameters& params,
                                                NNResultsVector* result) const {
  DCHECK(result);
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    TopNeighbors<float> top_n(params.pre_reordering_num_neighbors());
    FindNeighborsInternal(query, params, &top_n);
    *result = top_n.ExtractUnsorted();
  }
  return OkStatus();
}

template <typename T>
template <typename TopN>
void BruteForceSearcher<T>::FindNeighborsInternal(
    const DatapointPtr<T>& query, const SearchParameters& params,
    TopN* top_n_ptr) const {
  DCHECK(top_n_ptr);

  if (query.IsDense() && this->dataset()->IsDense()) {
    TopN top_n = std::move(*top_n_ptr);
    const float epsilon = params.pre_reordering_epsilon();
    float min_keep_distance = epsilon;
    const DenseDataset<T>& dataset =
        *down_cast<const DenseDataset<T>*>(this->dataset());

    if (params.restricts_enabled()) {
    } else {
      unique_ptr<float[]> distances_storage(new float[dataset.size()]);
      MutableSpan<float> distances(distances_storage.get(), dataset.size());
      DenseDistanceOneToMany<T, float>(*distance_, query, dataset, distances);
      for (DatapointIndex i : IndicesOf(dataset)) {
        if (distances[i] <= min_keep_distance) {
          top_n.push(std::make_pair(i, distances[i]));
          if (top_n.full()) {
            min_keep_distance = top_n.approx_bottom().second;
          }
        }
      }
    }
    *top_n_ptr = std::move(top_n);
  } else if (params.restricts_enabled()) {
    auto it = params.restrict_whitelist()->WhitelistedPointIterator();
    FindNeighborsOneToOneInternal(query, params, &it, top_n_ptr);
  } else {
    DummyAllowlist allowlist(this->dataset()->size());
    auto it = allowlist.AllowlistedPointIterator();
    FindNeighborsOneToOneInternal(query, params, &it, top_n_ptr);
  }
}

template <typename T>
template <typename AllowlistIterator, typename TopN>
void BruteForceSearcher<T>::FindNeighborsOneToOneInternal(
    const DatapointPtr<T>& query, const SearchParameters& params,
    AllowlistIterator* allowlist_iterator, TopN* top_n_ptr) const {
  DCHECK(top_n_ptr);

  TopN top_n = std::move(*top_n_ptr);
  const float epsilon = params.pre_reordering_epsilon();
  float min_keep_distance = epsilon;

  if (query.IsDense() && this->dataset()->IsDense()) {
    const DenseDataset<T>& dataset =
        *down_cast<const DenseDataset<T>*>(this->dataset());
    for (; !allowlist_iterator->Done(); allowlist_iterator->Next()) {
      const DatapointIndex i = allowlist_iterator->value();
      const DatapointPtr<T> dptr = dataset[i];
      const double dist = distance_->GetDistanceDense(query, dptr);
      if (dist <= min_keep_distance) {
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
      if (dist <= min_keep_distance) {
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
      if (dist <= min_keep_distance) {
        top_n.push(std::make_pair(i, dist));
        if (top_n.full()) {
          min_keep_distance = top_n.approx_bottom().second;
        }
      }
    }
  }
  *top_n_ptr = std::move(top_n);
}

SCANN_INSTANTIATE_TYPED_CLASS(, BruteForceSearcher);

}  // namespace scann_ops
}  // namespace tensorflow
