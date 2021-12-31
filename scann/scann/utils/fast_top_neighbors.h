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

#ifndef SCANN_UTILS_FAST_TOP_NEIGHBORS_H_
#define SCANN_UTILS_FAST_TOP_NEIGHBORS_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/numeric/int128.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

template <typename DistT, typename DatapointIndexT = DatapointIndex>
class FastTopNeighbors {
 public:
  FastTopNeighbors() {}

  explicit FastTopNeighbors(size_t max_results,
                            DistT epsilon = MaxOrInfinity<DistT>()) {
    Init(max_results, epsilon);
  }

  FastTopNeighbors(FastTopNeighbors&& rhs) { *this = std::move(rhs); }

  FastTopNeighbors& operator=(FastTopNeighbors&& rhs) {
    indices_ = std::move(rhs.indices_);
    distances_ = std::move(rhs.distances_);
    masks_ = std::move(rhs.masks_);
    sz_ = rhs.sz_;
    max_results_ = rhs.max_results_;
    capacity_ = rhs.capacity_;
    max_capacity_ = rhs.max_capacity_;
    epsilon_ = rhs.epsilon_.load(std::memory_order_relaxed);
    tiebreaker_idx_ = rhs.tiebreaker_idx_;
    mutator_held_ = rhs.mutator_held_;
    return *this;
  }

  void Init(size_t max_results, DistT epsilon = MaxOrInfinity<DistT>()) {
    CHECK(!mutator_held_);
    sz_ = 0;
    epsilon_.store(epsilon, std::memory_order_relaxed);
    if (max_results_ == max_results && indices_) {
      return;
    }

    max_results_ = max_results;

    const size_t max_no_realloc_results =
        (epsilon < MaxOrInfinity<DistT>()) ? 128 : 16384;
    if (max_results == 0) {
      capacity_ = 32;
    } else if (max_results <= max_no_realloc_results) {
      capacity_ = max_capacity_ = NextMultipleOf(2 * max_results, 32);
    } else {
      capacity_ = 2 * max_no_realloc_results;

      constexpr size_t kMaxPossibleResults =
          (numeric_limits<size_t>::max() ^ size_t(31)) / 2;
      max_capacity_ =
          NextMultipleOf(2 * std::min(kMaxPossibleResults, max_results), 32);
    }

    AllocateArrays(capacity_);
    FillDistancesForASan();
  }

  void InitWithCapacity(size_t capacity) {
    CHECK(!mutator_held_);
    epsilon_.store(MaxOrInfinity<DistT>(), std::memory_order_relaxed);
    capacity_ = max_capacity_ = capacity;
    AllocateArrays(capacity_);
    FillDistancesForASan();
  }

  SCANN_INLINE DistT epsilon() const {
    return epsilon_.load(std::memory_order_relaxed);
  }

  size_t max_results() const { return max_results_; }

  size_t capacity() const { return capacity_; }

  void PushBlock(ConstSpan<DistT> distances, DatapointIndexT base_dp_idx) {
    PushBlockToFastTopNeighbors(
        distances,
        [base_dp_idx](DatapointIndex offset) { return base_dp_idx + offset; },
        this);
  }

  template <typename LocalDistT>
  void PushBlock(ConstSpan<DistT> distances,
                 ConstSpan<LocalDistT> local_dp_indices,
                 DatapointIndexT base_dp_idx) {
    PushBlockToFastTopNeighbors(
        distances,
        [&](DatapointIndex offset) {
          return base_dp_idx + local_dp_indices[offset];
        },
        this);
  }

  class Mutator;
  SCANN_INLINE void AcquireMutator(Mutator* mutator);

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted(size_t max_results) {
    CHECK(!mutator_held_);
    GarbageCollect(max_results, max_results);
    auto indices = MutableSpan<DatapointIndexT>(indices_.get(), sz_);
    auto dists = MutableSpan<DistT>(distances_.get(), sz_);
    return std::make_pair(indices, dists);
  }

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted() {
    return FinishUnsorted(max_results_);
  }

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  GetRawStorage(size_t set_size_to) {
    CHECK(set_size_to <= capacity_);
    sz_ = set_size_to;
    auto indices = MutableSpan<DatapointIndexT>(indices_.get(), sz_);
    auto dists = MutableSpan<DistT>(distances_.get(), sz_);
    return std::make_pair(indices, dists);
  }

  pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>> FinishSorted();

  void FinishUnsorted(std::vector<pair<DatapointIndexT, DistT>>* results) {
    ConstSpan<DatapointIndexT> idxs;
    ConstSpan<DistT> dists;
    std::tie(idxs, dists) = FinishUnsorted();
    DCHECK_EQ(idxs.size(), dists.size());

    results->resize(idxs.size());
    auto* rr = results->data();
    for (size_t j : Seq(idxs.size())) {
      rr[j] = std::make_pair(idxs[j], dists[j]);
    }
  }

  void FinishSorted(std::vector<pair<DatapointIndexT, DistT>>* results) {
    FinishUnsorted(results);
    SortBranchOptimized(results->begin(), results->end(),
                        DistanceComparatorBranchOptimized());
  }

 private:
  void GarbageCollect(size_t keep_min, size_t keep_max);

  SCANN_INLINE void GarbageCollectApproximate() {
    if (capacity_ < max_capacity_) {
      return ReallocateForPureEnn();
    }
    const size_t keep_max = (max_results_ + capacity_) / 2 - 1;
    GarbageCollect(max_results_, keep_max);
  }

  void AllocateArrays(size_t capacity);
  void FillDistancesForASan();
  void ReallocateForPureEnn();

  SCANN_INLINE void ReleaseMutator(ssize_t pushes_remaining_negated) {
    mutator_held_ = false;
    sz_ = pushes_remaining_negated + capacity_;
  }

  static size_t ApproxNthElement(size_t keep_min, size_t keep_max, size_t sz,
                                 DatapointIndexT* ii, DistT* dd, uint32_t* mm);

  unique_ptr<DatapointIndexT[]> indices_;

  unique_ptr<DistT[]> distances_;

  unique_ptr<uint32_t[]> masks_;

  size_t sz_ = 0;

  size_t max_results_ = 0;

  size_t capacity_ = 0;

  size_t max_capacity_ = 0;

  std::atomic<DistT> epsilon_ = MaxOrInfinity<DistT>();
  DatapointIndexT tiebreaker_idx_ = kInvalidDatapointIndex;

  bool mutator_held_ = false;

  friend class Mutator;
  friend class FastTopNeighborsTest;
};

template <typename DistT, typename DatapointIndexT>
class FastTopNeighbors<DistT, DatapointIndexT>::Mutator {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(Mutator);

  Mutator() {}

  void Release() {
    if (parent_) {
      parent_->ReleaseMutator(pushes_remaining_negated_);
      parent_ = nullptr;
    }
  }

  ~Mutator() { Release(); }

  SCANN_INLINE bool Push(DatapointIndexT dp_idx, DistT distance) {
    DCHECK_LE(distance, epsilon());
    DCHECK(!std::isnan(distance));
    SCANN_LOG_NOOP(1) << StrFormat("Pushing {%d, %f}", dp_idx,
                                   static_cast<double>(distance));
    DCHECK_LT(pushes_remaining_negated_, 0);
    indices_end_[pushes_remaining_negated_] = dp_idx;
    distances_end_[pushes_remaining_negated_] = distance;
    ++pushes_remaining_negated_;
    return pushes_remaining_negated_ == 0;
  }

  SCANN_INLINE DistT epsilon() const {
    return parent_->epsilon_.load(std::memory_order_relaxed);
  }

  SCANN_INLINE void GarbageCollect() {
    parent_->sz_ = parent_->capacity_ + pushes_remaining_negated_;

    parent_->GarbageCollectApproximate();

    InitImpl();
  }

 private:
  SCANN_INLINE void Init(FastTopNeighbors* parent) {
    DCHECK(!parent_);
    parent_ = parent;
    InitImpl();
  }

  SCANN_INLINE void InitImpl() {
    DCHECK(parent_);
    indices_end_ = parent_->indices_.get() + parent_->capacity_;
    distances_end_ = parent_->distances_.get() + parent_->capacity_;
    pushes_remaining_negated_ = parent_->sz_ - parent_->capacity_;
  }

  FastTopNeighbors* parent_ = nullptr;

  DatapointIndexT* indices_end_;

  DistT* distances_end_;

  ssize_t pushes_remaining_negated_;

  friend class FastTopNeighbors;
};

template <typename DistT, typename DatapointIndexT = DatapointIndex>
using FastTopNeighborsMutator =
    typename FastTopNeighbors<DistT, DatapointIndexT>::Mutator;

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::AcquireMutator(
    Mutator* mutator) {
  DCHECK(!mutator_held_);
  mutator_held_ = true;
  return mutator->Init(this);
}

template <typename DistT, typename DocidFn, typename TopN>
void PushBlockToFastTopNeighbors(ConstSpan<DistT> distances, DocidFn docid_fn,
                                 TopN* top_n) {
  typename TopN::Mutator mutator;
  top_n->AcquireMutator(&mutator);
  DatapointIndex dist_idx = 0;

#ifdef __SSE4_1__
  if constexpr (std::is_same_v<DistT, float>) {
    Sse4<float> sse_epsilon = mutator.epsilon();
    constexpr size_t kNumFloatsPerSimdRegister =
        Sse4<float>::kElementsPerRegister;
    const size_t num_sse4_registers =
        distances.size() / kNumFloatsPerSimdRegister;
    for (uint32_t simd_idx : Seq(num_sse4_registers)) {
      const uint32_t i0 = simd_idx * kNumFloatsPerSimdRegister;
      Sse4<float> simd_dists = Sse4<float>::Load(&distances[i0]);
      int push_mask = GetComparisonMask(simd_dists <= sse_epsilon);
      while (ABSL_PREDICT_FALSE(push_mask)) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        if (ABSL_PREDICT_FALSE(
                mutator.Push(docid_fn(i0 + offset), (*simd_dists)[offset]))) {
          mutator.GarbageCollect();
          sse_epsilon = mutator.epsilon();

          push_mask &= GetComparisonMask(simd_dists < sse_epsilon);
        }
      }
    }
    dist_idx = num_sse4_registers * kNumFloatsPerSimdRegister;
  }
#endif

  DistT eps = mutator.epsilon();
  for (; dist_idx < distances.size(); ++dist_idx) {
    const DistT dist = distances[dist_idx];
    if (dist < eps) {
      if (ABSL_PREDICT_FALSE(mutator.Push(docid_fn(dist_idx), dist))) {
        mutator.GarbageCollect();
        eps = mutator.epsilon();
      }
    }
  }
}

extern template class FastTopNeighbors<int16_t, DatapointIndex>;
extern template class FastTopNeighbors<float, DatapointIndex>;
extern template class FastTopNeighbors<int16_t, uint64_t>;
extern template class FastTopNeighbors<float, uint64_t>;
extern template class FastTopNeighbors<float, absl::uint128>;

}  // namespace research_scann

#endif
