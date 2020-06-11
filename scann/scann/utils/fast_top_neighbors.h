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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__UTILS_FAST_TOP_NEIGHBORS_H_
#define SCANN__UTILS_FAST_TOP_NEIGHBORS_H_

#include <string>

#include "absl/numeric/int128.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace tensorflow {
namespace scann_ops {

template <typename DistT, typename DatapointIndexT = DatapointIndex>
class FastTopNeighbors {
 public:
  FastTopNeighbors() {}

  explicit FastTopNeighbors(size_t max_results,
                            DistT epsilon = MaxOrInfinity<DistT>()) {
    Init(max_results, epsilon);
  }

  void Init(size_t max_results, DistT epsilon = MaxOrInfinity<DistT>()) {
    CHECK(!mutator_held_);
    sz_ = 0;
    epsilon_ = epsilon;
    if (max_results_ == max_results) {
      CHECK(indices_);
      return;
    }

    max_results_ = max_results;

    const size_t max_no_realloc_results =
        (epsilon < MaxOrInfinity<DistT>()) ? 128 : 16384;
    if (max_results <= max_no_realloc_results) {
      capacity_ = max_capacity_ = NextMultipleOf(2 * max_results, 32);
    } else {
      capacity_ = 2 * max_no_realloc_results;

      constexpr size_t kMaxPossibleResults =
          (numeric_limits<size_t>::max() ^ size_t(31)) / 2;
      max_capacity_ =
          NextMultipleOf(2 * std::min(kMaxPossibleResults, max_results), 32);
    }

    indices_.reset(new DatapointIndexT[2 * capacity_ + kPadding]);
    distances_.reset(new DistT[capacity_ + kPadding]);
    std::fill(distances_.get(), distances_.get() + capacity_ + kPadding,
              epsilon);
    masks_.reset(new uint32_t[2 * capacity_ / 32 + 2]);
  }

  SCANN_INLINE DistT epsilon() const { return epsilon_; }

  size_t max_results() const { return max_results_; }

  size_t capacity() const { return capacity_; }

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

  DistT epsilon_;
  DatapointIndexT tiebreaker_idx_;

  bool mutator_held_ = false;

  enum : size_t { kPadding = 96 };

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
    SCANN_LOG_NOOP(1) << StrFormat("Pushing {%d, %f}", dp_idx,
                                   static_cast<double>(distance));
    DCHECK_LT(pushes_remaining_negated_, 0);
    indices_end_[pushes_remaining_negated_] = dp_idx;
    distances_end_[pushes_remaining_negated_] = distance;
    ++pushes_remaining_negated_;
    return pushes_remaining_negated_ == 0;
  }

  SCANN_OUTLINE void PushDistanceBlock(ConstSpan<DistT> distance_block,
                                       DatapointIndexT base_dp_idx);

  SCANN_INLINE DistT epsilon() const { return parent_->epsilon_; }

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

template <typename DistT, typename DatapointIndexT, typename Mutator>
void PushDistanceBlockTopFastTopNeighbors(ConstSpan<DistT> distance_block,
                                          DatapointIndexT base_dp_idx,
                                          Mutator* mutator) {
  DatapointIndex distance_block_idx = 0;

  if constexpr (std::is_same_v<DistT, float>) {
#ifdef __SSE4_1__
    M128_4Xfloat sse_epsilon = M128_4Xfloat::Broadcast(mutator->epsilon());
    constexpr size_t kNumFloatsPerSimdRegister = 4;
    const size_t num_sse4_registers =
        distance_block.size() / kNumFloatsPerSimdRegister;
    for (uint32_t simd_idx : Seq(num_sse4_registers)) {
      const uint32_t i0 = simd_idx * kNumFloatsPerSimdRegister;
      M128_4Xfloat simd_dists = M128_4Xfloat::Load(&distance_block[i0]);
      int push_mask = (simd_dists <= sse_epsilon).MaskFromHighBits();
      while (ABSL_PREDICT_FALSE(push_mask)) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        const DatapointIndexT dp_idx = base_dp_idx + i0 + offset;
        if (ABSL_PREDICT_FALSE(
                mutator->Push(dp_idx, simd_dists.val()[offset]))) {
          mutator->GarbageCollect();
          sse_epsilon = M128_4Xfloat::Broadcast(mutator->epsilon());

          push_mask &= (simd_dists < sse_epsilon).MaskFromHighBits();
        }
      }
    }
    distance_block_idx = num_sse4_registers * kNumFloatsPerSimdRegister;
#endif
  }

  DistT eps = mutator->epsilon();
  for (; distance_block_idx < distance_block.size(); ++distance_block_idx) {
    const DistT dist = distance_block[distance_block_idx];
    if (dist < eps) {
      if (mutator->Push(distance_block_idx + base_dp_idx, dist)) {
        mutator->GarbageCollect();
        eps = mutator->epsilon();
      }
    }
  }
}

extern template class FastTopNeighbors<int16_t, DatapointIndex>;
extern template class FastTopNeighbors<float, DatapointIndex>;
extern template class FastTopNeighbors<int16_t, unsigned long long_t>;
extern template class FastTopNeighbors<float, unsigned long long_t>;
extern template class FastTopNeighbors<float, absl::uint128>;

}  // namespace scann_ops
}  // namespace tensorflow

#endif
