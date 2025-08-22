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

#ifndef SCANN_UTILS_FAST_TOP_NEIGHBORS_H_
#define SCANN_UTILS_FAST_TOP_NEIGHBORS_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/numeric/int128.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/kv_sort.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

template <typename DistT, typename DatapointIndexT = DatapointIndex,
          size_t FixedCapacity = 0>
class FastTopNeighbors {
 public:
  FastTopNeighbors() = default;

  explicit FastTopNeighbors(size_t max_results,
                            DistT epsilon = MaxOrInfinity<DistT>()) {
    Init(max_results, epsilon);
  }

  FastTopNeighbors(FastTopNeighbors&& rhs) noexcept { *this = std::move(rhs); }

  StatusOr<FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>> Clone()
      const;

  FastTopNeighbors& operator=(FastTopNeighbors&& rhs) noexcept {
    if (rhs.dynamic_indices_ == nullptr) {
      std::copy(rhs.fixed_indices_, rhs.fixed_indices_ + rhs.sz_,
                fixed_indices_);
      indices_ = fixed_indices_;
    } else {
      dynamic_indices_ = std::move(rhs.dynamic_indices_);
      indices_ = dynamic_indices_.get();
    }
    if (rhs.dynamic_distances_ == nullptr) {
      std::copy(rhs.fixed_distances_, rhs.fixed_distances_ + rhs.sz_,
                fixed_distances_);
      distances_ = fixed_distances_;
    } else {
      dynamic_distances_ = std::move(rhs.dynamic_distances_);
      distances_ = dynamic_distances_.get();
    }
    if (rhs.dynamic_masks_ == nullptr) {
      masks_ = fixed_masks_;
    } else {
      dynamic_masks_ = std::move(rhs.dynamic_masks_);
      masks_ = dynamic_masks_.get();
    }

    sz_ = rhs.sz_;
    max_results_ = rhs.max_results_;
    capacity_ = rhs.capacity_;
    max_capacity_ = rhs.max_capacity_;
    epsilon_ = rhs.epsilon_.load(std::memory_order_relaxed);
    mutator_held_ = rhs.mutator_held_;
    return *this;
  }

  void Init(size_t max_results, DistT epsilon = MaxOrInfinity<DistT>()) {
    CHECK(!mutator_held_);
    sz_ = 0;
    epsilon_.store(epsilon, std::memory_order_relaxed);
    if (max_results_ >= max_results && indices_) {
      max_results_ = max_results;
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
    FillDistancesForMSan();
  }

  void InitWithCapacity(size_t capacity) {
    CHECK(!mutator_held_);
    epsilon_.store(MaxOrInfinity<DistT>(), std::memory_order_relaxed);
    capacity_ = max_capacity_ = capacity;
    AllocateArrays(capacity_);
    FillDistancesForMSan();
  }

  SCANN_INLINE DistT epsilon() const {
    return epsilon_.load(std::memory_order_relaxed);
  }

  size_t max_results() const { return max_results_; }

  size_t capacity() const { return capacity_; }

  void PushBlock(ConstSpan<DistT> distances, DatapointIndexT base_dp_idx) {
    PushBlockToFastTopNeighbors(
        distances,
        [base_dp_idx](DatapointIndex offset) {
          if constexpr (std::is_same_v<DatapointIndexT,
                                       pair<uint64_t, uint64_t>>) {
            return std::make_pair(base_dp_idx.first + offset,
                                  base_dp_idx.second);
          } else if constexpr (std::is_same_v<DatapointIndexT,
                                              std::shared_ptr<std::string>>) {
            return std::make_shared<std::string>();
          } else {
            return base_dp_idx + offset;
          }
        },
        this);
  }

  template <typename LocalDistT>
  void PushBlock(ConstSpan<DistT> distances,
                 ConstSpan<LocalDistT> local_dp_indices,
                 DatapointIndexT base_dp_idx) {
    PushBlockToFastTopNeighbors(
        distances,
        [&](DatapointIndex offset) {
          if constexpr (std::is_same_v<DatapointIndexT,
                                       pair<int64_t, uintptr_t>>) {
            return std::make_pair(base_dp_idx.first + local_dp_indices[offset],
                                  base_dp_idx.second);
          } else {
            return base_dp_idx + local_dp_indices[offset];
          }
        },
        this);
  }

  class Mutator;
  SCANN_INLINE void AcquireMutator(Mutator* mutator);

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted(size_t max_results) {
    CHECK(!mutator_held_);
    GarbageCollect(max_results, max_results);
    auto indices = MutableSpan<DatapointIndexT>(indices_, sz_);
    auto dists = MutableSpan<DistT>(distances_, sz_);
    return std::make_pair(indices, dists);
  }

  SCANN_INLINE void resize(size_t new_sz) {
    DCHECK_LE(new_sz, sz_);
    sz_ = new_sz;
  }

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted() {
    return FinishUnsorted(max_results_);
  }

  SCANN_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  GetRawStorage(size_t set_size_to) {
    CHECK(set_size_to <= capacity_);
    sz_ = set_size_to;
    auto indices = MutableSpan<DatapointIndexT>(indices_, sz_);
    auto dists = MutableSpan<DistT>(distances_, sz_);
    return std::make_pair(indices, dists);
  }

  ConstSpan<DatapointIndexT> GetAllRawIndices() const {
    return MutableSpan<DatapointIndexT>(indices_, sz_);
  }

  ConstSpan<DistT> GetAllRawDistances() const {
    return MutableSpan<DistT>(distances_, sz_);
  }

  pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>> FinishSorted() {
    return FinishSorted(max_results_);
  }
  pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>> FinishSorted(
      size_t max_results);

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

  void GarbageCollect(size_t keep_min, size_t keep_max);

  void MoveTopNToFront(size_t num_elements);

 protected:
  DatapointIndexT* indices_ = nullptr;

  DistT* distances_ = nullptr;

  size_t sz_ = 0;

  bool mutator_held_ = false;

 private:
  SCANN_INLINE void GarbageCollectApproximate() {
    if (capacity_ < max_capacity_) {
      return ReallocateForPureEnn();
    }
    const size_t keep_max = (max_results_ + capacity_) / 2 - 1;
    GarbageCollect(max_results_, keep_max);
  }

  void AllocateArrays(size_t capacity);
  void FillDistancesForMSan();
  void ReallocateForPureEnn();

  static void SortKVInPlace(MutableSpan<DistT> keys,
                            MutableSpan<DatapointIndexT> vals);

  SCANN_INLINE void ReleaseMutator(ssize_t pushes_remaining_negated) {
    mutator_held_ = false;
    sz_ = pushes_remaining_negated + capacity_;
  }

  uint32_t* masks_ = nullptr;

  size_t max_results_ = 0;

  size_t capacity_ = 0;

  size_t max_capacity_ = 0;

  std::atomic<DistT> epsilon_ = MaxOrInfinity<DistT>();

  constexpr static size_t kPadding = 96;
  constexpr static size_t AlignedFixedCapacity() {
    return NextMultipleOf(2 * FixedCapacity, 32);
  }

  unique_ptr<DatapointIndexT[]> dynamic_indices_;
  unique_ptr<DistT[]> dynamic_distances_;
  unique_ptr<uint32_t[]> dynamic_masks_;
  DatapointIndexT fixed_indices_[FixedCapacity == 0
                                     ? 0
                                     : 2 * AlignedFixedCapacity() + kPadding];
  DistT fixed_distances_[FixedCapacity == 0
                             ? 0
                             : AlignedFixedCapacity() + kPadding];
  uint32_t
      fixed_masks_[FixedCapacity == 0 ? 0 : AlignedFixedCapacity() / 32 + 2];

  friend class Mutator;
  friend class FastTopNeighborsTest;
};

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
class FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::Mutator {
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
    return PushNoEpsilonCheck(dp_idx, distance);
  }

  SCANN_INLINE bool PushNoEpsilonCheck(DatapointIndexT dp_idx, DistT distance) {
    DCHECK(!std::isnan(distance));
    if constexpr (std::is_same_v<DatapointIndexT, pair<uint64_t, uint64_t>>) {
      DVLOG(1) << StrFormat("Pushing {%d, %f}", dp_idx.first,
                            static_cast<double>(distance));
    } else if constexpr (std::is_same_v<DatapointIndexT,
                                        std::shared_ptr<std::string>>) {
      DVLOG(1) << StrFormat("Pushing {%d, %f}",
                            reinterpret_cast<uint64_t>(dp_idx.get()),
                            static_cast<double>(distance));
    } else {
      DVLOG(1) << StrFormat("Pushing {%d, %f}", dp_idx,
                            static_cast<double>(distance));
    }
    DCHECK_LE(pushes_remaining_negated_, 0);
    indices_end_[pushes_remaining_negated_] = dp_idx;
    distances_end_[pushes_remaining_negated_] = distance;
    ++pushes_remaining_negated_;
    return pushes_remaining_negated_ == 0;
  }

  SCANN_INLINE DistT epsilon() const {
    return parent_->epsilon_.load(std::memory_order_relaxed);
  }

  size_t max_results() const { return parent_->max_results_; }

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
    indices_end_ = parent_->indices_ + parent_->capacity_;
    distances_end_ = parent_->distances_ + parent_->capacity_;
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

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::AcquireMutator(
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

#if HWY_HAVE_CONSTEXPR_LANES

  if constexpr (std::is_same_v<DistT, float> &&
                highway::Simd<float>::kElementsPerRegister >= 2) {
    using Simd = highway::Simd<float>;
    Simd simd_epsilon = mutator.epsilon();
    constexpr size_t kNumFloatsPerSimdRegister = Simd::kElementsPerRegister;
    const size_t num_simd_registers =
        distances.size() / kNumFloatsPerSimdRegister;
    for (uint32_t simd_idx : Seq(num_simd_registers)) {
      const uint32_t i0 = simd_idx * kNumFloatsPerSimdRegister;
      Simd simd_dists = Simd::Load(&distances[i0]);
      int push_mask = GetComparisonMask(simd_dists <= simd_epsilon);
      while (ABSL_PREDICT_FALSE(push_mask)) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        if (ABSL_PREDICT_FALSE(mutator.Push(docid_fn(i0 + offset),
                                            simd_dists.ExtractLane(offset)))) {
          mutator.GarbageCollect();
          simd_epsilon = mutator.epsilon();

          push_mask &= GetComparisonMask(simd_dists < simd_epsilon);
        }
      }
    }
    dist_idx = num_simd_registers * kNumFloatsPerSimdRegister;
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

template <typename DistT, typename DatapointIndexT>
size_t ApproxNthElement(size_t keep_min, size_t keep_max, size_t sz,
                        DatapointIndexT* ii, DistT* dd, uint32_t* mm);

template <typename DistT, typename DatapointIndexT>
std::string FTN_DebugLogArrayContents(DatapointIndexT* indices, DistT* values,
                                      uint32_t* masks, size_t sz);

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
StatusOr<FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>>
FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::Clone() const {
  SCANN_RET_CHECK(!mutator_held_)
      << "FastTopNeighbors::Clone called on a TopN with a mutator held.";
  FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity> result;
  result.AllocateArrays(capacity_);
  result.FillDistancesForMSan();
  result.sz_ = sz_;
  result.max_capacity_ = max_capacity_;
  result.max_results_ = max_results_;
  std::copy(indices_, indices_ + sz_, result.indices_);
  std::copy(distances_, distances_ + sz_, result.distances_);
  result.epsilon_.store(epsilon_.load());
  return result;
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::AllocateArrays(
    size_t capacity) {
  capacity_ = capacity;
  size_t indices_capacity = 2 * capacity_ + kPadding;
  size_t distances_capacity = capacity_ + kPadding;
  size_t mask_capacity = 2 * capacity_ / 32 + 2;
  indices_ = fixed_indices_;
  distances_ = fixed_distances_;
  masks_ = fixed_masks_;
  if (sizeof(DatapointIndexT) * indices_capacity > sizeof(fixed_indices_)) {
    dynamic_indices_.reset(new DatapointIndexT[indices_capacity]);
    indices_ = dynamic_indices_.get();
  }
  if (sizeof(DistT) * distances_capacity > sizeof(fixed_distances_)) {
    dynamic_distances_.reset(new DistT[distances_capacity]);
    distances_ = dynamic_distances_.get();
  }
  if (sizeof(uint32_t) * mask_capacity > sizeof(fixed_masks_)) {
    dynamic_masks_.reset(new uint32_t[mask_capacity]);
    masks_ = dynamic_masks_.get();
  }
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT,
                      FixedCapacity>::FillDistancesForMSan() {
#ifdef MEMORY_SANITIZER

  DistT* start = distances_ + sz_;
  DistT* end = distances_ + capacity_ + kPadding;
  const size_t len = (end - start) * sizeof(DistT);
  __msan_unpoison(start, len);
#endif
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT,
                      FixedCapacity>::ReallocateForPureEnn() {
  if (sz_ < capacity_) return;

  unique_ptr<DatapointIndexT[]> old_indices = std::move(dynamic_indices_);
  unique_ptr<DistT[]> old_distances = std::move(dynamic_distances_);

  AllocateArrays(std::min(capacity_ * 2, max_capacity_));

  if (old_indices == nullptr) {
    if (indices_ != fixed_indices_) {
      std::copy(fixed_indices_, fixed_indices_ + sz_, indices_);
    }
  } else {
    std::copy(old_indices.get(), old_indices.get() + sz_, indices_);
  }
  if (old_distances == nullptr) {
    if (distances_ != fixed_distances_) {
      std::copy(fixed_distances_, fixed_distances_ + sz_, distances_);
    }
  } else {
    std::copy(old_distances.get(), old_distances.get() + sz_, distances_);
  }
  FillDistancesForMSan();
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::GarbageCollect(
    size_t keep_min, size_t keep_max) {
  constexpr bool kShouldLog = false;
  DCHECK_LE(keep_min, keep_max);
  if (keep_min == 0) {
    sz_ = 0;
    return;
  }
  if (sz_ <= keep_max) return;
  sz_ = ApproxNthElement(keep_min, keep_max, sz_, indices_, distances_, masks_);
  const DistT old_epsilon = epsilon_;
  epsilon_ = distances_[sz_];
  DLOG_IF(INFO, kShouldLog)
      << FTN_DebugLogArrayContents(indices_, distances_, nullptr, sz_);
  DLOG_IF(INFO, kShouldLog) << StrFormat("Threshold change: %f => %f (sz = %d)",
                                         static_cast<double>(old_epsilon),
                                         static_cast<double>(epsilon_), sz_);
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::MoveTopNToFront(
    size_t num_elements) {
  if (num_elements >= sz_) return;

  ZipNthElementBranchOptimized(std::less<DistT>(), num_elements, distances_,
                               distances_ + sz_, indices_, indices_ + sz_);
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
void FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::SortKVInPlace(
    MutableSpan<DistT> keys, MutableSpan<DatapointIndexT> vals) {
  if constexpr (std::is_same_v<DistT, float> &&
                std::is_same_v<DatapointIndexT, uint32_t>) {
    KVSort(keys, vals, KVSortOrder::kAscending);
    return;
  };
  ZipSortBranchOptimized(keys.begin(), keys.end(), vals.begin(), vals.end());
}

template <typename DistT, typename DatapointIndexT, size_t FixedCapacity>
pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
FastTopNeighbors<DistT, DatapointIndexT, FixedCapacity>::FinishSorted(
    size_t max_results) {
  MutableSpan<DatapointIndexT> ii;
  MutableSpan<DistT> vv;
  std::tie(ii, vv) = FinishUnsorted(max_results);
  SortKVInPlace(vv, ii);
  return {ii, vv};
}

#define SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(EXTERN_OR, DistT,           \
                                             DatapointIndexT)            \
  EXTERN_OR template class FastTopNeighbors<DistT, DatapointIndexT>;     \
  EXTERN_OR template size_t ApproxNthElement<DistT, DatapointIndexT>(    \
      size_t keep_min, size_t keep_max, size_t sz, DatapointIndexT * ii, \
      DistT * dd, uint32_t * mm);                                        \
  EXTERN_OR std::string FTN_DebugLogArrayContents(                       \
      DatapointIndexT* indices, DistT* values, uint32_t* masks, size_t sz);

SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, int16_t, uint32_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, float, uint32_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, int16_t, uint64_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, float, uint64_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, int16_t, absl::uint128);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, float, absl::uint128);

using VectorDBDatapointIndexT = pair<uint64_t, uint64_t>;
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, float, VectorDBDatapointIndexT);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(extern, float,
                                     std::shared_ptr<std::string>);

static_assert(std::is_same_v<uint32_t, DatapointIndex> ||
              std::is_same_v<uint64_t, DatapointIndex>);

}  // namespace research_scann

#endif
