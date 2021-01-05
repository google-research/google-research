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

#include "scann/utils/fast_top_neighbors.h"

#include "absl/strings/str_cat.h"
#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/avx2.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace tensorflow {
namespace scann_ops {
namespace {

constexpr bool kShouldLog = false;

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE std::string DebugLogArrayContents(DatapointIndexT* indices,
                                               DistT* values, uint32_t* masks,
                                               size_t sz) {
  std::string txt;
  for (size_t j : Seq(DivRoundUp(sz, 32))) {
    absl::StrAppend(&txt, j ? " || " : "");
    uint32_t mask = masks ? masks[j] : 0;
    string_view sep = "";
    for (size_t offset : Seq(32)) {
      const size_t idx = 32 * j + offset;
      const bool idx_is_masked = mask & 1;
      mask >>= 1;
      if (idx == sz) {
        sep = ".  END: ";
      }
      if (idx_is_masked) {
        StrAppendFormat(&txt, "%s%d:[%d, %s]", sep, idx, indices[idx],
                        absl::StrCat(values[idx]));
      } else {
        StrAppendFormat(&txt, "%s%d:{%d, %s}", sep, idx, indices[idx],
                        absl::StrCat(values[idx]));
      }
      sep = ", ";
      if (idx == sz) {
        if (mask) {
          StrAppendFormat(&txt, ". EXTRA_MASK_BITS=%d", mask);
        }
        break;
      }
    }
  }
  return StrFormat("Array Contents: %s. sz=%d", txt, sz);
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE bool CompIV(DatapointIndexT idx_a, DatapointIndexT idx_b,
                         DistT value_a, DistT value_b) {
  const bool is_eq_or_nan =
      value_a == value_b || std::isunordered(value_a, value_b);
  if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
    return idx_a < idx_b;
  }
  return value_a < value_b;
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void ZipSwap(size_t a, size_t b, DatapointIndexT* indices,
                          DistT* values) {
  std::swap(indices[a], indices[b]);
  std::swap(values[a], values[b]);
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void CompOrSwap(size_t a, size_t b, DatapointIndexT* indices,
                             DistT* values) {
  if (!CompIV(indices[a], indices[b], values[a], values[b])) {
    ZipSwap(a, b, indices, values);
  }
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void SelectionSort(DatapointIndexT* indices, DistT* values,
                                size_t sz) {
  DCHECK_LE(sz, 3);
  switch (sz) {
    case 3:

      CompOrSwap(0, 1, indices, values);
      CompOrSwap(1, 2, indices, values);
      ABSL_FALLTHROUGH_INTENDED;
    case 2:

      CompOrSwap(0, 1, indices, values);
      ABSL_FALLTHROUGH_INTENDED;
    case 1:

      break;
  }
}

template <typename DistT>
SCANN_INLINE DistT FastMedianOf3(DistT v0, DistT v1, DistT v2) {
  DistT big = std::max(v0, v1);
  DistT sml = std::min(v0, v1);
  return std::max(sml, std::min(big, v2));
}

}  // namespace

#ifdef __x86_64__

#define SCANN_SIMD_INLINE SCANN_SIMD_ATTRIBUTE SCANN_INLINE
#define SCANN_SIMD_INLINE_LAMBDA SCANN_SIMD_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_SIMD_OUTLINE SCANN_SIMD_ATTRIBUTE SCANN_OUTLINE

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2_ATTRIBUTE
#include "scann/utils/fast_top_neighbors_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE SCANN_SSE4_ATTRIBUTE
#include "scann/utils/fast_top_neighbors_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4

#endif

template <typename DistT, typename DatapointIndexT>
size_t FastTopNeighbors<DistT, DatapointIndexT>::ApproxNthElement(
    size_t keep_min, size_t keep_max, size_t sz, DatapointIndexT* ii, DistT* dd,
    uint32_t* mm) {
  DCHECK_GT(keep_min, 0);
#ifdef __x86_64__
  if (RuntimeSupportsAvx2()) {
    return avx2::ApproxNthElementImpl(keep_min, keep_max, sz, ii, dd, mm);
  } else if (RuntimeSupportsSse4()) {
    return sse4::ApproxNthElementImpl(keep_min, keep_max, sz, ii, dd, mm);
  }
#endif

  SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
      "ZipNthElementBranchOptimized from %d => %d elements", sz, keep_min);
  ZipNthElementBranchOptimized(std::less<DistT>(), keep_min - 1, dd, dd + sz,
                               ii, ii + sz);

  dd[keep_min] = dd[keep_min - 1];
  ii[keep_min] = ii[keep_min - 1];
  return keep_min;
}

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::Mutator::PushDistanceBlock(
    ConstSpan<DistT> distance_block, DatapointIndexT base_dp_idx) {
  PushDistanceBlockTopFastTopNeighbors<
      DistT, DatapointIndexT,
      FastTopNeighbors<DistT, DatapointIndexT>::Mutator>(distance_block,
                                                         base_dp_idx, this);
}

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::AllocateArrays(size_t capacity) {
  constexpr size_t kPadding = 96;

  capacity_ = capacity;
  indices_.reset(new DatapointIndexT[2 * capacity_ + kPadding]);
  distances_.reset(new DistT[capacity_ + kPadding]);
  masks_.reset(new uint32_t[2 * capacity_ / 32 + 2]);
}

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::FillDistancesForASan() {
  constexpr size_t kPadding = 96;
  std::fill(distances_.get() + sz_, distances_.get() + capacity_ + kPadding,
            epsilon_);
}

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::ReallocateForPureEnn() {
  if (sz_ < capacity_) return;

  unique_ptr<DatapointIndexT[]> old_indices = std::move(indices_);
  unique_ptr<DistT[]> old_distances = std::move(distances_);

  AllocateArrays(std::min(capacity_ * 2, max_capacity_));

  std::copy(old_indices.get(), old_indices.get() + sz_, indices_.get());
  std::copy(old_distances.get(), old_distances.get() + sz_, distances_.get());
  FillDistancesForASan();
}

template <typename DistT, typename DatapointIndexT>
void FastTopNeighbors<DistT, DatapointIndexT>::GarbageCollect(size_t keep_min,
                                                              size_t keep_max) {
  DCHECK_LE(keep_min, keep_max);
  if (keep_min == 0) {
    sz_ = 0;
    return;
  }
  if (sz_ <= keep_max) return;
  sz_ = ApproxNthElement(keep_min, keep_max, sz_, indices_.get(),
                         distances_.get(), masks_.get());
  const DistT old_epsilon = epsilon_;
  epsilon_ = distances_[sz_];
  SCANN_LOG_NOOP(INFO, kShouldLog)
      << DebugLogArrayContents(indices_.get(), distances_.get(), nullptr, sz_);
  SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
      "Threshold change: %f => %f (sz = %d)", static_cast<double>(old_epsilon),
      static_cast<double>(epsilon_), sz_);
}

template <typename DistT, typename DatapointIndexT>
pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
FastTopNeighbors<DistT, DatapointIndexT>::FinishSorted() {
  MutableSpan<DatapointIndexT> ii;
  MutableSpan<DistT> vv;
  std::tie(ii, vv) = FinishUnsorted();

  ZipSortBranchOptimized(vv.begin(), vv.end(), ii.begin(), ii.end());

  return {ii, vv};
}

template class FastTopNeighbors<int16_t, DatapointIndex>;
template class FastTopNeighbors<float, DatapointIndex>;
template class FastTopNeighbors<int16_t, uint64_t>;
template class FastTopNeighbors<float, uint64_t>;
template class FastTopNeighbors<float, absl::uint128>;

}  // namespace scann_ops
}  // namespace tensorflow
