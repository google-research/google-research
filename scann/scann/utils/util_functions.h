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



#ifndef SCANN_UTILS_UTIL_FUNCTIONS_H_
#define SCANN_UTILS_UTIL_FUNCTIONS_H_

#include <cmath>
#include <cstdint>
#include <stack>
#include <unordered_map>

#include "absl/base/internal/sysinfo.h"
#include "google/protobuf/repeated_field.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/features.pb.h"
#include "scann/data_format/internal/short_string_optimized_string.h"
#include "scann/proto/input_output.pb.h"
#include "scann/proto/results.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

namespace research_scann {

using std::shared_ptr;

template <typename T>
AccumulatorTypeFor<T> Sum(ConstSpan<T> terms);

template <typename T>
AccumulatorTypeFor<T> ParallelSum(ConstSpan<T> terms, ThreadPool* pool);

float MaxAbsValue(ConstSpan<float> arr);

template <typename T>
double Median(ConstSpan<T> nums);

template <typename Collection>
SCANN_INLINE double Median(const Collection& c) {
  return Median(MakeConstSpan(c));
}

template <typename T>
double MedianDestructive(MutableSpan<T> nums);

template <typename Float>
Float InverseErf(Float x);

template <typename T>
void PermuteInPlace(MutableSpan<T> a, MutableSpan<uint32_t> perm);

template <typename T>
void PermuteInPlaceInverse(MutableSpan<T> a, MutableSpan<uint32_t> perm);

template <typename Uint>
void InvertPermutationInPlace(MutableSpan<Uint> perm);

template <typename Uint>
void InvertPermutationOutOfPlace(ConstSpan<Uint> perm,
                                 MutableSpan<Uint> result);

template <typename T, typename U>
inline void PointwiseAdd(T* __restrict__ a, const U* __restrict__ b,
                         size_t size);

template <typename F>
inline F ZeroMinSqrt(F x) {
  return (x < 0.0) ? 0.0 : std::sqrt(x);
}

template <typename Iterator, typename Comparator>
void SiftFrontDown(Iterator begin, Iterator end, Comparator cmp);

class DistanceComparator {
 public:
  template <typename DistanceT, typename DatapointIndexT = DatapointIndex>
  bool operator()(const pair<DatapointIndexT, DistanceT>& a,
                  const pair<DatapointIndexT, DistanceT>& b) const {
    if (IsFloatingType<DistanceT>()) {
      DCHECK(!std::isnan(a.second));
      DCHECK(!std::isnan(b.second));
    }

    if (a.second < b.second) return true;
    if (a.second > b.second) return false;
    return a.first < b.first;
  }

  template <typename DistanceT, typename DatapointIndexT = DatapointIndex>
  static bool CompareBranchOptimized(
      const pair<DatapointIndexT, DistanceT>& a,
      const pair<DatapointIndexT, DistanceT>& b) {
    if (IsFloatingType<DistanceT>()) {
      DCHECK(!std::isnan(a.second));
      DCHECK(!std::isnan(b.second));
    }

    const bool is_eq_or_nan =
        (a.second == b.second ||
         (IsFloatingType<DistanceT>() && std::isunordered(a.second, b.second)));

    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return a.first < b.first;
    }
    return a.second < b.second;
  }

  bool operator()(const NearestNeighbors::Neighbor& a,
                  const NearestNeighbors::Neighbor& b) const {
    if (a.distance() < b.distance()) return true;
    if (a.distance() > b.distance()) return false;
    return a.docid() < b.docid();
  }
};

class DistanceComparatorBranchOptimized {
 public:
  template <typename DistanceT, typename DatapointIndexT>
  bool operator()(const pair<DatapointIndexT, DistanceT>& a,
                  const pair<DatapointIndexT, DistanceT>& b) const {
    return DistanceComparator::CompareBranchOptimized(a, b);
  }
};

void RemoveNeighborsPastLimit(DatapointIndex num_neighbors,
                              NNResultsVector* result) ABSL_ATTRIBUTE_NOINLINE;

NearestNeighbors MergeNeighborLists(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors);

NearestNeighbors MergeNeighborListsWithCrowding(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors,
    int per_crowding_attribute_num_neighbors);

void MergeNeighborListsSwap(MutableSpan<NearestNeighbors*> neighbor_lists,
                            int num_neighbors, NearestNeighbors* result);

void MergeNeighborListsWithCrowdingSwap(
    MutableSpan<NearestNeighbors*> neighbor_lists, int num_neighbors,
    int per_crowding_attribute_num_neighbors, NearestNeighbors* result);

NearestNeighbors MergeNeighborListsRemoveDuplicateDocids(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors);

template <typename T>
inline const google::protobuf::RepeatedField<T>& GfvValues(
    const GenericFeatureVector& gfv) {
  LOG(FATAL) << "Invalid GFV values type.";
}

template <>
inline const google::protobuf::RepeatedField<int64_t>& GfvValues<int64_t>(
    const GenericFeatureVector& gfv) {
  return gfv.feature_value_int64();
}

template <>
inline const google::protobuf::RepeatedField<float>& GfvValues<float>(
    const GenericFeatureVector& gfv) {
  return gfv.feature_value_float();
}

template <>
inline const google::protobuf::RepeatedField<double>& GfvValues<double>(
    const GenericFeatureVector& gfv) {
  return gfv.feature_value_double();
}

template <typename T>
shared_ptr<T> MakeDummyShared(T* ptr);

template <typename T>
AccumulatorTypeFor<T> Sum(ConstSpan<T> terms) {
  struct SimpleAdd {
    void operator()(AccumulatorTypeFor<T>* acc, const T term) { *acc += term; }
  };

  return DenseSingleAccumulate(terms, SimpleAdd());
}

#ifdef __SSE3__

template <>
inline double Sum<double>(ConstSpan<double> terms) {
  const double* cur = terms.data();
  const double* end = cur + terms.size();
  __m128d accumulator = _mm_setzero_pd();
  if (cur + 8 <= end) {
    __m128d accumulator0 = _mm_loadu_pd(cur);
    __m128d accumulator1 = _mm_loadu_pd(cur + 2);
    __m128d accumulator2 = _mm_loadu_pd(cur + 4);
    __m128d accumulator3 = _mm_loadu_pd(cur + 6);
    cur += 8;
    for (; cur + 8 <= end; cur += 8) {
      accumulator0 = _mm_add_pd(accumulator0, _mm_loadu_pd(cur));
      accumulator1 = _mm_add_pd(accumulator1, _mm_loadu_pd(cur + 2));
      accumulator2 = _mm_add_pd(accumulator2, _mm_loadu_pd(cur + 4));
      accumulator3 = _mm_add_pd(accumulator3, _mm_loadu_pd(cur + 6));
    }

    accumulator = _mm_add_pd(_mm_add_pd(accumulator0, accumulator1),
                             _mm_add_pd(accumulator2, accumulator3));
  }

  while (cur + 2 <= end) {
    accumulator = _mm_add_pd(accumulator, _mm_loadu_pd(cur));
    cur += 2;
  }

  accumulator = _mm_hadd_pd(accumulator, accumulator);
  double result = accumulator[0];

  if (cur < end) {
    result += *cur;
  }

  return result;
}

#endif

template <typename T>
double Median(ConstSpan<T> nums) {
  vector<T> copied(nums.begin(), nums.end());
  return MedianDestructive<T>(&copied);
}

template <typename T>
double MedianDestructive(MutableSpan<T> nums) {
  DCHECK_GT(nums.size(), 0);
  size_t mid = nums.size() / 2;
  if ((nums.size() & 1) == 0) --mid;
  ZipNthElementBranchOptimized(std::less<T>(), mid, nums.begin(), nums.end());
  if (nums.size() & 1) return nums[mid];

  T min_right = nums[mid + 1];
  for (size_t i = mid + 2; i < nums.size(); ++i) {
    min_right = std::min(min_right, nums[i]);
  }

  return 0.5 * min_right + 0.5 * nums[mid];
}

template <typename Float>
Float InverseErf(Float x) {
  constexpr Float kTol = 1e-5;
  Float max = (x > 0.0) ? (numeric_limits<Float>::max()) : 0.0;
  Float min = (x < 0.0) ? (-numeric_limits<Float>::max()) : 0.0;
  while (max - min > kTol) {
    const Float mid =
        static_cast<Float>(0.5) * max + static_cast<Float>(0.5) * min;
    const Float cur = std::erf(mid);
    if (cur > x) {
      max = mid;
    } else {
      min = mid;
    }
  }
  return static_cast<Float>(0.5) * max + static_cast<Float>(0.5) * min;
}

template <typename T>
void PermuteInPlace(MutableSpan<T> a, MutableSpan<uint32_t> perm) {
  constexpr uint32_t kDoneMask = static_cast<uint32_t>(1) << 31;
  constexpr uint32_t kIndexMask = ~kDoneMask;
  auto set_done = [&perm](size_t i) -> void { perm[i] |= kDoneMask; };

  auto permutation_index = [&perm](size_t i) -> uint32_t {
    return perm[i] & kIndexMask;
  };

  auto is_done = [&perm](size_t i) -> bool { return perm[i] & kDoneMask; };

  DCHECK_EQ(a.size(), perm.size());
  DCHECK_LT(perm.size(), kDoneMask);
  for (size_t i = 0; i < a.size(); ++i) {
    if (is_done(i)) continue;
    set_done(i);
    for (uint32_t j = permutation_index(i); j != i; j = permutation_index(j)) {
      DCHECK(!is_done(j));
      using std::swap;
      swap(a[i], a[j]);
      set_done(j);
    }
  }

  for (uint32_t& u : perm) {
    u &= kIndexMask;
  }
}

template <typename T>
void PermuteInPlaceInverse(MutableSpan<T> a, MutableSpan<uint32_t> perm) {
  constexpr uint32_t kDoneMask = static_cast<uint32_t>(1) << 31;
  constexpr uint32_t kIndexMask = ~kDoneMask;
  auto set_done = [&perm](size_t i) -> void { perm[i] |= kDoneMask; };

  auto permutation_index = [&perm](size_t i) -> uint32_t {
    return perm[i] & kIndexMask;
  };

  auto is_done = [&perm](size_t i) -> bool { return perm[i] & kDoneMask; };

  DCHECK_EQ(a.size(), perm.size());
  DCHECK_LT(perm.size(), kDoneMask);
  for (size_t start = 0; start < a.size(); ++start) {
    if (is_done(start)) continue;
    set_done(start);
    uint32_t i = start;
    for (uint32_t j = permutation_index(i); j != start;
         i = j, j = permutation_index(j)) {
      DCHECK(!is_done(j));
      using std::swap;
      swap(a[i], a[j]);
      set_done(j);
    }
  }

  for (uint32_t& u : perm) {
    u &= kIndexMask;
  }
}

template <typename Uint>
void InvertPermutationInPlace(MutableSpan<Uint> perm) {
  static_assert(
      std::is_unsigned<Uint>::value && IsIntegerType<Uint>(),
      "Uint must be an unsigned integral type for InvertPermutationInPlace.");
  constexpr uint8_t kNumBits = sizeof(Uint) * 8;
  constexpr Uint kDoneMask = static_cast<Uint>(1) << (kNumBits - 1);
  constexpr Uint kIndexMask = static_cast<Uint>(~kDoneMask);

  DCHECK_LT(perm.size(), kDoneMask)
      << "Upper bit must not be set in any permutation member for "
      << "InvertPermutationInPlace.";

  auto permutation_index = [&perm](Uint i) -> Uint {
    return perm[i] & kIndexMask;
  };

  auto is_done = [&perm](size_t i) -> bool { return perm[i] & kDoneMask; };

  const Uint sz = perm.size();
  for (Uint i = 0; i < sz; ++i) {
    if (is_done(i)) continue;
    Uint prev_j = i;
    Uint j = permutation_index(i);
    while (true) {
      const Uint next_j = permutation_index(j);
      perm[j] = prev_j | kDoneMask;
      if (j == i) break;
      prev_j = j;
      j = next_j;
    }
  }

  for (Uint& u : perm) {
    u &= kIndexMask;
  }
}

template <typename Uint>
void InvertPermutationOutOfPlace(ConstSpan<Uint> perm,
                                 MutableSpan<Uint> result) {
  static_assert(std::is_unsigned<Uint>::value && IsIntegerType<Uint>(),
                "Uint must be an unsigned integral type for "
                "InvertPermutationOutOfPlace.");
  CHECK_EQ(perm.size(), result.size());
  for (Uint i : IndicesOf(perm)) {
    result[perm[i]] = i;
  }
}

template <typename T, typename U>
inline void PointwiseAdd(T* __restrict__ a, const U* __restrict__ b,
                         size_t size) {
  T* stop = a + size;
  for (; a < stop; ++a, ++b) {
    *a += *b;
  }
}

template <typename Iterator, typename Comparator>
void SiftFrontDown(Iterator begin, Iterator end, Comparator cmp) {
  DCHECK(begin <= end);
  size_t root = 0;
  size_t child = 1;
  const size_t sz = end - begin;
  while (child < sz) {
    size_t to_swap = root;
    if (cmp(begin[root], begin[child])) {
      to_swap = child;
    }

    if (child + 1 < sz && cmp(begin[to_swap], begin[child + 1])) {
      to_swap = child + 1;
    }

    if (to_swap == root) return;
    using std::swap;
    swap(begin[root], begin[to_swap]);
    root = to_swap;
    child = root * 2 + 1;
  }
}

template <typename T>
AccumulatorTypeFor<T> ParallelSum(ConstSpan<T> terms, ThreadPool* pool) {
  constexpr size_t kBlockSize = 131072;
  if (terms.size() <= kBlockSize || pool == nullptr) {
    return Sum(terms);
  }

  const size_t num_blocks =
      terms.size() / kBlockSize + (terms.size() % kBlockSize > 0);
  vector<AccumulatorTypeFor<T>> block_sums(num_blocks);
  ParallelFor<1>(Seq(num_blocks), pool, [terms, &block_sums](size_t block_num) {
    const size_t start = block_num * kBlockSize;
    DCHECK_LT(start, terms.size());
    const size_t size = (start + kBlockSize > terms.size())
                            ? (terms.size() - start)
                            : kBlockSize;
    block_sums[block_num] = Sum(ConstSpan<T>(terms.data() + start, size));
  });

  return Sum<AccumulatorTypeFor<T>>(block_sums);
}

template <typename T>
shared_ptr<T> MakeDummyShared(T* ptr) {
  return shared_ptr<T>(ptr, [](T* ptr) {});
}

void PackNibblesDatapoint(const DatapointPtr<uint8_t>& hash,
                          Datapoint<uint8_t>* packed);

void PackNibblesDatapoint(ConstSpan<uint8_t> hash, MutableSpan<uint8_t> packed);

void UnpackNibblesDatapoint(const DatapointPtr<uint8_t>& packed,
                            Datapoint<uint8_t>* hash);

void UnpackNibblesDatapoint(ConstSpan<uint8_t> packed,
                            MutableSpan<uint8_t> hash,
                            DimensionIndex hash_size);

}  // namespace research_scann

#endif
