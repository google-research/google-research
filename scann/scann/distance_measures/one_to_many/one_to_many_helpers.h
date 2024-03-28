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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_HELPERS_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_HELPERS_H_

#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace one_to_many_low_level {

template <typename ValueT>
inline size_t GetDatapointIndex(ValueT* result, size_t index) {
  return index;
}
template <typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<ValueT> result, size_t index) {
  return index;
}

template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(pair<IndexT, ValueT>* result, size_t index) {
  return result[index].first;
}
template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<pair<IndexT, ValueT>> result,
                                size_t index) {
  return result[index].first;
}

template <typename ValueT>
SCANN_INLINE void SetDistance(ValueT* result, size_t index, float val) {
  result[index] = val;
}
template <typename ValueT1, typename ValueT2>
inline void SetDistance(MutableSpan<ValueT1> result, size_t index,
                        ValueT2 val) {
  result[index] = static_cast<ValueT1>(val);
}

template <typename IndexT, typename ValueT1, typename ValueT2>
inline void SetDistance(MutableSpan<pair<IndexT, ValueT1>> result, size_t index,
                        ValueT2 val) {
  result[index].second = static_cast<ValueT1>(val);
}
template <typename IndexT, typename ValueT1, typename ValueT2>
SCANN_INLINE void SetDistance(pair<IndexT, ValueT1>* result, size_t index,
                              ValueT2 val) {
  result[index].second = static_cast<ValueT1>(val);
}

template <typename ResultElem>
class SetDistanceFunctor {
 public:
  explicit SetDistanceFunctor(MutableSpan<ResultElem> result_span)
      : result_(result_span) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) const {
    SetDistance(result_, index, val);
  }

  SCANN_INLINE void prefetch(size_t index) const {}

 private:
  MutableSpan<ResultElem> result_;
};

template <typename ResultElem, typename ValueT>
class SetTop1Functor {
  static_assert(std::is_arithmetic<ValueT>(),
                "Value must be a arithmetic type");

 public:
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    index_ = index;
  }

  SCANN_INLINE void prefetch(size_t index) {}

  std::pair<DatapointIndex, ValueT> Top1Pair(
      MutableSpan<ResultElem> result_span) {
    if (result_span.empty()) {
      return std::make_pair(kInvalidDatapointIndex,
                            std::numeric_limits<ValueT>::max());
    }
    return std::make_pair(GetDatapointIndex(result_span, index_),
                          smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, ValueT val) {
    ValueT smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val ||
        (IsFloatingType<ValueT>() && std::isunordered(smallest, val));
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<ValueT> smallest_{std::numeric_limits<ValueT>::max()};
  DatapointIndex index_ = kInvalidDatapointIndex;
};

}  // namespace one_to_many_low_level
}  // namespace research_scann

#endif
