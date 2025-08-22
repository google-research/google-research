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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_HELPERS_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_HELPERS_H_

#include <atomic>
#include <type_traits>

#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/scalar_quantization_helpers.h"
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

template <typename ValueT>
SCANN_INLINE void AddDistance(ValueT* result, size_t index, float val) {
  result[index] += val;
}
template <typename ValueT1, typename ValueT2>
inline void AddDistance(MutableSpan<ValueT1> result, size_t index,
                        ValueT2 val) {
  result[index] += static_cast<ValueT1>(val);
}

template <typename IndexT, typename ValueT1, typename ValueT2>
inline void AddDistance(MutableSpan<pair<IndexT, ValueT1>> result, size_t index,
                        ValueT2 val) {
  result[index].second += static_cast<ValueT1>(val);
}
template <typename IndexT, typename ValueT1, typename ValueT2>
SCANN_INLINE void AddDistance(pair<IndexT, ValueT1>* result, size_t index,
                              ValueT2 val) {
  result[index].second += static_cast<ValueT1>(val);
}

template <typename ResultElem>
class AddDistanceFunctor {
 public:
  explicit AddDistanceFunctor(MutableSpan<ResultElem> result_span)
      : result_(result_span) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) const {
    AddDistance(result_, index, val);
  }

  SCANN_INLINE void prefetch(size_t index) const {}

 private:
  MutableSpan<ResultElem> result_;
};

template <typename Delegate>
struct DequantizeFunctor {
  DequantizeFunctor(float multiplier, float offset, Delegate delegate)
      : multiplier(multiplier), offset(offset), delegate(std::move(delegate)) {}

  SCANN_INLINE void invoke(size_t index, int32_t val,
                           uint32_t side_data) const {
    delegate.invoke(index, val * multiplier + offset, side_data);
  }
  SCANN_INLINE void invoke(size_t index, int32_t val) const {
    delegate.invoke(index, val * multiplier + offset);
  }
  SCANN_INLINE void prefetch(size_t index) const {}

  float multiplier;
  float offset;
  Delegate delegate;
};
template <typename Delegate>
SCANN_INLINE DequantizeFunctor<Delegate> MakeDequantizeFunctor(
    float multiplier, float offset, Delegate delegate) {
  return DequantizeFunctor<Delegate>(multiplier, offset, std::move(delegate));
}

template <ScaleEncoding scale_encoding, typename Delegate>
class ScaleFunctor {
 public:
  explicit ScaleFunctor(Delegate delegate) : delegate_(std::move(delegate)) {}

  SCANN_INLINE void invoke(size_t index, float val, uint32_t side_data) const {
    delegate_.invoke(index, val * absl::bit_cast<float>(side_data));
  }
  SCANN_INLINE void prefetch(size_t index) const {}

 private:
  Delegate delegate_;
};

template <ScaleEncoding scale_encoding, typename Delegate>
SCANN_INLINE ScaleFunctor<scale_encoding, Delegate> MakeScaleFunctor(
    Delegate delegate) {
  return ScaleFunctor<scale_encoding, Delegate>(delegate);
}

template <typename CallbackT, typename F>
SCANN_INLINE void WithScaleFunctor(ScaleEncoding scale_encoding,
                                   CallbackT callback, F f) {
  switch (scale_encoding) {
    case UNSPECIFIED_SCALE_ENCODING:
      return f(callback);
    case FLOAT32_SCALE_SUFFIX:
      return f(MakeScaleFunctor<FLOAT32_SCALE_SUFFIX>(callback));
    case FLOAT32_SCALE_BOTTOM_BITS:
      return f(MakeScaleFunctor<FLOAT32_SCALE_BOTTOM_BITS>(callback));
  }
}

template <typename T>
struct NeedsSuffixSideData : std::false_type {};
template <typename D>
struct NeedsSuffixSideData<ScaleFunctor<FLOAT32_SCALE_SUFFIX, D>>
    : std::true_type {};
template <typename D>
struct NeedsSuffixSideData<DequantizeFunctor<D>> : NeedsSuffixSideData<D> {};

template <typename T>
struct NeedsBottomBitsSideData : std::false_type {};
template <typename D>
struct NeedsBottomBitsSideData<ScaleFunctor<FLOAT32_SCALE_BOTTOM_BITS, D>>
    : std::true_type {};
template <typename D>
struct NeedsBottomBitsSideData<DequantizeFunctor<D>>
    : NeedsBottomBitsSideData<D> {};

template <typename CallbackT, typename ResultT, typename DataT>
SCANN_INLINE void InvokeCallback(const CallbackT& callback, size_t result_idx,
                                 ResultT val, size_t datapoint_bytes,
                                 const DataT* ptr) {
  if constexpr (NeedsSuffixSideData<CallbackT>::value) {
    callback.invoke(result_idx, val,
                    ABSL_INTERNAL_UNALIGNED_LOAD32(ptr + datapoint_bytes));
  } else if constexpr (NeedsBottomBitsSideData<CallbackT>::value) {
    const auto data = MakeConstSpan(ptr, datapoint_bytes);
    if constexpr (std::is_same_v<DataT, int8_t>) {
      callback.invoke(result_idx, val, DecodeBottomBitsDataFromInt8(data));
    } else {
      static_assert(std::is_same_v<DataT, uint8_t>);
      callback.invoke(result_idx, val,
                      DecodeBottomBitsDataFromPackedInt4(data));
    }
  } else {
    callback.invoke(result_idx, val);
  }
}

template <typename T>
SCANN_INLINE size_t DatapointBytes(size_t dims, ScaleEncoding scale_encoding) {
  size_t result;
  if constexpr (std::is_same_v<T, int8_t>) {
    result = dims;
  } else {
    static_assert(std::is_same_v<T, uint8_t>);
    result = DivRoundUp(dims, 2);
  }
  if (scale_encoding == FLOAT32_SCALE_SUFFIX) {
    result += sizeof(float);
  }
  return result;
}

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
