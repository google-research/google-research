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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_ASYMMETRIC_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_ASYMMETRIC_H_

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/one_to_many/one_to_many_helpers.h"
#include "scann/distance_measures/one_to_many/one_to_many_impl_highway.inc"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/intrinsics/fma.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<float> result);

template <typename DatasetView>
void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DatasetView* __restrict__ database,
    MutableSpan<float> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<double> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint32_t, float>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint64_t, float>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<DatapointIndex, double>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result);

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<float> result);

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<pair<DatapointIndex, float>> result);

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result);

void OneToManyBf16FloatSquaredL2(const DatapointPtr<float>& query,
                                 DefaultDenseDatasetView<int16_t> database,
                                 MutableSpan<float> result);

void OneToManyBf16FloatSquaredL2(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<pair<DatapointIndex, float>> result);

void OneToManyBf16FloatSquaredL2(const DatapointPtr<float>& query,
                                 DefaultDenseDatasetView<int16_t> database,
                                 ConstSpan<DatapointIndex> indices,
                                 MutableSpan<float> result);

#ifdef __x86_64__

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE SCANN_SSE4
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4

namespace avx1 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512

#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

#else

namespace fallback {

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_OUTLINE void OneToManyBf16FloatImpl(const float* __restrict__ query,
                                          DatasetViewT dataset_view,
                                          const IndexT* indices,
                                          MutableSpan<ResultElemT> result,
                                          CallbackT callback) {
  const DimensionIndex dims = dataset_view.dimensionality();
  DatapointPtr<float> query_dptr(nullptr, query, dims, dims);
  vector<float> db_fp32(dims);
  for (size_t j : Seq(result.size())) {
    const size_t idx =
        kHasIndices ? indices[j]
                    : one_to_many_low_level::GetDatapointIndex(result, j);
    DatapointPtr<int16_t> db_dptr(nullptr, dataset_view.GetPtr(idx), dims,
                                  dims);
    if constexpr (kIsSquaredL2) {
      for (int i : Seq(dims)) {
        db_fp32[i] = Bfloat16Decompress(db_dptr.values()[i]);
      }
      callback.invoke(
          j, DenseSquaredL2Distance(query_dptr, MakeDatapointPtr(db_fp32)));
    } else {
      callback.invoke(j, -::research_scann::dp_internal::DenseDotProductHighway(
                             db_dptr, query_dptr));
    }
  }
}

}  // namespace fallback

#endif

namespace fallback {

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT>
SCANN_OUTLINE void OneToManyUint4Int8Impl(const int8_t* __restrict__ query,
                                          DatasetViewT dataset_view,
                                          const IndexT* indices,
                                          MutableSpan<ResultElemT> result,
                                          CallbackT callback) {
  const DimensionIndex dims = dataset_view.dimensionality();
  const size_t datapoint_bytes = DivRoundUp(dims, 2);
  for (size_t j : Seq(result.size())) {
    const size_t idx =
        kHasIndices ? indices[j]
                    : one_to_many_low_level::GetDatapointIndex(result, j);
    const uint8_t* db_dptr = dataset_view.GetPtr(idx);

    int32_t dist = 0;
    const auto add_val = [&](size_t i, uint8_t fp4_val) SCANN_INLINE_LAMBDA {
      dist -= query[i] * static_cast<int32_t>(fp4_val);
    };
    size_t i = 0;
    for (; i + 1 < dims; i += 2) {
      uint8_t val_pair = db_dptr[i / 2];
      add_val(i, val_pair % 16);
      add_val(i + 1, val_pair / 16);
    }
    if (i < dims) {
      uint8_t val_pair = db_dptr[i / 2];
      add_val(i, val_pair % 16);
    }
    InvokeCallback(callback, j, dist, datapoint_bytes, db_dptr);
  }
}

}  // namespace fallback

namespace one_to_many_low_level {

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyInt8FloatDispatch(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
#ifdef __x86_64__

  if constexpr (false && RuntimeSupportsAvx512()) {
    LOG(FATAL) << "We aren't compiling Avx-512 support yet.";
  } else if (RuntimeSupportsAvx2()) {
    avx2::OneToManyInt8FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, inv_multipliers_for_squared_l2, indices, result,
        callback);
  } else if (RuntimeSupportsAvx1()) {
    avx1::OneToManyInt8FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, inv_multipliers_for_squared_l2, indices, result,
        callback);
  } else {
    sse4::OneToManyInt8FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, inv_multipliers_for_squared_l2, indices, result,
        callback);
  }
#else

  highway::OneToManyInt8FloatImpl<kHasIndices, kIsSquaredL2>(
      query, dataset_view, inv_multipliers_for_squared_l2, indices, result,
      callback);

#endif
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyScaledInt8FloatDispatch(
    ScaleEncoding scale_encoding, const float* __restrict__ query,
    DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  WithScaleFunctor(
      scale_encoding, callback, [&](auto functor) SCANN_INLINE_LAMBDA {
        return OneToManyInt8FloatDispatch<kHasIndices, kIsSquaredL2>(
            query, dataset_view, inv_multipliers_for_squared_l2, indices,
            result, functor);
      });
}

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyUint4Int8Dispatch(const int8_t* __restrict__ query,
                                             DatasetViewT dataset_view,
                                             const IndexT* indices,
                                             MutableSpan<ResultElemT> result,
                                             CallbackT callback) {
#ifdef __x86_64__
  if (RuntimeSupportsAvx2()) {
    avx2::OneToManyUint4Int8Impl<kHasIndices>(query, dataset_view, indices,
                                              result, callback);
  } else {
    sse4::OneToManyUint4Int8Impl<kHasIndices>(query, dataset_view, indices,
                                              result, callback);
  }

#else

  fallback::OneToManyUint4Int8Impl<kHasIndices>(query, dataset_view, indices,
                                                result, callback);
#endif
}

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyUint4FloatDispatch(const float* __restrict__ query,
                                              DatasetViewT dataset_view,
                                              const IndexT* indices,
                                              MutableSpan<ResultElemT> result,
                                              CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  vector<int8_t> int8_query(dims);
  float query_max_abs = 0.0f;
  float query_sum = 0.0f;
  for (size_t j : Seq(dataset_view.dimensionality())) {
    const float qval = query[j];
    query_max_abs = std::max(query_max_abs, std::abs(qval));
    query_sum += qval;
  }
  const float quantize_query_multiplier = kFP8Max / query_max_abs;
  const float dequantize_query_multiplier = query_max_abs * (1.0f / kFP8Max);
  for (size_t j : Seq(dims)) {
    int8_query[j] = Int8Quantize(query[j] * quantize_query_multiplier);
  }
  const float query_offset = query_sum * kFP4Max;
  OneToManyUint4Int8Dispatch<kHasIndices>(
      int8_query.data(), std::move(dataset_view), indices, result,
      MakeDequantizeFunctor(dequantize_query_multiplier, query_offset,
                            callback));
}

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyScaledUint4FloatDispatch(
    ScaleEncoding scale_encoding, const float* __restrict__ query,
    DatasetViewT dataset_view, const IndexT* indices,
    MutableSpan<ResultElemT> result, CallbackT callback) {
  WithScaleFunctor(scale_encoding, callback,
                   [&](auto functor) SCANN_INLINE_LAMBDA {
                     return OneToManyUint4FloatDispatch<kHasIndices>(
                         query, dataset_view, indices, result, functor);
                   });
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT,
          typename = std::enable_if_t<!std::is_pointer_v<DatasetViewT>>,
          typename = std::enable_if_t<!std::is_pointer_v<CallbackT>>>
SCANN_INLINE void OneToManyBf16FloatDispatch(const float* __restrict__ query,
                                             DatasetViewT dataset_view,
                                             const IndexT* indices,
                                             MutableSpan<ResultElemT> result,
                                             CallbackT callback) {
#ifdef __x86_64__

  if constexpr (false && RuntimeSupportsAvx512()) {
    LOG(FATAL) << "We aren't compiling Avx-512 support yet.";
  } else if (RuntimeSupportsAvx2()) {
    avx2::OneToManyBf16FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, indices, result, callback);
  } else if (RuntimeSupportsAvx1()) {
    avx1::OneToManyBf16FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, indices, result, callback);
  } else {
    sse4::OneToManyBf16FloatImpl<kHasIndices, kIsSquaredL2>(
        query, dataset_view, indices, result, callback);
  }

#else

  highway::OneToManyBf16FloatImpl<kHasIndices, kIsSquaredL2>(
      query, dataset_view, indices, result, callback);

#endif
}

template <typename DatasetViewT>
class OneToManyDatasetViewPtr {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(OneToManyDatasetViewPtr);

  explicit OneToManyDatasetViewPtr(const DatasetViewT* ptr) : ptr_(ptr) {}

  SCANN_INLINE auto GetPtr(size_t idx) const { return ptr_->GetPtr(idx); }

  SCANN_INLINE size_t dimensionality() const { return ptr_->dimensionality(); }

 private:
  const DatasetViewT* ptr_ = nullptr;
};

template <typename DatasetViewT>
SCANN_INLINE auto GetCopyableDatasetView(const DatasetViewT* dataset_view) {
  if constexpr (std::is_same_v<DatasetViewT, DefaultDenseDatasetView<float>>) {
    return *dataset_view;
  } else {
    return OneToManyDatasetViewPtr<DatasetViewT>(dataset_view);
  }
}

template <typename CallbackT>
class OneToManyCallbackPtr {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(OneToManyCallbackPtr);

  explicit OneToManyCallbackPtr(CallbackT* ptr) : ptr_(ptr) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t result_idx, ValueT val) const {
    ptr_->invoke(result_idx, val);
  }

  SCANN_INLINE void prefetch(size_t db_idx) const { ptr_->prefetch(db_idx); }

 private:
  CallbackT* ptr_ = nullptr;
};

template <typename CallbackT>
SCANN_INLINE auto GetCopyableCallback(CallbackT* callback) {
  if constexpr (std::is_same_v<CallbackT, SetDistanceFunctor<int8_t>>) {
    return *callback;
  } else {
    return OneToManyCallbackPtr<CallbackT>(callback);
  }
}

template <typename DatasetView, bool kHasIndices, typename IndexT,
          typename ResultElemT, typename CallbackLambda>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatLowLevel(
    const float* __restrict__ query, const DatasetView* dataset_view,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackLambda* callback) {
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  OneToManyInt8FloatDispatch<kHasIndices, false>(
      query, GetCopyableDatasetView(dataset_view),
      kNoMultipliersForDotProductDistance, indices, result,
      GetCopyableCallback(callback));
}

}  // namespace one_to_many_low_level

template <typename DatasetView>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DatasetView* database,
    MutableSpan<float> result) {
  constexpr const DatapointIndex* kNoIndices = nullptr;
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  using one_to_many_low_level::GetCopyableDatasetView;
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<false, false>(
      query.values(), GetCopyableDatasetView(database),
      kNoMultipliersForDotProductDistance, kNoIndices, result,
      SetDistanceFunctor<float>(result));
}

template <typename DatasetView>
SCANN_INLINE void OneToManyInt8FloatDotProductDistance(
    ConstSpan<float> query, DatasetView dataset_view,
    MutableSpan<float> result) {
  constexpr const DatapointIndex* kNoIndices = nullptr;
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<false, false>(
      query.data(), dataset_view, kNoMultipliersForDotProductDistance,
      kNoIndices, result, SetDistanceFunctor<float>(result));
}

template <typename DatasetView>
SCANN_INLINE void OneToManyInt8FloatDotProductDistance(
    ConstSpan<float> query, DatasetView dataset_view,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result) {
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<true, false>(
      query.data(), dataset_view, kNoMultipliersForDotProductDistance,
      indices.data(), result, SetDistanceFunctor<float>(result));
}

template <typename DatasetView>
SCANN_INLINE void OneToManyInt8FloatSquaredL2(ConstSpan<float> query,
                                              DatasetView dataset_view,
                                              ConstSpan<float> inv_multipliers,
                                              MutableSpan<float> result) {
  DCHECK_EQ(query.size(), inv_multipliers.size());
  constexpr const DatapointIndex* kNoIndices = nullptr;
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<false, true>(
      query.data(), dataset_view, inv_multipliers.data(), kNoIndices, result,
      SetDistanceFunctor<float>(result));
}

template <typename DatasetView>
SCANN_INLINE void OneToManyInt8FloatSquaredL2(ConstSpan<float> query,
                                              DatasetView dataset_view,
                                              ConstSpan<float> inv_multipliers,
                                              ConstSpan<DatapointIndex> indices,
                                              MutableSpan<float> result) {
  DCHECK_EQ(query.size(), inv_multipliers.size());
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<true, true>(
      query.data(), dataset_view, inv_multipliers.data(), indices.data(),
      result, SetDistanceFunctor<float>(result));
}

template <typename DatasetView, typename DatapointIndexT>
SCANN_INLINE void OneToManyInt8FloatSquaredL2(
    ConstSpan<float> query, DatasetView dataset_view,
    ConstSpan<float> inv_multipliers,
    MutableSpan<pair<DatapointIndexT, float>> result) {
  DCHECK_EQ(query.size(), inv_multipliers.size());
  using one_to_many_low_level::SetDistanceFunctor;
  constexpr const DatapointIndex* kNoIndices = nullptr;
  one_to_many_low_level::OneToManyInt8FloatDispatch<false, true>(
      query.data(), dataset_view, inv_multipliers.data(), kNoIndices, result,
      SetDistanceFunctor<pair<DatapointIndexT, float>>(result));
}

constexpr size_t kOneToManyUint4Int8SlopBytes = 32;

void DenseDotProductDistanceOneToManyUint4Int8(
    const DatapointPtr<int8_t>& query, const uint8_t* dataset,
    ConstSpan<DatapointIndex> indices, MutableSpan<int32_t> result);

void DenseDotProductDistanceOneToManyScaledUint4Float(
    ScaleEncoding scale_encoding, const DatapointPtr<float>& query,
    const uint8_t* dataset, ConstSpan<DatapointIndex> indices,
    MutableSpan<float> result);

}  // namespace research_scann

#endif
