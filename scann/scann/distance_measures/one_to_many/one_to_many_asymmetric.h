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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_ASYMMETRIC_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_ASYMMETRIC_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "absl/base/optimization.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/one_to_many/one_to_many_helpers.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/fma.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/prefetch.h"

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
SCANN_OUTLINE void OneToManyInt8FloatImpl(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const DimensionIndex dims = dataset_view.dimensionality();
  DatapointPtr<float> query_dptr(nullptr, query, dims, dims);
  for (size_t j : Seq(result.size())) {
    const size_t idx =
        kHasIndices ? indices[j]
                    : one_to_many_low_level::GetDatapointIndex(result, j);
    DatapointPtr<int8_t> db_dptr(nullptr, dataset_view.GetPtr(idx), dims, dims);
    if constexpr (kIsSquaredL2) {
      float dist = 0.0;
      for (size_t j : Seq(dims)) {
        const float scaled_db_val = static_cast<float>(db_dptr.values()[j]) *
                                    inv_multipliers_for_squared_l2[j];
        const float diff = query_dptr.values()[j] - scaled_db_val;
        dist += (diff * diff);
      }
      callback.invoke(j, dist);
    } else {
      callback.invoke(j, -DenseDotProduct(query_dptr, db_dptr));
    }
  }
}

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
    for (int i : Seq(dims)) {
      db_fp32[i] = Bfloat16Decompress(db_dptr.values()[i]);
    }
    if constexpr (kIsSquaredL2) {
      callback.invoke(
          j, DenseSquaredL2Distance(query_dptr, MakeDatapointPtr(db_fp32)));
    } else {
      callback.invoke(j,
                      -DenseDotProduct(query_dptr, MakeDatapointPtr(db_fp32)));
    }
  }
}

}  // namespace fallback

#endif

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

  fallback::OneToManyInt8FloatImpl<kHasIndices, kIsSquaredL2>(
      query, dataset_view, inv_multipliers_for_squared_l2, indices, result,
      callback);

#endif
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

  fallback::OneToManyBf16FloatImpl<kHasIndices, kIsSquaredL2>(
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
      query, GetCopyableDatasetView(database),
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
    ConstSpan<DatapointIndex> idxs, MutableSpan<float> result) {
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<true, false>(
      query.data(), dataset_view, kNoMultipliersForDotProductDistance,
      idxs.data(), result, SetDistanceFunctor<float>(result));
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
                                              ConstSpan<DatapointIndex> idxs,
                                              MutableSpan<float> result) {
  DCHECK_EQ(query.size(), inv_multipliers.size());
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyInt8FloatDispatch<true, true>(
      query.data(), dataset_view, inv_multipliers.data(), idxs.data(), result,
      SetDistanceFunctor<float>(result));
}

}  // namespace research_scann

#endif
