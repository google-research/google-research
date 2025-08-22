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

#include "scann/distance_measures/one_to_many/one_to_many_asymmetric.h"

#include <cstdint>

#include "absl/log/check.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many_helpers.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace one_to_many_low_level {

using one_to_many_low_level::SetDistanceFunctor;

template <bool kHasIndices = false, typename ResultElemT>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatDispatch(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> view,
    const DatapointIndex* indices, MutableSpan<ResultElemT> result) {
  constexpr const float* kNoMultipliersForDotProductDistance = nullptr;
  OneToManyInt8FloatDispatch<kHasIndices, false>(
      query.values(), view, kNoMultipliersForDotProductDistance, indices,
      result, SetDistanceFunctor<ResultElemT>(result));
}

}  // namespace one_to_many_low_level

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<float> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<double> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint32_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint64_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<DatapointIndex, double>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> dataset,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result) {
  QCHECK_EQ(indices.size(), result.size());
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch<
      true>(query, dataset, indices.data(), result);
}

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<float> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<false, false>(
      query.values(), database, static_cast<uint32_t*>(nullptr), result,
      one_to_many_low_level::SetDistanceFunctor<float>(result));
}

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<pair<DatapointIndex, float>> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<false, false>(
      query.values(), database, static_cast<DatapointIndex*>(nullptr), result,
      one_to_many_low_level::SetDistanceFunctor<pair<DatapointIndex, float>>(
          result));
}

void DenseDotProductDistanceOneToManyBf16Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<true, false>(
      query.values(), database, indices.data(), result,
      one_to_many_low_level::SetDistanceFunctor<float>(result));
}

void OneToManyBf16FloatSquaredL2(const DatapointPtr<float>& query,
                                 DefaultDenseDatasetView<int16_t> database,
                                 MutableSpan<float> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<false, true>(
      query.values(), database, static_cast<uint32_t*>(nullptr), result,
      one_to_many_low_level::SetDistanceFunctor<float>(result));
}

void OneToManyBf16FloatSquaredL2(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int16_t> database,
    MutableSpan<pair<DatapointIndex, float>> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<false, true>(
      query.values(), database, static_cast<DatapointIndex*>(nullptr), result,
      one_to_many_low_level::SetDistanceFunctor<pair<DatapointIndex, float>>(
          result));
}

void OneToManyBf16FloatSquaredL2(const DatapointPtr<float>& query,
                                 DefaultDenseDatasetView<int16_t> database,
                                 ConstSpan<DatapointIndex> indices,
                                 MutableSpan<float> result) {
  one_to_many_low_level::OneToManyBf16FloatDispatch<true, true>(
      query.values(), database, indices.data(), result,
      one_to_many_low_level::SetDistanceFunctor<float>(result));
}

namespace {

class Int4DenseDatasetView {
 public:
  Int4DenseDatasetView(const uint8_t* ptr, size_t dims,
                       ScaleEncoding scale_encoding)
      : ptr_(ptr),
        dims_(dims),
        stride_(one_to_many_low_level::DatapointBytes<uint8_t>(
            dims, scale_encoding)) {}

  SCANN_INLINE const uint8_t* GetPtr(size_t i) const {
    return ptr_ + i * stride_;
  }

  SCANN_INLINE size_t dimensionality() const { return dims_; }

 private:
  const uint8_t* ptr_;
  size_t dims_;
  size_t stride_;
};

}  // namespace

void DenseDotProductDistanceOneToManyUint4Int8(
    const DatapointPtr<int8_t>& query, const uint8_t* dataset,
    ConstSpan<DatapointIndex> indices, MutableSpan<int32_t> result) {
  DCHECK_EQ(indices.size(), result.size());
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyUint4Int8Dispatch<true>(
      query.values(),
      Int4DenseDatasetView(dataset, query.dimensionality(),
                           UNSPECIFIED_SCALE_ENCODING),
      indices.data(), result, SetDistanceFunctor<int32_t>(result));
}

void DenseDotProductDistanceOneToManyScaledUint4Float(
    ScaleEncoding scale_encoding, const DatapointPtr<float>& query,
    const uint8_t* dataset, ConstSpan<DatapointIndex> indices,
    MutableSpan<float> result) {
  DCHECK_EQ(indices.size(), result.size());
  using one_to_many_low_level::SetDistanceFunctor;
  one_to_many_low_level::OneToManyScaledUint4FloatDispatch<true>(
      scale_encoding, query.values(),
      Int4DenseDatasetView(dataset, query.dimensionality(), scale_encoding),
      indices.data(), result, SetDistanceFunctor<float>(result));
}

}  // namespace research_scann
