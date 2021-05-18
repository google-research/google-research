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

#include "scann/distance_measures/one_to_many/one_to_many.h"

#include <cstdint>

#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/simd.h"

namespace research_scann {
namespace one_to_many_low_level {

#ifdef __x86_64__

namespace avx1 {
using AvxFuncs = ::research_scann::AvxFunctionsAvx;
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/distance_measures/one_to_many/one_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace avx2 {
using AvxFuncs = ::research_scann::AvxFunctionsAvx2Fma;
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/distance_measures/one_to_many/one_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

#endif

using one_to_many_low_level::SetDistanceFunctor;

template <bool kHasIndices = false, typename ResultElemT>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatDispatch(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    const DatapointIndex* indices, MutableSpan<ResultElemT> result) {
  size_t j = 0;

  SetDistanceFunctor<ResultElemT> callback(result);

#ifdef __x86_64__
  constexpr size_t kUnrollFactor = 3;
  using DatasetView = DefaultDenseDatasetView<int8_t>;
  auto view = DatasetView(database);
  if (RuntimeSupportsAvx2()) {
    avx2::DenseDotProductDistanceOneToManyInt8Float<DatasetView, kHasIndices>(
        query.values(), &view, indices, result, &callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  } else if (RuntimeSupportsAvx1()) {
    avx1::DenseDotProductDistanceOneToManyInt8Float<DatasetView, kHasIndices>(
        query.values(), &view, indices, result, &callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  }
#endif

  for (; j < result.size(); ++j) {
    const size_t idx = kHasIndices ? indices[j] : GetDatapointIndex(result, j);
    const float dist = -DenseDotProduct(query, database[idx]);
    callback.invoke(j, dist);
  }
}

}  // namespace one_to_many_low_level

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<float> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<double> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<uint32_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<uint64_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<uint32_t, double>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    ConstSpan<uint32_t> indices, MutableSpan<float> result) {
  QCHECK_EQ(indices.size(), result.size());
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch<
      true>(query, database, indices.data(), result);
}

}  // namespace research_scann
