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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/distance_measures/one_to_many/one_to_many.h"

#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/flags.h"

namespace tensorflow {
namespace scann_ops {
namespace one_to_many_low_level {

#ifdef __x86_64__

#define SCANN_SIMD_INLINE SCANN_SIMD_ATTRIBUTE SCANN_INLINE
#define SCANN_SIMD_INLINE_LAMBDA SCANN_SIMD_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_SIMD_OUTLINE SCANN_SIMD_ATTRIBUTE SCANN_OUTLINE

namespace avx1 {
using AvxFuncs = ::tensorflow::scann_ops::AvxFunctionsAvx;
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1_ATTRIBUTE
#include "scann/distance_measures/one_to_many/one_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace avx2 {
using AvxFuncs = ::tensorflow::scann_ops::AvxFunctionsAvx2Fma;
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2_ATTRIBUTE
#include "scann/distance_measures/one_to_many/one_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

#endif

template <bool kHasIndices = false, typename ResultElemT,
          typename CallbackFunctor>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatDispatch(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    const DatapointIndex* indices, MutableSpan<ResultElemT> result,
    CallbackFunctor* __restrict__ callback) {
  size_t j = 0;

#ifdef __x86_64__
  constexpr size_t kUnrollFactor = 3;
  using DatasetView = DefaultDenseDatasetView<int8_t>;
  auto view = DatasetView(database);
  if (RuntimeSupportsAvx2()) {
    avx2::DenseDotProductDistanceOneToManyInt8Float<DatasetView, kHasIndices>(
        query.values(), &view, indices, result, callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  } else if (RuntimeSupportsAvx1()) {
    avx1::DenseDotProductDistanceOneToManyInt8Float<DatasetView, kHasIndices>(
        query.values(), &view, indices, result, callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  }
#endif

  for (; j < result.size(); ++j) {
    const size_t idx = kHasIndices ? indices[j] : GetDatapointIndex(result, j);
    const float dist = -DenseDotProduct(query, database[idx]);
    callback->invoke(j, dist);
  }
}

}  // namespace one_to_many_low_level

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<float> result) {
  auto set_distance_functor =
      one_to_many_low_level::SetDistanceFunctor<float>(result);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result,
      &set_distance_functor);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<double> result) {
  auto set_distance_functor =
      one_to_many_low_level::SetDistanceFunctor<double>(result);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result,
      &set_distance_functor);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<DatapointIndex, float>> result) {
  auto set_distance_functor = one_to_many_low_level::SetDistanceFunctor<
      std::pair<DatapointIndex, float>>(result);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result,
      &set_distance_functor);
}
void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<DatapointIndex, double>> result) {
  auto set_distance_functor = one_to_many_low_level::SetDistanceFunctor<
      std::pair<DatapointIndex, double>>(result);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<DatapointIndex*>(nullptr), result,
      &set_distance_functor);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result) {
  QCHECK_EQ(indices.size(), result.size());
  auto set_distance_functor =
      one_to_many_low_level::SetDistanceFunctor<float>(result);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch<
      true>(query, database, indices.data(), result, &set_distance_functor);
}

}  // namespace scann_ops
}  // namespace tensorflow
