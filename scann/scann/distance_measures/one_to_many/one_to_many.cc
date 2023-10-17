// Copyright 2023 The Google Research Authors.
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
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<double> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint32_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint64_t, float>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> database,
    MutableSpan<pair<uint32_t, double>> result) {
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      query, database, static_cast<uint32_t*>(nullptr), result);
}

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, DefaultDenseDatasetView<int8_t> dataset,
    ConstSpan<uint32_t> indices, MutableSpan<float> result) {
  QCHECK_EQ(indices.size(), result.size());
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch<
      true>(query, dataset, indices.data(), result);
}

}  // namespace research_scann
