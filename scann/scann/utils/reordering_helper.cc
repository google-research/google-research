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

#include "scann/utils/reordering_helper.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/prefetch.h"

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

template <typename ResultElemT, typename CallbackFunctor>
SCANN_INLINE void DenseDotProductDistanceOneToManyInt8FloatDispatch(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<ResultElemT> result, CallbackFunctor* __restrict__ callback) {
  size_t j = 0;

#ifdef __x86_64__
  constexpr size_t kUnrollFactor = 3;
  using DatasetView = DefaultDenseDatasetView<int8_t>;
  auto view = DatasetView(database);
  if (RuntimeSupportsAvx2()) {
    avx2::DenseDotProductDistanceOneToManyInt8Float<DatasetView, false,
                                                    DatapointIndex>(
        query.values(), &view, nullptr, result, callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  } else if (RuntimeSupportsAvx1()) {
    avx1::DenseDotProductDistanceOneToManyInt8Float<DatasetView, false,
                                                    DatapointIndex>(
        query.values(), &view, nullptr, result, callback);
    j = result.size() / kUnrollFactor * kUnrollFactor;
  }
#endif

  for (; j < result.size(); ++j) {
    const size_t idx = GetDatapointIndex(result, j);
    const float dist = -DenseDotProduct(query, database[idx]);
    callback->invoke(j, dist);
  }
}

using NeighborResult = std::pair<DatapointIndex, float>;

class SetCosineDistanceFunctor {
 public:
  explicit SetCosineDistanceFunctor(MutableSpan<NeighborResult> result_span)
      : result_(result_span) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    SetDistance(result_, index, val + 1);
  }

  SCANN_INLINE void prefetch(size_t index) {}

 private:
  MutableSpan<NeighborResult> result_;
};

class SetCosineTop1Functor {
 public:
  SCANN_INLINE void invoke(size_t index, float val) {
    val += 1;

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    nn_result_index_ = index;
  }

  SCANN_INLINE void prefetch(size_t index) {}

  NeighborResult Top1Pair(const NNResultsVector& nn_result) {
    return std::make_pair(nn_result[nn_result_index_].first,
                          smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < nn_result_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex nn_result_index_ = kInvalidDatapointIndex;
};

class SetSquaredL2DistanceFunctor {
 public:
  explicit SetSquaredL2DistanceFunctor(MutableSpan<NeighborResult> result_span,
                                       ConstSpan<float> dp_squared_l2_norms,
                                       float query_norm)
      : result_(result_span),
        dp_squared_l2_norms_(dp_squared_l2_norms),
        query_norm_(query_norm) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    SetDistance(
        result_, index,
        query_norm_ + dp_squared_l2_norms_[result_[index].first] + 2.0f * val);
  }

  SCANN_INLINE void prefetch(size_t index) {
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        &dp_squared_l2_norms_[index]);
  }

 private:
  MutableSpan<NeighborResult> result_;
  ConstSpan<float> dp_squared_l2_norms_;
  const float query_norm_;
};

class SetSquaredL2Top1Functor {
 public:
  explicit SetSquaredL2Top1Functor(ConstSpan<NeighborResult> result_span,
                                   ConstSpan<float> dp_squared_l2_norms,
                                   float query_norm)
      : result_(result_span),
        dp_squared_l2_norms_(dp_squared_l2_norms),
        query_norm_(query_norm) {}

  SCANN_INLINE void invoke(size_t index, float val) {
    val = query_norm_ + dp_squared_l2_norms_[result_[index].first] + 2.0 * val;

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    dp_index_ = result_[index].first;
  }

  SCANN_INLINE void prefetch(size_t index) {
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        &dp_squared_l2_norms_[index]);
  }

  std::pair<DatapointIndex, float> Top1Pair() {
    return std::make_pair(dp_index_, smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < dp_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex dp_index_ = kInvalidDatapointIndex;
  ConstSpan<NeighborResult> result_;
  ConstSpan<float> dp_squared_l2_norms_;
  const float query_norm_;
};

class SetLimitedInnerDistanceFunctor {
 public:
  explicit SetLimitedInnerDistanceFunctor(
      MutableSpan<NeighborResult> result_span,
      ConstSpan<float> inverse_database_l2_norms, float inverse_query_norm)
      : result_(result_span),
        inverse_database_l2_norms_(inverse_database_l2_norms),
        inverse_query_norm_(inverse_query_norm) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    val *= inverse_query_norm_ *
           std::min(inverse_database_l2_norms_[result_[index].first],
                    inverse_query_norm_);
    SetDistance(result_, index, val);
  }

  SCANN_INLINE void prefetch(size_t index) {
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        &inverse_database_l2_norms_[index]);
  }

 private:
  MutableSpan<NeighborResult> result_;
  ConstSpan<float> inverse_database_l2_norms_;
  const float inverse_query_norm_;
};

class SetLimitedInnerTop1Functor {
 public:
  explicit SetLimitedInnerTop1Functor(
      ConstSpan<NeighborResult> result_span,
      ConstSpan<float> inverse_database_l2_norms, float inverse_query_norm)
      : result_(result_span),
        inverse_database_l2_norms_(inverse_database_l2_norms),
        inverse_query_norm_(inverse_query_norm) {}

  SCANN_INLINE void invoke(size_t index, float val) {
    val *= inverse_query_norm_ *
           std::min(inverse_database_l2_norms_[result_[index].first],
                    inverse_query_norm_);

    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    dp_index_ = result_[index].first;
  }

  SCANN_INLINE void prefetch(size_t index) {
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        &inverse_database_l2_norms_[index]);
  }

  std::pair<DatapointIndex, float> Top1Pair() {
    return std::make_pair(dp_index_, smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, float val) {
    float smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val || std::isunordered(smallest, val);
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < dp_index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<float> smallest_{std::numeric_limits<float>::max()};
  DatapointIndex dp_index_ = kInvalidDatapointIndex;
  ConstSpan<NeighborResult> result_;
  ConstSpan<float> inverse_database_l2_norms_;
  const float inverse_query_norm_;
};

}  // namespace one_to_many_low_level

template <typename T>
Status ExactReorderingHelper<T>::ComputeDistancesForReordering(
    const DatapointPtr<T>& query, NNResultsVector* result) const {
  DCHECK(exact_reordering_dataset_);

  if (query.IsDense() && exact_reordering_dataset_->IsDense()) {
    const auto& dense_dataset =
        *down_cast<const DenseDataset<T>*>(exact_reordering_dataset_.get());
    DenseDistanceOneToMany<T, pair<DatapointIndex, float>>(
        *exact_reordering_distance_, query, dense_dataset,
        MakeMutableSpan(*result));
  } else if (query.IsSparse() && exact_reordering_dataset_->IsSparse()) {
    const auto& sparse_dataset =
        *down_cast<const SparseDataset<T>*>(exact_reordering_dataset_.get());
    for (auto& elem : *result) {
      elem.second = exact_reordering_distance_->GetDistanceSparse(
          query, sparse_dataset[elem.first]);
    }
  } else {
    for (auto& elem : *result) {
      elem.second = exact_reordering_distance_->GetDistanceHybrid(
          query, (*exact_reordering_dataset_)[elem.first]);
    }
  }

  return OkStatus();
}

template <typename T>
StatusOr<std::pair<DatapointIndex, float>>
ExactReorderingHelper<T>::ComputeTop1ReorderingDistance(
    const DatapointPtr<T>& query, NNResultsVector* result) const {
  if (query.IsDense() && exact_reordering_dataset_->IsDense()) {
    const auto& dense_dataset =
        *down_cast<const DenseDataset<T>*>(exact_reordering_dataset_.get());
    return DenseDistanceOneToManyTop1<T, float, pair<DatapointIndex, float>>(
        *exact_reordering_distance_, query, dense_dataset,
        MakeMutableSpan(*result));
  }
  if (query.IsSparse() && exact_reordering_dataset_->IsSparse()) {
    const auto& sparse_dataset =
        *down_cast<const SparseDataset<T>*>(exact_reordering_dataset_.get());
    float smallest = std::numeric_limits<float>::max();
    DatapointIndex idx = kInvalidDatapointIndex;
    for (auto& elem : *result) {
      float dist = exact_reordering_distance_->GetDistanceSparse(
          query, sparse_dataset[elem.first]);
      idx = dist < smallest ? elem.first : idx;
      smallest = std::min(smallest, dist);
    }
    return std::make_pair(idx, smallest);
  }
  float smallest = std::numeric_limits<float>::max();
  DatapointIndex idx = kInvalidDatapointIndex;
  for (auto& elem : *result) {
    float dist = exact_reordering_distance_->GetDistanceHybrid(
        query, (*exact_reordering_dataset_)[elem.first]);
    idx = dist < smallest ? elem.first : idx;
    smallest = std::min(smallest, dist);
  }
  return std::make_pair(idx, smallest);
}

FixedPointFloatDenseDotProductReorderingHelper::
    FixedPointFloatDenseDotProductReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile) {
  ScalarQuantizationResults quantization_results = ScalarQuantizeFloatDataset(
      exact_reordering_dataset, fixed_point_multiplier_quantile);
  fixed_point_dataset_ = std::move(quantization_results.quantized_dataset);
  inverse_multipliers_ =
      std::move(quantization_results.inverse_multiplier_by_dimension);
}

FixedPointFloatDenseDotProductReorderingHelper::
    FixedPointFloatDenseDotProductReorderingHelper(
        DenseDataset<int8_t> fixed_point_dataset,
        const shared_ptr<const vector<float>>& multiplier_by_dimension)
    : fixed_point_dataset_(std::move(fixed_point_dataset)) {
  DCHECK_EQ(multiplier_by_dimension->size(),
            fixed_point_dataset_.dimensionality());
  inverse_multipliers_.resize(multiplier_by_dimension->size());
  for (size_t i = 0; i < multiplier_by_dimension->size(); ++i) {
    inverse_multipliers_[i] = 1.0f / (*multiplier_by_dimension)[i];
  }
}

FixedPointFloatDenseDotProductReorderingHelper::
    ~FixedPointFloatDenseDotProductReorderingHelper() {}

Status
FixedPointFloatDenseDotProductReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  auto preprocessed = PrepareForAsymmetricScalarQuantizedDotProduct(
      query, inverse_multipliers_);
  DenseDotProductDistanceOneToManyInt8Float(
      MakeDatapointPtr(preprocessed.get(), query.nonzero_entries()),
      fixed_point_dataset_, MakeMutableSpan(*result));

  return OkStatus();
}

template <typename CallbackFunctor>
Status
FixedPointFloatDenseDotProductReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result,
    CallbackFunctor* __restrict__ callback) const {
  auto preprocessed = PrepareForAsymmetricScalarQuantizedDotProduct(
      query, inverse_multipliers_);
  one_to_many_low_level::DenseDotProductDistanceOneToManyInt8FloatDispatch(
      MakeDatapointPtr(preprocessed.get(), query.nonzero_entries()),
      fixed_point_dataset_, MakeMutableSpan(*result), callback);
  return OkStatus();
}

StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseDotProductReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetTop1Functor<std::pair<DatapointIndex, float>, float>
      set_top1_functor;
  SCANN_RETURN_IF_ERROR(
      ComputeDistancesForReordering(query, result, &set_top1_functor));
  return set_top1_functor.Top1Pair(MakeMutableSpan(*result));
}

Status FixedPointFloatDenseDotProductReorderingHelper::Reconstruct(
    DatapointIndex i, MutableSpan<float> output) const {
  if (i >= fixed_point_dataset_.size())
    return InvalidArgumentError(
        "The datapoint index %d is >= the dataset size %d", i,
        fixed_point_dataset_.size());

  const auto* dp_start = fixed_point_dataset_[i].values();
  std::transform(dp_start, dp_start + dimensionality(),
                 inverse_multipliers_.begin(), output.begin(),
                 std::multiplies<float>());
  return OkStatus();
}

FixedPointFloatDenseCosineReorderingHelper::
    FixedPointFloatDenseCosineReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile) {
  DCHECK_EQ(exact_reordering_dataset.normalization(), UNITL2NORM);
}

FixedPointFloatDenseCosineReorderingHelper::
    FixedPointFloatDenseCosineReorderingHelper(
        DenseDataset<int8_t> fixed_point_dataset,
        shared_ptr<const vector<float>> multiplier_by_dimension)
    : dot_product_helper_(std::move(fixed_point_dataset),
                          multiplier_by_dimension) {}

FixedPointFloatDenseCosineReorderingHelper::
    ~FixedPointFloatDenseCosineReorderingHelper() {}

Status
FixedPointFloatDenseCosineReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetCosineDistanceFunctor set_cosine_dist_functor(
      MakeMutableSpan(*result));
  return dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_cosine_dist_functor);
}

StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseCosineReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  one_to_many_low_level::SetCosineTop1Functor set_cosine_top1_functor;
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_cosine_top1_functor));
  return set_cosine_top1_functor.Top1Pair(*result);
}

FixedPointFloatDenseSquaredL2ReorderingHelper::
    FixedPointFloatDenseSquaredL2ReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile) {
  vector<float> database_squared_l2_norms;
  database_squared_l2_norms.reserve(exact_reordering_dataset.size());
  for (DatapointIndex i = 0; i < exact_reordering_dataset.size(); ++i) {
    database_squared_l2_norms.push_back(
        SquaredL2Norm(exact_reordering_dataset[i]));
  }
  database_squared_l2_norms_ =
      std::make_shared<vector<float>>(std::move(database_squared_l2_norms));
}

FixedPointFloatDenseSquaredL2ReorderingHelper::
    FixedPointFloatDenseSquaredL2ReorderingHelper(
        DenseDataset<int8_t> fixed_point_dataset,
        shared_ptr<const vector<float>> multiplier_by_dimension,
        shared_ptr<const vector<float>> squared_l2_norm_by_datapoint)
    : dot_product_helper_(std::move(fixed_point_dataset),
                          multiplier_by_dimension),
      database_squared_l2_norms_(std::move(squared_l2_norm_by_datapoint)) {
  DCHECK_EQ(database_squared_l2_norms_->size(),
            dot_product_helper_.fixed_point_dataset_.size());
}

Status
FixedPointFloatDenseSquaredL2ReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float query_norm = SquaredL2Norm(query);
  one_to_many_low_level::SetSquaredL2DistanceFunctor set_sql2_dist(
      MakeMutableSpan(*result), *database_squared_l2_norms_, query_norm);
  return dot_product_helper_.ComputeDistancesForReordering(query, result,
                                                           &set_sql2_dist);
}

StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseSquaredL2ReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float query_norm = SquaredL2Norm(query);
  one_to_many_low_level::SetSquaredL2Top1Functor set_sql2_top1(
      *result, *database_squared_l2_norms_, query_norm);
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &set_sql2_top1));
  return set_sql2_top1.Top1Pair();
}

FixedPointFloatDenseLimitedInnerReorderingHelper::
    FixedPointFloatDenseLimitedInnerReorderingHelper(
        const DenseDataset<float>& exact_reordering_dataset,
        float fixed_point_multiplier_quantile)
    : dot_product_helper_(exact_reordering_dataset,
                          fixed_point_multiplier_quantile) {
  vector<float> inverse_database_l2_norms;
  inverse_database_l2_norms.reserve(exact_reordering_dataset.size());
  for (DatapointIndex i = 0; i < exact_reordering_dataset.size(); ++i) {
    inverse_database_l2_norms.push_back(
        1.0 / std::sqrt(SquaredL2Norm(exact_reordering_dataset[i])));
  }
  inverse_database_l2_norms_ = std::move(inverse_database_l2_norms);
}

Status
FixedPointFloatDenseLimitedInnerReorderingHelper::ComputeDistancesForReordering(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float inverse_query_norm = 1.0 / std::sqrt(SquaredL2Norm(query));
  one_to_many_low_level::SetLimitedInnerDistanceFunctor set_limited_dist(
      MakeMutableSpan(*result), inverse_database_l2_norms_, inverse_query_norm);
  return dot_product_helper_.ComputeDistancesForReordering(query, result,
                                                           &set_limited_dist);
}

StatusOr<std::pair<DatapointIndex, float>>
FixedPointFloatDenseLimitedInnerReorderingHelper::ComputeTop1ReorderingDistance(
    const DatapointPtr<float>& query, NNResultsVector* result) const {
  const float inverse_query_norm = 1.0 / std::sqrt(SquaredL2Norm(query));
  one_to_many_low_level::SetLimitedInnerTop1Functor top1_functor(
      *result, inverse_database_l2_norms_, inverse_query_norm);
  SCANN_RETURN_IF_ERROR(dot_product_helper_.ComputeDistancesForReordering(
      query, result, &top1_functor));
  return top1_functor.Top1Pair();
}

SCANN_INSTANTIATE_TYPED_CLASS(, ExactReorderingHelper);

}  // namespace research_scann
