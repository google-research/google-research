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

#include "scann/base/reordering_helper_factory.h"

#include <memory>
#include <utility>

#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/utils/reordering_helper.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
using StatusOrHelper = StatusOr<unique_ptr<ReorderingInterface<T>>>;

namespace {

template <typename T>
StatusOrHelper<T> BuildBfloat16ReorderingHelper(
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<T>>& dataset,
    SingleMachineFactoryOptions* opts, float noise_shaping_threshold) {
  return InvalidArgumentError(
      "BFloat16 reordering is only supported for float32 return types.");
}

template <typename T>
StatusOrHelper<T> BuildFixedPointReorderingHelper(
    const FixedPoint& config,
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<T>>& dataset,
    SingleMachineFactoryOptions* opts) {
  return InvalidArgumentError(
      "Fixed-point reordering is only supported for float types.");
}

template <>
StatusOrHelper<float> BuildBfloat16ReorderingHelper(
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<float>>& dataset,
    SingleMachineFactoryOptions* opts, float noise_shaping_threshold) {
  if (dataset && !dataset->IsDense()) return {nullptr};
  const auto& distance_type = typeid(*reordering_dist);
  const bool is_dot = (distance_type == typeid(const DotProductDistance));
  const bool is_squared_l2 = (distance_type == typeid(const SquaredL2Distance));
  if (!is_dot && !is_squared_l2) {
    return UnimplementedError(
        "For now, bfloat16 reordering only supports DotProductDistance and "
        "SquaredL2Distance.");
  }

  if (opts->bfloat16_dataset) {
    if (is_dot) {
      return {make_unique<Bfloat16DenseDotProductReorderingHelper>(
          std::move(opts->bfloat16_dataset), noise_shaping_threshold)};
    } else {
      return {make_unique<Bfloat16DenseSquaredL2ReorderingHelper>(
          std::move(opts->bfloat16_dataset))};
    }
  } else {
    if (dataset == nullptr)
      return FailedPreconditionError(
          "No dataset provided; this is required when bfloat16_dataset isn't "
          "present in opts");
    const DenseDataset<float>* dense_dataset =
        down_cast<const DenseDataset<float>*>(dataset.get());
    if (dense_dataset == nullptr)
      return FailedPreconditionError("Failed to cast to DenseDataset<float>.");
    if (is_dot) {
      return {make_unique<Bfloat16DenseDotProductReorderingHelper>(
          *dense_dataset, noise_shaping_threshold,
          opts->parallelization_pool.get())};
    } else {
      return {
          make_unique<Bfloat16DenseSquaredL2ReorderingHelper>(*dense_dataset)};
    }
  }
}

template <>
StatusOrHelper<float> BuildFixedPointReorderingHelper<float>(
    const FixedPoint& config,
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<float>>& dataset,
    SingleMachineFactoryOptions* opts) {
  if (dataset && !dataset->IsDense()) return {nullptr};
  const auto& distance_type = typeid(*reordering_dist);

  if (opts->pre_quantized_fixed_point) {
    SCANN_RET_CHECK(opts->pre_quantized_fixed_point->fixed_point_dataset);
    SCANN_RET_CHECK(opts->pre_quantized_fixed_point->multiplier_by_dimension);
    SCANN_RET_CHECK_EQ(
        opts->pre_quantized_fixed_point->fixed_point_dataset->dimensionality(),
        opts->pre_quantized_fixed_point->multiplier_by_dimension->size())
            .SetCode(absl::StatusCode::kInvalidArgument)
        << "Multipliers for pre-quantized FP8 reordering must be of the same "
           "dimensionality as the pre-quantized dataset.";
    if (distance_type == typeid(const DotProductDistance)) {
      return {make_unique<FixedPointFloatDenseDotProductReorderingHelper>(
          std::move(opts->pre_quantized_fixed_point->fixed_point_dataset),
          *opts->pre_quantized_fixed_point->multiplier_by_dimension,
          config.noise_shaping_threshold())};
    } else if (distance_type == typeid(const CosineDistance)) {
      return {make_unique<FixedPointFloatDenseCosineReorderingHelper>(
          std::move(opts->pre_quantized_fixed_point->fixed_point_dataset),
          *opts->pre_quantized_fixed_point->multiplier_by_dimension,
          config.noise_shaping_threshold())};
    } else if (distance_type == typeid(const SquaredL2Distance)) {
      return {make_unique<FixedPointFloatDenseSquaredL2ReorderingHelper>(
          std::move(opts->pre_quantized_fixed_point->fixed_point_dataset),
          *opts->pre_quantized_fixed_point->multiplier_by_dimension,
          std::move(
              opts->pre_quantized_fixed_point->squared_l2_norm_by_datapoint))};
    } else {
      return InvalidArgumentError(
          "Fixed-point reordering is supported only for dot product, cosine "
          "and squared L2 distance.");
    }
  } else {
    DCHECK(dataset);
    const float fp_quantile = config.fixed_point_multiplier_quantile();
    if (fp_quantile > 1.0f || fp_quantile <= 0.0f) {
      return InvalidArgumentError(
          "exact_reordering.fixed_point.fixed_point_multiplier_quantile must "
          "be in the range (0.0, 1.0].");
    }
    const DenseDataset<float>& dense_dataset =
        *down_cast<const DenseDataset<float>*>(dataset.get());
    if (distance_type == typeid(const DotProductDistance)) {
      return {make_unique<FixedPointFloatDenseDotProductReorderingHelper>(
          dense_dataset, fp_quantile, config.noise_shaping_threshold(),
          opts->parallelization_pool.get())};
    } else if (distance_type == typeid(const CosineDistance)) {
      return {make_unique<FixedPointFloatDenseCosineReorderingHelper>(
          dense_dataset, fp_quantile, config.noise_shaping_threshold(),
          opts->parallelization_pool.get())};
    } else if (distance_type == typeid(const SquaredL2Distance)) {
      return {make_unique<FixedPointFloatDenseSquaredL2ReorderingHelper>(
          dense_dataset, fp_quantile)};
    } else if (distance_type == typeid(const LimitedInnerProductDistance)) {
      return {make_unique<FixedPointFloatDenseLimitedInnerReorderingHelper>(
          dense_dataset, fp_quantile)};
    } else {
      return InvalidArgumentError(
          "Fixed-point reordering is supported only for dot product, cosine "
          "and squared L2 distance.");
    }
  }
}

template <typename T>
StatusOrHelper<T> ExactReorderingFactory(
    const ExactReordering& config,
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<T>>& dataset,
    SingleMachineFactoryOptions* opts) {
  if (config.bfloat16().enabled()) {
    return BuildBfloat16ReorderingHelper<T>(
        reordering_dist, dataset, opts,
        config.bfloat16().noise_shaping_threshold());
  }
  if (config.fixed_point().enabled() || config.use_fixed_point_if_possible()) {
    auto statusor = BuildFixedPointReorderingHelper<T>(
        config.fixed_point(), reordering_dist, dataset, opts);
    if (statusor.ok()) {
      return statusor;
    } else if (!config.use_fixed_point_if_possible()) {
      return statusor;
    } else {
    }
  }
  return {make_unique<ExactReorderingHelper<T>>(reordering_dist, dataset)};
}

}  // namespace

template <typename T>
StatusOr<unique_ptr<const ReorderingInterface<T>>>
ReorderingHelperFactory<T>::Build(
    const ScannConfig& config,
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    shared_ptr<TypedDataset<T>> dataset, SingleMachineFactoryOptions* opts) {
  if (config.has_exact_reordering()) {
    return ExactReorderingFactory<T>(config.exact_reordering(), reordering_dist,
                                     dataset, opts);
  } else {
    return {nullptr};
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, ReorderingHelperFactory);

}  // namespace research_scann
