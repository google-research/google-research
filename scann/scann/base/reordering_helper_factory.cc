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

#include "scann/base/reordering_helper_factory.h"

#include <memory>

#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_malloc_extension.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/utils/reordering_helper.h"
#include "scann/utils/types.h"

using std::move;

namespace research_scann {

template <typename T>
using StatusOrHelper = StatusOr<unique_ptr<ReorderingInterface<T>>>;

namespace {

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
StatusOrHelper<float> BuildFixedPointReorderingHelper<float>(
    const FixedPoint& config,
    const shared_ptr<const DistanceMeasure>& reordering_dist,
    const shared_ptr<TypedDataset<float>>& dataset,
    SingleMachineFactoryOptions* opts) {
  if (dataset && !dataset->IsDense()) return {nullptr};
  const auto& distance_type = typeid(*reordering_dist);

  if (opts->pre_quantized_fixed_point) {
    auto fixed_point_dataset =
        move(*opts->pre_quantized_fixed_point->fixed_point_dataset);
    auto multiplier_by_dimension =
        move(opts->pre_quantized_fixed_point->multiplier_by_dimension);
    SCANN_RET_CHECK(multiplier_by_dimension);
    SCANN_RET_CHECK_EQ(fixed_point_dataset.dimensionality(),
                       multiplier_by_dimension->size())
            .SetErrorCode(error::INVALID_ARGUMENT)
        << "Multipliers for pre-quantized FP8 reordering must be of the same "
           "dimensionality as the pre-quantized dataset.";
    if (distance_type == typeid(const DotProductDistance)) {
      return {make_unique<FixedPointFloatDenseDotProductReorderingHelper>(
          move(fixed_point_dataset), move(multiplier_by_dimension))};
    } else if (distance_type == typeid(const CosineDistance)) {
      return {make_unique<FixedPointFloatDenseCosineReorderingHelper>(
          move(fixed_point_dataset), move(multiplier_by_dimension))};
    } else if (distance_type == typeid(const SquaredL2Distance)) {
      return {make_unique<FixedPointFloatDenseSquaredL2ReorderingHelper>(
          move(fixed_point_dataset), move(multiplier_by_dimension),
          move(opts->pre_quantized_fixed_point->squared_l2_norm_by_datapoint))};
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
          dense_dataset, fp_quantile)};
    } else if (distance_type == typeid(const CosineDistance)) {
      return {make_unique<FixedPointFloatDenseCosineReorderingHelper>(
          dense_dataset, fp_quantile)};
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
