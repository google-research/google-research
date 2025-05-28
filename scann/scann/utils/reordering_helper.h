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



#ifndef SCANN_UTILS_REORDERING_HELPER_H_
#define SCANN_UTILS_REORDERING_HELPER_H_

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/reordering_helper_interface.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

class FixedPointFloatDenseCosineReorderingHelper;

template <typename T>
class SingleMachineSearcherBase;

template <typename T>
class ExactReorderingHelper : public ReorderingHelper<T> {
 public:
  ExactReorderingHelper(
      shared_ptr<const DistanceMeasure> exact_reordering_distance,
      shared_ptr<const TypedDataset<T>> exact_reordering_dataset)
      : exact_reordering_distance_(exact_reordering_distance),
        exact_reordering_dataset_(exact_reordering_dataset) {
    if (!exact_reordering_dataset) {
      LOG(FATAL) << "Cannot enable exact reordering when the original "
                 << "dataset is empty.";
    }
  }

  std::string name() const override { return "ExactReordering"; }

  bool needs_dataset() const override { return true; }

  Status ComputeDistancesForReordering(const DatapointPtr<T>& query,
                                       NNResultsVector* result) const override;

  absl::StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<T>& query,
                                NNResultsVector* result) const override;

  StatusOrPtr<SingleMachineSearcherBase<T>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const final;

  bool owns_mutation_data_structures() const override { return false; }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const final {
    for (auto j : Seq((*exact_reordering_dataset_)[i].dimensionality()))
      output[j] = (*exact_reordering_dataset_)[i].values_span()[j];
    return OkStatus();
  }

  shared_ptr<const Dataset> dataset() const final {
    return exact_reordering_dataset_;
  }

 private:
  shared_ptr<const DistanceMeasure> exact_reordering_distance_ = nullptr;

  shared_ptr<const TypedDataset<T>> exact_reordering_dataset_ = nullptr;
};

class FixedPointFloatDenseDotProductReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseDotProductReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f,
      float noise_shaping_threshold = NAN, ThreadPool* pool = nullptr);

  explicit FixedPointFloatDenseDotProductReorderingHelper(
      shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
      absl::Span<const float> multiplier_by_dimension,
      float noise_shaping_threshold = NAN);

  ~FixedPointFloatDenseDotProductReorderingHelper() override;

  std::string name() const override {
    return "FixedPointFloatDenseDotProductReordering";
  }

  bool needs_dataset() const override { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  StatusOrPtr<SingleMachineSearcherBase<float>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const final;

  template <typename CallbackFunctor>
  Status ComputeDistancesForReordering(
      const DatapointPtr<float>& query, NNResultsVector* result,
      CallbackFunctor* __restrict__ callback) const;

  absl::StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<float>& query,
                                NNResultsVector* result) const override;

  DimensionIndex dimensionality() const {
    return fixed_point_dataset_->dimensionality();
  }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const;

  shared_ptr<const Dataset> dataset() const final {
    return fixed_point_dataset_;
  }

  class Mutator;
  StatusOr<ReorderingInterface<float>::Mutator*> GetMutator() const override;

  void AppendDataToSingleMachineFactoryOptions(
      SingleMachineFactoryOptions* opts) const override {
    opts->pre_quantized_fixed_point =
        make_shared<PreQuantizedFixedPoint>(CreatePreQuantizedFixedPoint(
            *fixed_point_dataset_, *inverse_multipliers_, {}, true));
  }

 private:
  shared_ptr<DenseDataset<int8_t>> fixed_point_dataset_;
  shared_ptr<const vector<float>> inverse_multipliers_;
  const float noise_shaping_threshold_ = NAN;
  mutable unique_ptr<Mutator> mutator_;

  friend class FixedPointFloatDenseSquaredL2ReorderingHelper;
  friend class FixedPointFloatDenseCosineReorderingHelper;
};

class FixedPointFloatDenseCosineReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseCosineReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f,
      float noise_shaping_threshold = NAN, ThreadPool* pool = nullptr);

  explicit FixedPointFloatDenseCosineReorderingHelper(
      shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
      absl::Span<const float> multiplier_by_dimension,
      float noise_shaping_threshold = NAN);

  ~FixedPointFloatDenseCosineReorderingHelper() override;

  std::string name() const override {
    return "FixedPointFloatCosineReordering";
  }

  bool needs_dataset() const override { return false; }

  StatusOrPtr<SingleMachineSearcherBase<float>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const final;

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  absl::StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<float>& query,
                                NNResultsVector* result) const override;

  class Mutator;
  StatusOr<ReorderingInterface<float>::Mutator*> GetMutator() const override;

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const final {
    return dot_product_helper_.Reconstruct(i, output);
  }

  shared_ptr<const Dataset> dataset() const final {
    return dot_product_helper_.dataset();
  }

  void AppendDataToSingleMachineFactoryOptions(
      SingleMachineFactoryOptions* opts) const override {
    dot_product_helper_.AppendDataToSingleMachineFactoryOptions(opts);
  }

 private:
  FixedPointFloatDenseDotProductReorderingHelper dot_product_helper_;

  mutable unique_ptr<Mutator> mutator_;
  friend class Mutator;
};

class FixedPointFloatDenseSquaredL2ReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseSquaredL2ReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f);

  FixedPointFloatDenseSquaredL2ReorderingHelper(
      shared_ptr<DenseDataset<int8_t>> fixed_point_dataset,
      absl::Span<const float> multiplier_by_dimension,
      shared_ptr<const std::vector<float>> squared_l2_norm_by_datapoint);

  std::string name() const override {
    return "FixedPointFloatSquaredL2Reordering";
  }

  bool needs_dataset() const override { return false; }

  StatusOrPtr<SingleMachineSearcherBase<float>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const final;

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  absl::StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<float>& query,
                                NNResultsVector* result) const override;

  DimensionIndex dimensionality() const {
    return dot_product_helper_.dimensionality();
  }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const {
    return dot_product_helper_.Reconstruct(i, output);
  }

  shared_ptr<const Dataset> dataset() const final {
    return dot_product_helper_.dataset();
  }

  void AppendDataToSingleMachineFactoryOptions(
      SingleMachineFactoryOptions* opts) const override {
    dot_product_helper_.AppendDataToSingleMachineFactoryOptions(opts);
    opts->pre_quantized_fixed_point->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(database_squared_l2_norms_->begin(),
                                   database_squared_l2_norms_->end());
  }

 private:
  FixedPointFloatDenseDotProductReorderingHelper dot_product_helper_;

  shared_ptr<const std::vector<float>> database_squared_l2_norms_;
};

class FixedPointFloatDenseLimitedInnerReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseLimitedInnerReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f);

  std::string name() const override {
    return "FixedPointFloatDenseLimitedInnerReordering";
  }

  bool needs_dataset() const override { return false; }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const final {
    return dot_product_helper_.Reconstruct(i, output);
  }

  shared_ptr<const Dataset> dataset() const final {
    return dot_product_helper_.dataset();
  }

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  absl::StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<float>& query,
                                NNResultsVector* result) const override;

 private:
  FixedPointFloatDenseDotProductReorderingHelper dot_product_helper_;

  std::vector<float> inverse_database_l2_norms_;
};

template <bool kIsDotProduct>
class Bfloat16ReorderingHelper : public ReorderingHelper<float> {
 public:
  explicit Bfloat16ReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float noise_shaping_threshold = NAN, ThreadPool* pool = nullptr);

  explicit Bfloat16ReorderingHelper(
      shared_ptr<DenseDataset<int16_t>> bfloat16_dataset,
      float noise_shaping_threshold = NAN);

  ~Bfloat16ReorderingHelper() override;

  std::string name() const override {
    if constexpr (kIsDotProduct) {
      return "Bfloat16DenseDotProductReordering";
    } else {
      return "Bfloat16DenseSquaredL2Reordering";
    }
  }

  bool needs_dataset() const override { return false; }

  StatusOrPtr<SingleMachineSearcherBase<float>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const final;

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  DimensionIndex dimensionality() const {
    return bfloat16_dataset_->dimensionality();
  }

  class Mutator;
  StatusOr<ReorderingInterface<float>::Mutator*> GetMutator() const override;

  void AppendDataToSingleMachineFactoryOptions(
      SingleMachineFactoryOptions* opts) const override {
    opts->bfloat16_dataset = bfloat16_dataset_;
  }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const final;
  shared_ptr<const Dataset> dataset() const final;

 private:
  shared_ptr<DenseDataset<int16_t>> bfloat16_dataset_;
  const float noise_shaping_threshold_ = NAN;
  mutable unique_ptr<Mutator> mutator_;
};

using Bfloat16DenseDotProductReorderingHelper = Bfloat16ReorderingHelper<true>;
using Bfloat16DenseSquaredL2ReorderingHelper = Bfloat16ReorderingHelper<false>;

extern template class Bfloat16ReorderingHelper<true>;
extern template class Bfloat16ReorderingHelper<false>;

SCANN_INSTANTIATE_TYPED_CLASS(extern, ExactReorderingHelper);

}  // namespace research_scann

#endif
