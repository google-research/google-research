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


#ifndef SCANN__UTILS_REORDERING_HELPER_H_
#define SCANN__UTILS_REORDERING_HELPER_H_

#include <limits>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class ReorderingInterface {
 public:
  class Mutator;
  virtual std::string name() const = 0;
  virtual bool needs_dataset() const = 0;
  virtual Status ComputeDistancesForReordering(
      const DatapointPtr<T>& query, NNResultsVector* result) const = 0;

  virtual StatusOr<std::pair<DatapointIndex, float>>
  ComputeTop1ReorderingDistance(const DatapointPtr<T>& query,
                                NNResultsVector* result) const {
    SCANN_RETURN_IF_ERROR(ComputeDistancesForReordering(query, result));
    std::pair<DatapointIndex, float> best = {kInvalidDatapointIndex,
                                             std::numeric_limits<float>::max()};
    DistanceComparatorBranchOptimized comparator;
    for (const auto& neighbor : *result) {
      bool better_than_best = comparator(neighbor, best);
      best.first = better_than_best ? neighbor.first : best.first;
      best.second = better_than_best ? neighbor.second : best.second;
    }
    return best;
  }

  virtual StatusOr<ReorderingInterface<T>::Mutator*> GetMutator() const = 0;

  virtual bool owns_mutation_data_structures() const = 0;

  virtual ~ReorderingInterface() {}
};

template <typename T>
class ReorderingInterface<T>::Mutator : public VirtualDestructor {
 public:
  virtual StatusOr<DatapointIndex> AddDatapoint(
      const DatapointPtr<T>& dptr) = 0;

  virtual StatusOr<DatapointIndex> RemoveDatapoint(DatapointIndex idx) = 0;

  virtual Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                                 DatapointIndex idx) = 0;

  virtual void Reserve(DatapointIndex num_datapoints) {}
};

template <typename T>
class ReorderingHelper : public ReorderingInterface<T> {
 public:
  StatusOr<typename ReorderingInterface<T>::Mutator*> GetMutator()
      const override {
    return FailedPreconditionError(
        StrCat("Mutation not supported for reordering helper of type ",
               this->name(), "."));
  }
  bool owns_mutation_data_structures() const override { return true; }
};

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

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<T>& query, NNResultsVector* result) const override;

  bool owns_mutation_data_structures() const override { return false; }

 private:
  shared_ptr<const DistanceMeasure> exact_reordering_distance_ = nullptr;

  shared_ptr<const TypedDataset<T>> exact_reordering_dataset_ = nullptr;
};

template <typename T>
class CompressedReorderingHelper final : public ReorderingHelper<T> {
 public:
  CompressedReorderingHelper(
      unique_ptr<const asymmetric_hashing2::AsymmetricQueryer<T>>
          compressed_queryer,
      shared_ptr<const DenseDataset<uint8_t>> compressed_dataset,
      AsymmetricHasherConfig::LookupType lookup_type)
      : compressed_queryer_(std::move(compressed_queryer)),
        compressed_dataset_(std::move(compressed_dataset)) {}

  std::string name() const final { return "CompressedReordering"; }

  bool needs_dataset() const final { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<T>& query,
                                       NNResultsVector* result) const final;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<T>& query, NNResultsVector* result) const final;

 private:
  unique_ptr<const asymmetric_hashing2::AsymmetricQueryer<T>>
      compressed_queryer_ = nullptr;

  shared_ptr<const DenseDataset<uint8_t>> compressed_dataset_ = nullptr;

  AsymmetricHasherConfig::LookupType lookup_type_ =
      AsymmetricHasherConfig::FLOAT;
};

template <typename T>
class CompressedResidualReorderingHelper final : public ReorderingHelper<T> {
 public:
  CompressedResidualReorderingHelper(
      unique_ptr<const asymmetric_hashing2::AsymmetricQueryer<T>>
          compressed_queryer,
      shared_ptr<const DenseDataset<uint8_t>> compressed_dataset,
      AsymmetricHasherConfig::LookupType lookup_type)
      : compressed_queryer_(std::move(compressed_queryer)),
        compressed_dataset_(std::move(compressed_dataset)) {}

  std::string name() const final { return "CompressedResidualReordering"; }

  bool needs_dataset() const final { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<T>& query,
                                       NNResultsVector* result) const final;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<T>& query, NNResultsVector* result) const final;

 private:
  unique_ptr<const asymmetric_hashing2::AsymmetricQueryer<T>>
      compressed_queryer_ = nullptr;

  shared_ptr<const DenseDataset<uint8_t>> compressed_dataset_ = nullptr;

  AsymmetricHasherConfig::LookupType lookup_type_ =
      AsymmetricHasherConfig::FLOAT;
};

class FixedPointFloatDenseDotProductReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseDotProductReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f);

  FixedPointFloatDenseDotProductReorderingHelper(
      DenseDataset<int8_t> fixed_point_dataset,
      const shared_ptr<const std::vector<float>>& multiplier_by_dimension);

  ~FixedPointFloatDenseDotProductReorderingHelper() override;

  std::string name() const override {
    return "FixedPointFloatDenseDotProductReordering";
  }

  bool needs_dataset() const override { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  template <typename CallbackFunctor>
  Status ComputeDistancesForReordering(
      const DatapointPtr<float>& query, NNResultsVector* result,
      CallbackFunctor* __restrict__ callback) const;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<float>& query, NNResultsVector* result) const override;

  DimensionIndex dimensionality() const {
    return fixed_point_dataset_.dimensionality();
  }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const;

 private:
  DenseDataset<int8_t> fixed_point_dataset_;
  std::vector<float> inverse_multipliers_;

  friend class FixedPointFloatDenseSquaredL2ReorderingHelper;
};

class FixedPointFloatDenseCosineReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseCosineReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f);

  FixedPointFloatDenseCosineReorderingHelper(
      DenseDataset<int8_t> fixed_point_dataset,
      shared_ptr<const std::vector<float>> multiplier_by_dimension);

  ~FixedPointFloatDenseCosineReorderingHelper() override;

  std::string name() const override {
    return "FixedPointFloatCosineReordering";
  }

  bool needs_dataset() const override { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<float>& query, NNResultsVector* result) const override;

 private:
  FixedPointFloatDenseDotProductReorderingHelper dot_product_helper_;
};

class FixedPointFloatDenseSquaredL2ReorderingHelper
    : public ReorderingHelper<float> {
 public:
  explicit FixedPointFloatDenseSquaredL2ReorderingHelper(
      const DenseDataset<float>& exact_reordering_dataset,
      float fixed_point_multiplier_quantile = 1.0f);

  FixedPointFloatDenseSquaredL2ReorderingHelper(
      DenseDataset<int8_t> fixed_point_dataset,
      shared_ptr<const std::vector<float>> multiplier_by_dimension,
      shared_ptr<const std::vector<float>> squared_l2_norm_by_datapoint);

  std::string name() const override {
    return "FixedPointFloatSquaredL2Reordering";
  }

  bool needs_dataset() const override { return false; }

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<float>& query, NNResultsVector* result) const override;

  DimensionIndex dimensionality() const {
    return dot_product_helper_.dimensionality();
  }

  Status Reconstruct(DatapointIndex i, MutableSpan<float> output) const {
    return dot_product_helper_.Reconstruct(i, output);
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

  Status ComputeDistancesForReordering(const DatapointPtr<float>& query,
                                       NNResultsVector* result) const override;

  StatusOr<std::pair<DatapointIndex, float>> ComputeTop1ReorderingDistance(
      const DatapointPtr<float>& query, NNResultsVector* result) const override;

 private:
  FixedPointFloatDenseDotProductReorderingHelper dot_product_helper_;

  std::vector<float> inverse_database_l2_norms_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, ExactReorderingHelper);
SCANN_INSTANTIATE_TYPED_CLASS(extern, CompressedReorderingHelper);
SCANN_INSTANTIATE_TYPED_CLASS(extern, CompressedResidualReorderingHelper);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
