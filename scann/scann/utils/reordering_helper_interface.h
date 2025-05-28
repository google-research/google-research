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


#ifndef SCANN_UTILS_REORDERING_HELPER_INTERFACE_H_
#define SCANN_UTILS_REORDERING_HELPER_INTERFACE_H_

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

template <typename T>
class ReorderingInterface {
 public:
  class Mutator;
  virtual std::string name() const = 0;
  virtual bool needs_dataset() const = 0;
  virtual Status ComputeDistancesForReordering(
      const DatapointPtr<T>& query, NNResultsVector* result) const = 0;

  virtual absl::StatusOr<std::pair<DatapointIndex, float>>
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

  virtual StatusOrPtr<SingleMachineSearcherBase<T>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const = 0;

  virtual bool owns_mutation_data_structures() const = 0;

  virtual void AppendDataToSingleMachineFactoryOptions(
      SingleMachineFactoryOptions* opts) const {}

  virtual Status Reconstruct(DatapointIndex idx,
                             MutableSpan<float> output) const = 0;

  virtual shared_ptr<const Dataset> dataset() const { return nullptr; }

  virtual StatusOr<shared_ptr<const DenseDataset<float>>>
  ReconstructFloatDataset() const {
    if (dataset() == nullptr) {
      return FailedPreconditionError(
          "Cannot reconstruct float dataset if reordering helper does not own "
          "a dataset.");
    }
    auto reconstructed = make_shared<DenseDataset<float>>();
    auto dim = dataset()->dimensionality();
    Datapoint<float> dp;
    dp.mutable_values()->resize(dim);
    for (auto idx : Seq(dataset()->size())) {
      SCANN_RETURN_IF_ERROR(Reconstruct(idx, dp.mutable_values_span()));
      SCANN_RETURN_IF_ERROR(reconstructed->Append(dp.ToPtr()));
    }
    return reconstructed;
  }

  virtual ~ReorderingInterface() = default;
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

  StatusOrPtr<SingleMachineSearcherBase<T>> CreateBruteForceSearcher(
      int32_t num_neighbors, float epsilon) const override;

  bool owns_mutation_data_structures() const override { return true; }
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, ReorderingHelper);

}  // namespace research_scann

#endif
