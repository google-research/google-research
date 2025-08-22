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

#ifndef SCANN_BRUTE_FORCE_BFLOAT16_BRUTE_FORCE_H_
#define SCANN_BRUTE_FORCE_BFLOAT16_BRUTE_FORCE_H_

#include <cmath>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class Bfloat16BruteForceSearcher final
    : public SingleMachineSearcherBase<float> {
 public:
  Bfloat16BruteForceSearcher(shared_ptr<const DistanceMeasure> distance,
                             shared_ptr<const DenseDataset<float>> dataset,
                             int32_t default_num_neighbors,
                             float default_epsilon,
                             float noise_shaping_threshold = NAN);

  Bfloat16BruteForceSearcher(
      shared_ptr<const DistanceMeasure> distance,
      shared_ptr<const DenseDataset<int16_t>> bfloat16_dataset,
      int32_t default_num_neighbors, float default_epsilon,
      float noise_shaping_threshold = NAN);

  StatusOr<const SingleMachineSearcherBase<float>*> CreateBruteForceSearcher(
      const DistanceMeasureConfig& distance_config,
      unique_ptr<SingleMachineSearcherBase<float>>* storage) const final;

  ~Bfloat16BruteForceSearcher() override = default;

  bool supports_crowding() const final { return true; }

  class Mutator : public SingleMachineSearcherBase<float>::Mutator {
   public:
    using PrecomputedMutationArtifacts =
        UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    static StatusOr<unique_ptr<Mutator>> Create(
        Bfloat16BruteForceSearcher* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final = default;

    StatusOr<Datapoint<float>> GetDatapoint(DatapointIndex i) const final;
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr,
                                          string_view docid,
                                          const MutationOptions&) final;
    Status RemoveDatapoint(string_view docid) final;
    void Reserve(size_t size) final;
    Status RemoveDatapoint(DatapointIndex index) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             string_view docid,
                                             const MutationOptions&) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions&) final;

   private:
    Mutator(Bfloat16BruteForceSearcher* searcher,
            TypedDataset<int16_t>::Mutator* quantized_dataset_mutator)
        : searcher_(searcher),
          quantized_dataset_mutator_(quantized_dataset_mutator) {}
    StatusOr<DatapointIndex> LookupDatapointIndexOrError(
        string_view docid) const;

    Bfloat16BruteForceSearcher* searcher_;
    TypedDataset<int16_t>::Mutator* quantized_dataset_mutator_;
  };

  StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
      const final;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

 protected:
  Status FindNeighborsImpl(const DatapointPtr<float>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
      ConstSpan<std::string> crowding_dimension_names) final;

 private:
  bool impl_needs_dataset() const override { return false; }

  bool is_dot_product_;
  shared_ptr<const DenseDataset<int16_t>> bfloat16_dataset_;

  const float noise_shaping_threshold_ = NAN;

  mutable unique_ptr<Mutator> mutator_ = nullptr;
};

}  // namespace research_scann

#endif
