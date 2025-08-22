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



#ifndef SCANN_BRUTE_FORCE_BRUTE_FORCE_H_
#define SCANN_BRUTE_FORCE_BRUTE_FORCE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class BruteForceSearcher final : public SingleMachineSearcherBase<T> {
 public:
  BruteForceSearcher(shared_ptr<const DistanceMeasure> distance,
                     shared_ptr<const TypedDataset<T>> dataset,
                     const int32_t default_pre_reordering_num_neighbors,
                     const float default_pre_reordering_epsilon);

  ~BruteForceSearcher() final;

  bool supports_crowding() const final { return true; }

  DatapointIndex optimal_batch_size() const final {
    return supports_low_level_batching_ ? 128 : 1;
  }

  void set_thread_pool(std::shared_ptr<ThreadPool> p) { pool_ = std::move(p); }

  void set_min_distance(float min_distance) { min_distance_ = min_distance; }

  StatusOr<const SingleMachineSearcherBase<T>*> CreateBruteForceSearcher(
      const DistanceMeasureConfig&,
      unique_ptr<SingleMachineSearcherBase<T>>* storage) const final {
    return this;
  }

  using PrecomputedMutationArtifacts =
      UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;

  class Mutator : public SingleMachineSearcherBase<T>::Mutator {
   public:
    using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    static StatusOr<unique_ptr<typename BruteForceSearcher<T>::Mutator>> Create(
        BruteForceSearcher<T>* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final {}
    absl::StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex i) const final;
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                          string_view docid,
                                          const MutationOptions& mo) final;
    Status RemoveDatapoint(string_view docid) final;
    void Reserve(size_t size) final;
    Status RemoveDatapoint(DatapointIndex index) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             string_view docid,
                                             const MutationOptions& mo) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions& mo) final;

    StatusOr<std::optional<ScannConfig>> IncrementalMaintenance() final;

   private:
    explicit Mutator(BruteForceSearcher<T>* searcher) : searcher_(searcher) {}

    StatusOr<DatapointIndex> LookupDatapointIndexOrError(
        string_view docid) const;

    BruteForceSearcher<T>* searcher_;
  };

  StatusOr<typename SingleMachineSearcherBase<T>::Mutator*> GetMutator()
      const final;

 protected:
  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<FastTopNeighbors<float>*> results,
      ConstSpan<DatapointIndex> datapoint_index_mapping) const final;

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
      ConstSpan<std::string> crowding_dimension_names) final;

 private:
  template <bool kUseMinDistance, typename TopN>
  Status FindNeighborsInternal(const DatapointPtr<T>& query,
                               const SearchParameters& params,
                               TopN* top_n_ptr) const;

  template <bool kUseMinDistance, typename WhitelistIterator, typename TopN>
  void FindNeighborsOneToOneInternal(const DatapointPtr<T>& query,
                                     const SearchParameters& params,
                                     WhitelistIterator* allowlist_iterator,
                                     TopN* top_n_ptr) const;

  template <typename Float>
  enable_if_t<IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  void FinishBatchedSearchSimple(const DenseDataset<float>& db,
                                 const DenseDataset<float>& queries,
                                 ConstSpan<SearchParameters> params,
                                 MutableSpan<NNResultsVector> results) const;

  template <typename Float>
  enable_if_t<!IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  shared_ptr<const DistanceMeasure> distance_;

  const bool supports_low_level_batching_;

  std::shared_ptr<ThreadPool> pool_;

  float min_distance_ = -numeric_limits<float>::infinity();

  mutable unique_ptr<typename BruteForceSearcher<T>::Mutator> mutator_ =
      nullptr;

  bool is_immutable_ = false;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, BruteForceSearcher);

}  // namespace research_scann

#endif
