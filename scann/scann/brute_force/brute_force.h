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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__BRUTE_FORCE_BRUTE_FORCE_H_
#define SCANN__BRUTE_FORCE_BRUTE_FORCE_H_

#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class BruteForceSearcher final : public SingleMachineSearcherBase<T> {
 public:
  BruteForceSearcher(shared_ptr<const DistanceMeasure> distance,
                     shared_ptr<const TypedDataset<T>> dataset,
                     const int32_t default_pre_reordering_num_neighbors,
                     const float default_pre_reordering_epsilon);

  ~BruteForceSearcher() override;

  bool supports_crowding() const final { return true; }

  DatapointIndex optimal_batch_size() const final {
    return supports_low_level_batching_ ? 128 : 1;
  }

  void set_thread_pool(std::shared_ptr<thread::ThreadPool> p) {
    pool_ = std::move(p);
  }

  using MutationMetadata = UntypedSingleMachineSearcherBase::MutationMetadata;

 protected:
  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute) final;

 private:
  template <typename TopN>
  void FindNeighborsInternal(const DatapointPtr<T>& query,
                             const SearchParameters& params,
                             TopN* top_n_ptr) const;

  template <typename WhitelistIterator, typename TopN>
  void FindNeighborsOneToOneInternal(const DatapointPtr<T>& query,
                                     const SearchParameters& params,
                                     WhitelistIterator* whitelist_iterator,
                                     TopN* top_n_ptr) const;

  template <typename Float>
  enable_if_t<IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  template <typename Float>
  enable_if_t<!IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  shared_ptr<const DistanceMeasure> distance_;

  const bool supports_low_level_batching_;

  const bool using_squared_db_norms_;

  vector<T> squared_db_norms_;

  std::shared_ptr<thread::ThreadPool> pool_;

  mutable unique_ptr<typename BruteForceSearcher<T>::Mutator> mutator_ =
      nullptr;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, BruteForceSearcher);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
