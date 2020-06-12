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



#include "scann/partitioning/partitioner_base.h"

#include "absl/base/internal/spinlock.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace tensorflow {
namespace scann_ops {

UntypedPartitioner::~UntypedPartitioner() {}

void UntypedPartitioner::set_training_parallelization_pool(
    shared_ptr<thread::ThreadPool> pool) {}

template <typename T>
Status Partitioner<T>::TokenForDatapointBatched(const TypedDataset<T>& queries,
                                                vector<int32_t>* results,
                                                thread::ThreadPool*) const {
  DCHECK(results);
  results->resize(queries.size());
  for (DatapointIndex i = 0; i < queries.size(); ++i) {
    SCANN_RETURN_IF_ERROR(TokenForDatapoint(queries[i], &results->at(i)));
  }

  return OkStatus();
}

template <typename T>
Status Partitioner<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries,
    MutableSpan<vector<int32_t>> results) const {
  if (results.size() != queries.size()) {
    return InvalidArgumentError(
        absl::StrCat("queries.size must be equal to results.size.  (",
                     queries.size(), " vs. ", results.size(), ")"));
  }

  for (DatapointIndex i = 0; i < queries.size(); ++i) {
    SCANN_RETURN_IF_ERROR(
        TokensForDatapointWithSpilling(queries[i], &results[i]));
  }

  return OkStatus();
}

template <typename T>
StatusOr<vector<std::vector<DatapointIndex>>> Partitioner<T>::TokenizeDatabase(
    const TypedDataset<T>& database, thread::ThreadPool* pool_or_null) const {
  if (tokenization_mode() != DATABASE) {
    return FailedPreconditionError(
        "Cannot run TokenizeDatabase when not in database tokenization mode.");
  }
  vector<std::vector<DatapointIndex>> datapoints_by_token(this->n_tokens());

  absl::base_internal::SpinLock tokenization_status_spinlock;
  Status tokenization_status = OkStatus();

  constexpr size_t kNumTokenizationSpinlocks = 128;

  std::array<absl::base_internal::SpinLock, kNumTokenizationSpinlocks>
      tokenization_spinlocks;

  ParallelFor<kDynamicBatchSize>(
      Seq(database.size()), pool_or_null, [&](size_t i) {
        const DatapointPtr<T> dptr = database[i];
        vector<int32_t> tokens_for_datapoint;
        Status status =
            TokensForDatapointWithSpilling(dptr, &tokens_for_datapoint);

        if (!status.ok()) {
          absl::base_internal::SpinLockHolder lock(
              &tokenization_status_spinlock);
          if (tokenization_status.ok()) tokenization_status = status;
        }
        for (int32_t token : tokens_for_datapoint) {
          if (pool_or_null) {
            absl::base_internal::SpinLockHolder lock(
                &tokenization_spinlocks[token % kNumTokenizationSpinlocks]);
            datapoints_by_token[token].push_back(i);
          } else {
            datapoints_by_token[token].push_back(i);
          }
        }
      });

  if (pool_or_null) {
    ParallelFor<kDynamicBatchSize>(
        Seq(datapoints_by_token.size()), pool_or_null, [&](size_t i) {
          ZipSortBranchOptimized(datapoints_by_token[i].begin(),
                                 datapoints_by_token[i].end());
        });
  }
  return std::move(datapoints_by_token);
}

SCANN_INSTANTIATE_TYPED_CLASS(, Partitioner);

}  // namespace scann_ops
}  // namespace tensorflow
