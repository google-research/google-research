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

#include <optional>

#include "absl/strings/str_cat.h"
#include "scann/brute_force/brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOr<unique_ptr<typename BruteForceSearcher<T>::Mutator>>
BruteForceSearcher<T>::Mutator::Create(BruteForceSearcher<T>* searcher) {
  return absl::WrapUnique<typename BruteForceSearcher<T>::Mutator>(
      new typename BruteForceSearcher<T>::Mutator(searcher));
}

template <typename T>
void BruteForceSearcher<T>::Mutator::Reserve(size_t size) {
  this->ReserveInBase(size);
}

template <typename T>
StatusOr<Datapoint<T>> BruteForceSearcher<T>::Mutator::GetDatapoint(
    DatapointIndex i) const {
  return this->GetDatapointFromBase(i);
}

template <typename T>
StatusOr<DatapointIndex> BruteForceSearcher<T>::Mutator::AddDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForAdd(dptr, docid, mo));
  SCANN_ASSIGN_OR_RETURN(
      const DatapointIndex result,
      this->AddDatapointToBase(dptr, docid, MutateBaseOptions{}));
  return result;
}

template <typename T>
Status BruteForceSearcher<T>::Mutator::RemoveDatapoint(DatapointIndex index) {
  SCANN_RETURN_IF_ERROR(this->ValidateForRemove(index));
  SCANN_ASSIGN_OR_RETURN(const DatapointIndex swapped_in,
                         this->RemoveDatapointFromBase(index));
  this->OnDatapointIndexRename(swapped_in, index);
  return OkStatus();
}

template <typename T>
Status BruteForceSearcher<T>::Mutator::RemoveDatapoint(string_view docid) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return RemoveDatapoint(index);
}

template <typename T>
StatusOr<DatapointIndex> BruteForceSearcher<T>::Mutator::UpdateDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return UpdateDatapoint(dptr, index, mo);
}

template <typename T>
StatusOr<DatapointIndex> BruteForceSearcher<T>::Mutator::UpdateDatapoint(
    const DatapointPtr<T>& dptr, DatapointIndex index,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForUpdate(dptr, index, mo));
  SCANN_RETURN_IF_ERROR(
      this->UpdateDatapointInBase(dptr, index, MutateBaseOptions{}));
  return index;
}

template <typename T>
StatusOr<DatapointIndex>
BruteForceSearcher<T>::Mutator::LookupDatapointIndexOrError(
    string_view docid) const {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  auto ds = searcher_->dataset();
  SCANN_RET_CHECK(ds)
      << "Dataset is null in BruteForceSearcher.  This is likely an "
         "internal error.";
  SCANN_RET_CHECK_LT(index, ds->size())
      << "Docid: " << docid << " has an invalid (too-large) datapoint index.";
  return index;
}

template <typename T>
StatusOr<std::optional<ScannConfig>>
BruteForceSearcher<T>::Mutator::IncrementalMaintenance() {
  if (searcher_->config().has_value() &&
      searcher_->config().value().has_autopilot()) {
    auto shared_dataset = searcher_->shared_dataset()
                              ? searcher_->shared_dataset()
                              : searcher_->reordering_helper().dataset();
    SCANN_ASSIGN_OR_RETURN(
        auto config,
        Autopilot(this->searcher_->config().value(), shared_dataset,
                  kInvalidDatapointIndex, kInvalidDimension));

    if (!config.has_brute_force()) return config;
  }
  return std::nullopt;
}

SCANN_INSTANTIATE_TYPED_CLASS(, BruteForceSearcher);

}  // namespace research_scann
