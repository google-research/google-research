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

#include <algorithm>

#include "absl/strings/substitute.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOr<unique_ptr<typename DenseDataset<T>::Mutator>>
DenseDataset<T>::Mutator::Create(DenseDataset<T>* dataset) {
  SCANN_ASSIGN_OR_RETURN(auto mutator, dataset->docids()->GetMutator());
  return make_unique<Mutator>(Mutator(dataset, mutator));
}

template <typename T>
bool DenseDataset<T>::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  return docid_mutator_->LookupDatapointIndex(docid, index);
}

template <typename T>
void DenseDataset<T>::Mutator::Reserve(size_t size) {
  docid_mutator_->Reserve(size);
  dataset_->ReserveImpl(size);
}

template <typename T>
Status DenseDataset<T>::Mutator::AddDatapoint(const DatapointPtr<T>& dptr,
                                              string_view docid) {
  return dataset_->Append(dptr, docid);
}

template <typename T>
Status DenseDataset<T>::Mutator::RemoveDatapoint(DatapointIndex index) {
  if (index >= dataset_->size()) {
    return OutOfRangeError(
        "Removing a datapoint out of bound: index = %d, but size() = %d.",
        index, dataset_->size());
  }

  std::copy(
      dataset_->data_.begin() + (dataset_->size() - 1) * dataset_->stride_,
      dataset_->data_.begin() + dataset_->size() * dataset_->stride_,
      dataset_->data_.begin() + index * dataset_->stride_);
  dataset_->data_.resize((dataset_->size() - 1) * dataset_->stride_);

  CHECK_OK(docid_mutator_->RemoveDatapoint(index));
  return OkStatus();
}

template <typename T>
Status DenseDataset<T>::Mutator::RemoveDatapoint(string_view docid) {
  DatapointIndex index;
  if (!LookupDatapointIndex(docid, &index)) {
    return NotFoundError("Docid: %s is not found.", docid);
  }
  return RemoveDatapoint(index);
}

template <typename T>
Status DenseDataset<T>::Mutator::UpdateDatapoint(const DatapointPtr<T>& dptr,
                                                 string_view docid) {
  DatapointIndex index;
  if (!LookupDatapointIndex(docid, &index)) {
    return NotFoundError("Docid: %s is not found.", docid);
  }
  return UpdateDatapoint(dptr, index);
}

template <typename T>
Status DenseDataset<T>::Mutator::UpdateDatapoint(const DatapointPtr<T>& dptr,
                                                 DatapointIndex index) {
  if (dptr.dimensionality() != dataset_->dimensionality()) {
    return InvalidArgumentError(
        absl::Substitute("Dimensionality mismatch ($0 vs. $1)",
                         dptr.dimensionality(), dataset_->dimensionality()));
  }

  Datapoint<T> dp;
  CopyToDatapoint(dptr, &dp);
  SCANN_RETURN_IF_ERROR(NormalizeByTag(dataset_->normalization(), &dp));

  std::copy(dp.values().begin(), dp.values().end(),
            dataset_->data_.begin() + index * dataset_->stride_);
  return OkStatus();
}

template <typename T>
StatusOr<Datapoint<T>> DenseDataset<T>::Mutator::GetDatapoint(
    DatapointIndex index) const {
  if (index >= dataset_->size()) {
    return OutOfRangeError(
        "Datapoint index out of bound: index = %d, but size = %d.", index,
        dataset_->size());
  }

  auto dptr = dataset_->at(index);
  Datapoint<T> dp;
  CopyToDatapoint(dptr, &dp);
  return dp;
}

SCANN_INSTANTIATE_TYPED_CLASS(, DenseDataset);

}  // namespace research_scann
