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

#include <cmath>
#include <cstdint>
#include <utility>

#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/docid_collection.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher::Mutator>>
ScalarQuantizedBruteForceSearcher::Mutator::Create(
    ScalarQuantizedBruteForceSearcher* searcher) {
  const_cast<DenseDataset<int8_t>*>(searcher->quantized_dataset_.get())
      ->ReleaseDocids();

  SCANN_ASSIGN_OR_RETURN(auto quantized_dataset_mutator,
                         searcher->quantized_dataset_->GetMutator());
  ConstSpan<float> inverse_multipliers =
      *searcher->inverse_multiplier_by_dimension_;
  vector<float> multipliers(inverse_multipliers.size());
  for (auto i : Seq(multipliers.size())) {
    multipliers[i] = 1.0f / inverse_multipliers[i];
  }
  if (!searcher->docids()) {
    SCANN_RETURN_IF_ERROR(
        searcher->set_docids(make_unique<VariableLengthDocidCollection>(
            VariableLengthDocidCollection::CreateWithEmptyDocids(
                searcher->quantized_dataset_->size()))));
  }

  return absl::WrapUnique<ScalarQuantizedBruteForceSearcher::Mutator>(
      new ScalarQuantizedBruteForceSearcher::Mutator(
          searcher, quantized_dataset_mutator, std::move(multipliers)));
}

void ScalarQuantizedBruteForceSearcher::Mutator::Reserve(size_t size) {
  quantized_dataset_mutator_->Reserve(size);
  if (searcher_->distance_->specially_optimized_distance_tag() ==
      DistanceMeasure::SQUARED_L2) {
    if (!searcher_->squared_l2_norms_) {
      searcher_->squared_l2_norms_ = make_shared<vector<float>>();
    }
    searcher_->squared_l2_norms_->reserve(size);
  }
}

DatapointPtr<int8_t> ScalarQuantizedBruteForceSearcher::Mutator::ScalarQuantize(
    const DatapointPtr<float>& dptr) {
  if (std::isnan(searcher_->opts_.noise_shaping_threshold)) {
    return ScalarQuantizeFloatDatapoint(dptr, multipliers_,
                                        &quantized_datapoint_);
  } else {
    return ScalarQuantizeFloatDatapointWithNoiseShaping(
        dptr, multipliers_, searcher_->opts_.noise_shaping_threshold,
        &quantized_datapoint_);
  }
}

StatusOr<DatapointIndex>
ScalarQuantizedBruteForceSearcher::Mutator::AddDatapoint(
    const DatapointPtr<float>& dptr, string_view docid,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForAdd(dptr, docid, mo));
  const DatapointIndex result = searcher_->quantized_dataset_->size();
  SCANN_RETURN_IF_ERROR(
      quantized_dataset_mutator_->AddDatapoint(ScalarQuantize(dptr), ""));
  if (searcher_->distance_->specially_optimized_distance_tag() ==
      DistanceMeasure::SQUARED_L2) {
    SCANN_RET_CHECK(searcher_->squared_l2_norms_);
    searcher_->squared_l2_norms_->push_back(SquaredL2Norm(dptr));
  }
  SCANN_ASSIGN_OR_RETURN(
      auto result2, this->AddDatapointToBase(dptr, docid, MutateBaseOptions{}));
  SCANN_RET_CHECK_EQ(result, result2);
  return result;
}

Status ScalarQuantizedBruteForceSearcher::Mutator::RemoveDatapoint(
    DatapointIndex index) {
  SCANN_RETURN_IF_ERROR(this->ValidateForRemove(index));
  SCANN_RETURN_IF_ERROR(quantized_dataset_mutator_->RemoveDatapoint(index));
  if (searcher_->distance_->specially_optimized_distance_tag() ==
      DistanceMeasure::SQUARED_L2) {
    std::swap((*searcher_->squared_l2_norms_)[index],
              searcher_->squared_l2_norms_->back());
    searcher_->squared_l2_norms_->pop_back();
  }
  SCANN_ASSIGN_OR_RETURN(auto swapped_from,
                         this->RemoveDatapointFromBase(index));
  SCANN_RET_CHECK_EQ(swapped_from, searcher_->quantized_dataset_->size());
  OnDatapointIndexRename(swapped_from, index);
  return OkStatus();
}

Status ScalarQuantizedBruteForceSearcher::Mutator::RemoveDatapoint(
    string_view docid) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return RemoveDatapoint(index);
}

StatusOr<DatapointIndex>
ScalarQuantizedBruteForceSearcher::Mutator::UpdateDatapoint(
    const DatapointPtr<float>& dptr, string_view docid,
    const MutationOptions& mo) {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex index,
                         LookupDatapointIndexOrError(docid));
  return UpdateDatapoint(dptr, index, mo);
}

StatusOr<DatapointIndex>
ScalarQuantizedBruteForceSearcher::Mutator::UpdateDatapoint(
    const DatapointPtr<float>& dptr, DatapointIndex index,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForUpdate(dptr, index, mo));

  const bool mutate_values_vector = true;
  if (mutate_values_vector) {
    SCANN_RETURN_IF_ERROR(quantized_dataset_mutator_->UpdateDatapoint(
        ScalarQuantize(dptr), index));
    if (searcher_->distance_->specially_optimized_distance_tag() ==
        DistanceMeasure::SQUARED_L2) {
      (*searcher_->squared_l2_norms_)[index] = SquaredL2Norm(dptr);
    }
  }
  SCANN_RETURN_IF_ERROR(
      this->UpdateDatapointInBase(dptr, index, MutateBaseOptions{}));
  return index;
}

StatusOr<DatapointIndex>
ScalarQuantizedBruteForceSearcher::Mutator::LookupDatapointIndexOrError(
    string_view docid) const {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("DocId: ", docid, " is not found."));
  }
  return index;
}

}  // namespace research_scann
