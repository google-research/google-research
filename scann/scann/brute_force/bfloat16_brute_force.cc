// Copyright 2024 The Google Research Authors.
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

#include "scann/brute_force/bfloat16_brute_force.h"

#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

Bfloat16BruteForceSearcher::Bfloat16BruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const DenseDataset<float>> dataset,
    const int32_t default_num_neighbors, const float default_epsilon)
    : SingleMachineSearcherBase<float>(dataset, default_num_neighbors,
                                       default_epsilon),
      distance_(distance) {
  if (distance->specially_optimized_distance_tag() != distance->DOT_PRODUCT) {
    LOG(FATAL) << "Bfloat16 brute force only supports dot product distance.";
  }
  bfloat16_dataset_ = Bfloat16QuantizeFloatDataset(*dataset);
}
Bfloat16BruteForceSearcher::Bfloat16BruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    DenseDataset<int16_t> bfloat16_dataset, const int32_t default_num_neighbors,
    const float default_epsilon)
    : SingleMachineSearcherBase<float>(nullptr, default_num_neighbors,
                                       default_epsilon),
      distance_(distance),
      bfloat16_dataset_(std::move(bfloat16_dataset)) {
  if (distance->specially_optimized_distance_tag() != distance->DOT_PRODUCT) {
    LOG(FATAL) << "Bfloat16 brute force only supports dot product distance.";
  }
  TF_CHECK_OK(this->set_docids(bfloat16_dataset_.ReleaseDocids()));
}

Status Bfloat16BruteForceSearcher::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  size_t sz1 = datapoint_index_to_crowding_attribute.size();
  size_t sz2 = bfloat16_dataset_.size();
  if (sz1 != sz2) {
    return InvalidArgumentError(
        "Crowding attributes don't match dataset in size: %d vs %d.", sz1, sz2);
  }
  return OkStatus();
}

Status Bfloat16BruteForceSearcher::FindNeighborsImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  if (!query.IsDense()) {
    return InvalidArgumentError("Bfloat16 brute force requires dense data.");
  }
  if (query.dimensionality() != bfloat16_dataset_.dimensionality()) {
    return FailedPreconditionError(
        "Query/database dimensionality mismatch: %d vs %d.",
        query.dimensionality(), bfloat16_dataset_.dimensionality());
  }
  if (params.pre_reordering_crowding_enabled() && !this->crowding_enabled()) {
    return FailedPreconditionError(
        "Received query with pre-reordering crowding enabled, but crowding "
        "isn't enabled in this bfloat16 brute-force searcher instance.");
  }

  if (params.restricts_enabled()) {
    return UnimplementedError("Restricts not supported.");
  } else {
    auto dot_products_ptr =
        static_cast<float*>(malloc(bfloat16_dataset_.size() * sizeof(float)));
    MutableSpan<float> dot_products_span(dot_products_ptr,
                                         bfloat16_dataset_.size());
    DenseDotProductDistanceOneToManyBf16Float(query, bfloat16_dataset_,
                                              dot_products_span);
    if (params.pre_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      FastTopNeighbors<float> top_n(params.pre_reordering_num_neighbors(),
                                    params.pre_reordering_epsilon());
      top_n.PushBlock(dot_products_span, 0);
      top_n.FinishUnsorted(result);
    }
    free(dot_products_ptr);
  }
  return OkStatus();
}

StatusOr<SingleMachineSearcherBase<float>::Mutator*>
Bfloat16BruteForceSearcher::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<Bfloat16BruteForceSearcher*>(this);
    TF_ASSIGN_OR_RETURN(
        mutator_, Bfloat16BruteForceSearcher::Mutator::Create(mutable_this));
    SCANN_RETURN_IF_ERROR(mutator_->PrepareForBaseMutation(mutable_this));
  }
  return static_cast<SingleMachineSearcherBase::Mutator*>(mutator_.get());
}

}  // namespace research_scann
