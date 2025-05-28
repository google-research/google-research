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

#include "scann/brute_force/bfloat16_brute_force.h"

#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/one_to_many/one_to_many_asymmetric.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/bfloat16_helpers.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"

namespace research_scann {

Bfloat16BruteForceSearcher::Bfloat16BruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const DenseDataset<float>> dataset,
    const int32_t default_num_neighbors, const float default_epsilon,
    float noise_shaping_threshold)
    : SingleMachineSearcherBase<float>(dataset, default_num_neighbors,
                                       default_epsilon),
      is_dot_product_(distance->specially_optimized_distance_tag() ==
                      distance->DOT_PRODUCT),
      noise_shaping_threshold_(noise_shaping_threshold) {
  if (distance->specially_optimized_distance_tag() != distance->DOT_PRODUCT &&
      distance->specially_optimized_distance_tag() != distance->SQUARED_L2) {
    LOG(FATAL) << "Bfloat16 brute force only supports dot product and squared "
                  "L2 distance.";
  }
  if (std::isfinite(noise_shaping_threshold)) {
    bfloat16_dataset_ = make_shared<DenseDataset<int16_t>>(
        Bfloat16QuantizeFloatDatasetWithNoiseShaping(*dataset,
                                                     noise_shaping_threshold));
  } else {
    bfloat16_dataset_ = make_shared<DenseDataset<int16_t>>(
        Bfloat16QuantizeFloatDataset(*dataset));
  }
}
Bfloat16BruteForceSearcher::Bfloat16BruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const DenseDataset<int16_t>> bfloat16_dataset,
    const int32_t default_num_neighbors, const float default_epsilon,
    float noise_shaping_threshold)
    : SingleMachineSearcherBase<float>(nullptr, default_num_neighbors,
                                       default_epsilon),
      is_dot_product_(distance->specially_optimized_distance_tag() ==
                      distance->DOT_PRODUCT),
      bfloat16_dataset_(std::move(bfloat16_dataset)),
      noise_shaping_threshold_(noise_shaping_threshold) {
  if (distance->specially_optimized_distance_tag() != distance->DOT_PRODUCT &&
      distance->specially_optimized_distance_tag() != distance->SQUARED_L2) {
    LOG(FATAL) << "Bfloat16 brute force only supports dot product and squared "
                  "L2 distance.";
  }
  QCHECK_OK(this->set_docids(bfloat16_dataset_->docids()));
}

Status Bfloat16BruteForceSearcher::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  size_t sz1 = datapoint_index_to_crowding_attribute.size();
  size_t sz2 = bfloat16_dataset_->size();
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
  if (query.dimensionality() != bfloat16_dataset_->dimensionality()) {
    return FailedPreconditionError(
        "Query/database dimensionality mismatch: %d vs %d.",
        query.dimensionality(), bfloat16_dataset_->dimensionality());
  }
  if (params.pre_reordering_crowding_enabled() && !this->crowding_enabled()) {
    return FailedPreconditionError(
        "Received query with pre-reordering crowding enabled, but crowding "
        "isn't enabled in this bfloat16 brute-force searcher instance.");
  }

  if (params.restricts_enabled()) {
    return UnimplementedError("Restricts not supported.");
  } else {
    auto dists_ptr =
        static_cast<float*>(malloc(bfloat16_dataset_->size() * sizeof(float)));
    MutableSpan<float> dists_span(dists_ptr, bfloat16_dataset_->size());
    if (is_dot_product_) {
      DenseDotProductDistanceOneToManyBf16Float(query, *bfloat16_dataset_,
                                                dists_span);
    } else {
      OneToManyBf16FloatSquaredL2(query, *bfloat16_dataset_, dists_span);
    }
    if (params.pre_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      FastTopNeighbors<float> top_n(params.pre_reordering_num_neighbors(),
                                    params.pre_reordering_epsilon());
      top_n.PushBlock(dists_span, 0);
      top_n.FinishUnsorted(result);
    }
    free(dists_ptr);
  }
  return OkStatus();
}

StatusOr<const SingleMachineSearcherBase<float>*>
Bfloat16BruteForceSearcher::CreateBruteForceSearcher(
    const DistanceMeasureConfig& distance_config,
    unique_ptr<SingleMachineSearcherBase<float>>* storage) const {
  auto base_result = SingleMachineSearcherBase<float>::CreateBruteForceSearcher(
      distance_config, storage);
  if (base_result.ok()) {
    return base_result;
  }
  return this;
}

StatusOr<SingleMachineSearcherBase<float>::Mutator*>
Bfloat16BruteForceSearcher::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<Bfloat16BruteForceSearcher*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_, Bfloat16BruteForceSearcher::Mutator::Create(mutable_this));
    SCANN_RETURN_IF_ERROR(mutator_->PrepareForBaseMutation(mutable_this));
  }
  return static_cast<SingleMachineSearcherBase::Mutator*>(mutator_.get());
}

StatusOr<SingleMachineFactoryOptions>
Bfloat16BruteForceSearcher::ExtractSingleMachineFactoryOptions() {
  SCANN_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<float>::ExtractSingleMachineFactoryOptions());
  opts.bfloat16_dataset =
      make_shared<DenseDataset<int16_t>>(bfloat16_dataset_->Copy());
  return opts;
}

}  // namespace research_scann
