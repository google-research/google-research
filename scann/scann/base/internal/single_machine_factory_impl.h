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



#ifndef SCANN_BASE_INTERNAL_SINGLE_MACHINE_FACTORY_IMPL_H_
#define SCANN_BASE_INTERNAL_SINGLE_MACHINE_FACTORY_IMPL_H_

#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "scann/base/reordering_helper_factory.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/crowding.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/hash_leaf_helpers.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class DenseDataset;
template <typename T>
class TypedDataset;
class ScannConfig;
class ScannInputData;

using StatusOrSearcherUntyped =
    StatusOr<unique_ptr<UntypedSingleMachineSearcherBase>>;

namespace internal {

inline int NumQueryDatabaseSearchTypesConfigured(const ScannConfig& config) {
  return config.has_brute_force() + config.has_hash();
}

template <typename LeafSearcherT>
class SingleMachineFactoryImplClass {
 public:
  template <typename T>
  static StatusOrSearcherUntyped SingleMachineFactoryImpl(
      ScannConfig config, const shared_ptr<Dataset>& dataset,
      const GenericSearchParameters& params,
      SingleMachineFactoryOptions* opts) {
    config.mutable_input_output()->set_in_memory_data_type(TagForType<T>());
    SCANN_RETURN_IF_ERROR(CanonicalizeScannConfigForRetrieval(&config));
    auto typed_dataset = std::dynamic_pointer_cast<TypedDataset<T>>(dataset);
    if (dataset && !typed_dataset) {
      return InvalidArgumentError("Dataset is the wrong type");
    }

    SCANN_ASSIGN_OR_RETURN(auto searcher,
                           LeafSearcherT::SingleMachineFactoryLeafSearcher(
                               config, typed_dataset, params, opts));
    auto* typed_searcher =
        down_cast<SingleMachineSearcherBase<T>*>(searcher.get());

    SCANN_ASSIGN_OR_RETURN(
        auto reordering_helper,
        ReorderingHelperFactory<T>::Build(config, params.reordering_dist,
                                          typed_dataset, opts));
    typed_searcher->EnableReordering(std::move(reordering_helper),
                                     params.post_reordering_num_neighbors,
                                     params.post_reordering_epsilon);

    if (config.partitioning().has_incremental_training_config()) {
      searcher->MaybeReleaseDataset();
      SCANN_ASSIGN_OR_RETURN(auto mutator, typed_searcher->GetMutator());
      SCANN_RETURN_IF_ERROR(mutator->EnableIncrementalTraining(config));
    }
    return {std::move(searcher)};
  }
};

template <typename LeafSearcherT>
StatusOrSearcherUntyped SingleMachineFactoryUntypedImpl(
    const ScannConfig& orig_config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts) {
  ScannConfig config = orig_config;

  if (config.has_autopilot()) {
    shared_ptr<Dataset> autopilot_dataset = dataset;
    if (opts.bfloat16_dataset) autopilot_dataset = opts.bfloat16_dataset;
    if (opts.pre_quantized_fixed_point &&
        opts.pre_quantized_fixed_point->fixed_point_dataset)
      autopilot_dataset = opts.pre_quantized_fixed_point->fixed_point_dataset;
    SCANN_ASSIGN_OR_RETURN(config, Autopilot(orig_config, autopilot_dataset));
  }

  GenericSearchParameters params;
  SCANN_RETURN_IF_ERROR(params.PopulateValuesFromScannConfig(config));
  if (params.reordering_dist->NormalizationRequired() != NONE && dataset &&
      dataset->normalization() !=
          params.reordering_dist->NormalizationRequired()) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the exact distance measure.");
  }

  if (params.pre_reordering_dist->NormalizationRequired() != NONE && dataset &&
      dataset->normalization() !=
          params.pre_reordering_dist->NormalizationRequired()) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the pre-reordering distance "
        "measure.");
  }

  if (opts.type_tag == kInvalidTypeTag) {
    CHECK(dataset) << "Code fails to wire-through the type tag";
    opts.type_tag = dataset->TypeTag();
  }

  SCANN_ASSIGN_OR_RETURN(
      auto searcher, SCANN_CALL_FUNCTION_BY_TAG(
                         opts.type_tag,
                         SingleMachineFactoryImplClass<
                             LeafSearcherT>::template SingleMachineFactoryImpl,
                         config, dataset, params, &opts));
  CHECK(searcher) << "Returning nullptr instead of Status is a bug";

  if (config.crowding().enabled() && opts.crowding_attributes) {
    SCANN_RETURN_IF_ERROR(
        searcher->EnableCrowding(std::move(opts.crowding_attributes),
                                 std::move(opts.crowding_dimension_names)));
  }

  searcher->set_config(std::move(config));
  return {std::move(searcher)};
}

std::vector<float> InverseMultiplier(PreQuantizedFixedPoint* fixed_point);

template <typename T>
StatusOrSearcherUntyped AsymmetricHasherFactory(
    shared_ptr<TypedDataset<T>> dataset, const ScannConfig& config,
    SingleMachineFactoryOptions* opts, const GenericSearchParameters& params) {
  const auto& ah_config = config.hash().asymmetric_hash();
  shared_ptr<const DistanceMeasure> quantization_distance;
  std::shared_ptr<ThreadPool> pool = opts->parallelization_pool;
  if (ah_config.has_quantization_distance()) {
    SCANN_ASSIGN_OR_RETURN(
        quantization_distance,
        GetDistanceMeasure(ah_config.quantization_distance()));
  } else {
    quantization_distance = params.pre_reordering_dist;
  }

  internal::TrainedAsymmetricHashingResults<T> training_results;
  if (config.hash().asymmetric_hash().has_centers_filename() ||
      opts->ah_codebook.get()) {
    SCANN_ASSIGN_OR_RETURN(
        training_results,
        internal::HashLeafHelpers<T>::LoadAsymmetricHashingModel(
            ah_config, params, pool, opts->ah_codebook.get()));
  } else {
    if (!dataset) {
      return InvalidArgumentError(
          "Cannot train AH centers because the dataset is null.");
    }

    if (dataset->size() < ah_config.num_clusters_per_block()) {
      return {make_unique<BruteForceSearcher<T>>(
          params.pre_reordering_dist, dataset,
          params.pre_reordering_num_neighbors, params.pre_reordering_epsilon)};
    }

    const int num_workers = (!pool) ? 0 : (pool->NumThreads());
    LOG(INFO) << "Single-machine AH training with dataset size = "
              << dataset->size() << ", " << num_workers + 1 << " thread(s).";

    SCANN_ASSIGN_OR_RETURN(
        training_results,
        internal::HashLeafHelpers<T>::TrainAsymmetricHashingModel(
            dataset, ah_config, params, pool));
  }
  return internal::HashLeafHelpers<T>::AsymmetricHasherFactory(
      dataset, opts->hashed_dataset, training_results, params, pool);
}

}  // namespace internal
}  // namespace research_scann

#endif
