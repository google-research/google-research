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



#ifndef SCANN__BASE_INTERNAL_SINGLE_MACHINE_FACTORY_IMPL_H_
#define SCANN__BASE_INTERNAL_SINGLE_MACHINE_FACTORY_IMPL_H_

#include <memory>

#include "scann/base/reordering_helper_factory.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/crowding.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/scann_config_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace scann_ops {

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

    TF_ASSIGN_OR_RETURN(auto searcher,
                        LeafSearcherT::SingleMachineFactoryLeafSearcher(
                            config, typed_dataset, params, opts));
    auto* typed_searcher =
        down_cast<SingleMachineSearcherBase<T>*>(searcher.get());

    TF_ASSIGN_OR_RETURN(
        auto reordering_helper,
        ReorderingHelperFactory<T>::Build(config, params.reordering_dist,
                                          typed_dataset, opts));
    typed_searcher->EnableReordering(std::move(reordering_helper),
                                     params.post_reordering_num_neighbors,
                                     params.post_reordering_epsilon);
    if (config.has_compressed_reordering()) {
      DCHECK(!typed_searcher->needs_dataset());
      typed_searcher->ReleaseDatasetAndDocids();
      typed_searcher->set_compressed_dataset(opts->compressed_dataset);
    }

    return {std::move(searcher)};
  }
};

template <typename LeafSearcherT>
StatusOrSearcherUntyped SingleMachineFactoryUntypedImpl(
    const ScannConfig& config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts) {
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

  TF_ASSIGN_OR_RETURN(auto searcher,
                      SCANN_CALL_FUNCTION_BY_TAG(
                          opts.type_tag,
                          SingleMachineFactoryImplClass<
                              LeafSearcherT>::template SingleMachineFactoryImpl,
                          config, dataset, params, &opts));
  CHECK(searcher) << "Returning nullptr instead of Status is a bug";

  if (config.crowding().enabled() && opts.crowding_attributes) {
    SCANN_RETURN_IF_ERROR(
        searcher->EnableCrowding(std::move(opts.crowding_attributes)));
  }

  searcher->set_creation_timestamp(opts.creation_timestamp);
  return {std::move(searcher)};
}

}  // namespace internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
