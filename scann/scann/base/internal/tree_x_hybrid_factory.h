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

#ifndef SCANN_BASE_INTERNAL_TREE_X_HYBRID_FACTORY_H_
#define SCANN_BASE_INTERNAL_TREE_X_HYBRID_FACTORY_H_

#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOrPtr<UntypedSingleMachineSearcherBase> TreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params,
    std::function<StatusOrPtr<UntypedSingleMachineSearcherBase>(
        const ScannConfig&, const shared_ptr<TypedDataset<T>>&,
        const GenericSearchParameters&, SingleMachineFactoryOptions*)>
        leaf_factory,
    SingleMachineFactoryOptions* opts);

#define SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, Type) \
  extern_keyword template StatusOrPtr<UntypedSingleMachineSearcherBase>        \
  TreeXHybridFactory<Type>(                                                    \
      const ScannConfig& config,                                               \
      const shared_ptr<TypedDataset<Type>>& dataset,                           \
      const GenericSearchParameters& params,                                   \
      std::function<StatusOrPtr<UntypedSingleMachineSearcherBase>(             \
          const ScannConfig&, const shared_ptr<TypedDataset<Type>>&,           \
          const GenericSearchParameters&, SingleMachineFactoryOptions*)>       \
          leaf_factory,                                                        \
      SingleMachineFactoryOptions* opts);

#define SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY(extern_keyword)               \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, int8_t);   \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, uint8_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, int16_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, int32_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, uint32_t); \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, int64_t);  \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, float);    \
  SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY_FOR_TYPE(extern_keyword, double);

SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY(extern);

}  // namespace research_scann

#endif
