// Copyright 2021 The Google Research Authors.
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



#ifndef SCANN__BASE_SINGLE_MACHINE_FACTORY_SCANN_H_
#define SCANN__BASE_SINGLE_MACHINE_FACTORY_SCANN_H_

#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/utils/factory_helpers.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class TypedDataset;
class ScannConfig;

template <typename T>
using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

using StatusOrSearcherUntyped =
    StatusOr<unique_ptr<UntypedSingleMachineSearcherBase>>;

template <typename T>
StatusOr<unique_ptr<SingleMachineSearcherBase<T>>> SingleMachineFactoryScann(
    const ScannConfig& config, shared_ptr<TypedDataset<T>> dataset,
    SingleMachineFactoryOptions opts = SingleMachineFactoryOptions());

StatusOrSearcherUntyped SingleMachineFactoryUntypedScann(
    const ScannConfig& config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts);

namespace internal {

template <typename T>
StatusOrSearcherUntyped SingleMachineFactoryLeafSearcherScann(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts);

}

#define SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(          \
    extern_keyword, Type)                                                 \
  extern_keyword template StatusOr<                                       \
      unique_ptr<SingleMachineSearcherBase<Type>>>                        \
  SingleMachineFactoryScann<Type>(const ScannConfig& config,              \
                                  shared_ptr<TypedDataset<Type>> dataset, \
                                  SingleMachineFactoryOptions opts);      \
  extern_keyword template StatusOrSearcherUntyped                         \
  internal::SingleMachineFactoryLeafSearcherScann<Type>(                  \
      const ScannConfig& config,                                          \
      const shared_ptr<TypedDataset<Type>>& dataset,                      \
      const GenericSearchParameters& params,                              \
      SingleMachineFactoryOptions* opts);

#define SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN(extern_keyword)    \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          int8_t);        \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          uint8_t);       \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          int16_t);       \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          uint16_t);      \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          int32_t);       \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          uint32_t);      \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          int64_t);       \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          uint64_t);      \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          float);         \
  SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                          double);

SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN(extern);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
