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

#ifndef SCANN__BASE_REORDERING_HELPER_FACTORY_H_
#define SCANN__BASE_REORDERING_HELPER_FACTORY_H_

#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/reordering_helper.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class ReorderingHelperFactory {
 public:
  static StatusOr<unique_ptr<const ReorderingInterface<T>>> Build(
      const ScannConfig& config,
      const shared_ptr<const DistanceMeasure>& reordering_dist,
      shared_ptr<TypedDataset<T>> dataset, SingleMachineFactoryOptions* opts);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, ReorderingHelperFactory)

}  // namespace scann_ops
}  // namespace tensorflow

#endif
