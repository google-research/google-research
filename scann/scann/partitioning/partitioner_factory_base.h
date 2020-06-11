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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__PARTITIONING_PARTITIONER_FACTORY_BASE_H_
#define SCANN__PARTITIONING_PARTITIONER_FACTORY_BASE_H_

#include "scann/data_format/dataset.h"
#include "scann/partitioning/kmeans_tree_partitioner_utils.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/projection/projection_base.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactory(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<thread::ThreadPool> pool = nullptr);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryPreSampledAndProjected(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<thread::ThreadPool> training_parallelization_pool = nullptr);

template <typename T, typename ProjectionType = double>
unique_ptr<Partitioner<T>> MakeProjectingDecorator(
    shared_ptr<const Projection<T>> projection,
    unique_ptr<Partitioner<ProjectionType>> partitioner) {
  Partitioner<T>* result;
  if (dynamic_cast<KMeansTreeLikePartitioner<ProjectionType>*>(
          partitioner.get())) {
    result = new KMeansTreeProjectingDecorator<T, ProjectionType>(
        std::move(projection),
        absl::WrapUnique(down_cast<KMeansTreeLikePartitioner<ProjectionType>*>(
            partitioner.release())));
  } else {
    result = new GenericProjectingDecorator<T, ProjectionType>(
        std::move(projection), std::move(partitioner));
  }
  return absl::WrapUnique(result);
}

#define SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern_or_nothing, type)     \
  extern_or_nothing template StatusOr<unique_ptr<Partitioner<type>>>       \
  PartitionerFactory<type>(                                                \
      const TypedDataset<type>* dataset, const PartitioningConfig& config, \
      shared_ptr<thread::ThreadPool> training_parallelization_pool);       \
  extern_or_nothing template StatusOr<unique_ptr<Partitioner<type>>>       \
  PartitionerFactoryPreSampledAndProjected<type>(                          \
      const TypedDataset<type>* dataset, const PartitioningConfig& config, \
      shared_ptr<thread::ThreadPool> training_parallelization_pool);

SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, int8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, uint8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, int16_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, uint16_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, int32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, uint32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, int64_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, unsigned long long_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, float);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(extern, double);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
