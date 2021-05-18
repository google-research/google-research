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



#ifndef SCANN_PARTITIONING_PARTITIONER_FACTORY_H_
#define SCANN_PARTITIONING_PARTITIONER_FACTORY_H_

#include <cstdint>
#include <memory>

#include "scann/data_format/dataset.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/partitioning/partitioner_factory_base.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerialized(
    const SerializedPartitioner& proto, const PartitioningConfig& config,
    int32_t projection_seed_offset = 0);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerialized(
    const std::string& serialized, const PartitioningConfig& config,
    int32_t projection_seed_offset = 0);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromKMeansTree(
    shared_ptr<const KMeansTree> kmeans_tree, const PartitioningConfig& config);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> LoadSinglePartitioner(
    const PartitioningConfig& config, int32_t epoch,
    int32_t projection_seed_offset = 0);

template <typename T>
StatusOr<std::shared_ptr<const Partitioner<T>>> LoadSinglePartitionerWithCache(
    const PartitioningConfig& config, int32_t epoch = 0);

template <typename T>
StatusOr<std::shared_ptr<const Partitioner<T>>>
PartitionerFromSerializedWithCache(
    const SerializedPartitioner& serialized_partitioner,
    const PartitioningConfig& config);

StatusOr<SerializedPartitioner> ReadSerializedPartitioner(
    const PartitioningConfig& config, int32_t epoch = 0);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerializedImpl(
    const SerializedPartitioner& proto, const PartitioningConfig& config);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerialized(
    const std::string& serialized, const PartitioningConfig& config,
    int32_t projection_seed_offset) {
  SerializedPartitioner proto;
  if (!proto.ParseFromString(serialized)) {
    return InvalidArgumentError(
        "Could not parse serialized Partitioner proto.");
  }
  return PartitionerFromSerialized<T>(proto, config, projection_seed_offset);
}

StatusOr<std::string> CanonicalizePartitionerFilename(
    const PartitioningConfig& config, int32_t epoch = 0);

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> LoadSinglePartitioner(
    const PartitioningConfig& config, int32_t epoch,
    int32_t projection_seed_offset) {
  TF_ASSIGN_OR_RETURN(SerializedPartitioner sp,
                      ReadSerializedPartitioner(config));
  return PartitionerFromSerialized<T>(sp, config, projection_seed_offset);
}

#define SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern_or_nothing, \
                                                         type)              \
  extern_or_nothing template StatusOr<unique_ptr<Partitioner<type>>>        \
  PartitionerFromSerialized<type>(const SerializedPartitioner& proto,       \
                                  const PartitioningConfig& config,         \
                                  int32_t projection_seed_offset);          \
  extern_or_nothing template StatusOr<unique_ptr<Partitioner<type>>>        \
  PartitionerFromKMeansTree<type>(                                          \
      std::shared_ptr<const KMeansTree> kmeans_tree,                        \
      const PartitioningConfig& config);

SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, int8_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, uint8_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, int16_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, uint16_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, int32_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, uint32_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, int64_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, uint64_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, float);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(extern, double);

}  // namespace research_scann

#endif
