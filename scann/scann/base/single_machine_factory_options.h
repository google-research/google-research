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

#ifndef SCANN__BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_
#define SCANN__BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_

#include <memory>
#include <vector>

#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace scann_ops {
template <typename T>
class DenseDataset;
template <typename T>
class TypedDataset;
template <typename T>
class SingleMachineSearcherBase;
class ScannConfig;

struct SingleMachineFactoryOptions {
  SingleMachineFactoryOptions() {}

  StatusOr<DatapointIndex> ComputeConsistentSize(
      const Dataset* dataset = nullptr) const;

  StatusOr<DimensionIndex> ComputeConsistentDimensionality(
      const HashConfig& config, const Dataset* dataset = nullptr) const;

  TypeTag type_tag = kInvalidTypeTag;

  shared_ptr<vector<std::vector<DatapointIndex>>> datapoints_by_token;

  shared_ptr<DenseDataset<uint8_t>> compressed_dataset;

  shared_ptr<PreQuantizedFixedPoint> pre_quantized_fixed_point;

  shared_ptr<DenseDataset<uint8_t>> hashed_dataset;

  std::shared_ptr<CentersForAllSubspaces> ah_codebook;

  std::shared_ptr<CentersForAllSubspaces> reordering_ah_codebook;

  std::shared_ptr<SerializedPartitioner> serialized_partitioner;

  std::shared_ptr<const KMeansTree> kmeans_tree;

  std::shared_ptr<std::vector<std::string>> hash_parameters;

  shared_ptr<vector<int64_t>> crowding_attributes;

  shared_ptr<thread::ThreadPool> parallelization_pool;

  int64_t creation_timestamp = numeric_limits<int64_t>::max();
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
