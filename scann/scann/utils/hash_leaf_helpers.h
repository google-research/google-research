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

#ifndef SCANN__UTILS_HASH_LEAF_HELPERS_H_
#define SCANN__UTILS_HASH_LEAF_HELPERS_H_

#include <memory>

#include "scann/base/single_machine_base.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/proto/centers.pb.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

namespace internal {

template <typename T>
struct TrainedAsymmetricHashingResults {
  shared_ptr<const asymmetric_hashing2::Indexer<T>> indexer = nullptr;
  shared_ptr<const asymmetric_hashing2::AsymmetricQueryer<T>> queryer = nullptr;
  AsymmetricHasherConfig::LookupType lookup_type =
      AsymmetricHasherConfig::FLOAT;
  AsymmetricHasherConfig::FixedPointLUTConversionOptions
      fixed_point_lut_conversion_options;
  double noise_shaping_threshold = NAN;
};

template <typename T>
struct HashLeafHelpers {
  static StatusOr<TrainedAsymmetricHashingResults<T>>
  TrainAsymmetricHashingModel(shared_ptr<TypedDataset<T>> dataset,
                              const AsymmetricHasherConfig& config,
                              const GenericSearchParameters& params,
                              shared_ptr<thread::ThreadPool> pool);

  static StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>
  AsymmetricHasherFactory(
      shared_ptr<TypedDataset<T>> dataset,
      shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
      const TrainedAsymmetricHashingResults<T>& training_results,
      const GenericSearchParameters& params,
      shared_ptr<thread::ThreadPool> pool);

  static StatusOr<TrainedAsymmetricHashingResults<T>>
  LoadAsymmetricHashingModel(
      const AsymmetricHasherConfig& config,
      const GenericSearchParameters& params,
      shared_ptr<thread::ThreadPool> pool,
      CentersForAllSubspaces* preloaded_codebook = nullptr);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TrainedAsymmetricHashingResults);
SCANN_INSTANTIATE_TYPED_CLASS(extern, HashLeafHelpers);

}  // namespace internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
