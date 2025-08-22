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



#ifndef SCANN_PROJECTION_PROJECTION_FACTORY_H_
#define SCANN_PROJECTION_PROJECTION_FACTORY_H_

#include <cstdint>
#include <optional>
#include <variant>

#include "scann/data_format/dataset.h"
#include "scann/projection/chunking_projection.h"
#include "scann/projection/eigenvalue_opq_projection.h"
#include "scann/projection/projection_base.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/common.h"

namespace research_scann {

template <typename T>
using MaybeDatasetOrSerializedProjection =
    std::variant<std::nullopt_t, const TypedDataset<T>*,
                 const SerializedProjection*>;

template <typename T>
class ProjectionFactoryImpl {
 public:
  static StatusOr<unique_ptr<Projection<T>>> Create(
      const ProjectionConfig& config,
      MaybeDatasetOrSerializedProjection<T> dataset_or_serialized_projection,
      int32_t seed_offset, ThreadPool* parallelization_pool);
};

template <typename T>
StatusOr<unique_ptr<Projection<T>>> ProjectionFactory(
    const ProjectionConfig& config,
    MaybeDatasetOrSerializedProjection<T> dataset_or_serialized_projection =
        std::nullopt,
    int32_t seed_offset = 0, ThreadPool* parallelization_pool = nullptr) {
  return ProjectionFactoryImpl<T>::Create(config,
                                          dataset_or_serialized_projection,
                                          seed_offset, parallelization_pool);
}

template <typename T>
StatusOr<unique_ptr<Projection<T>>> ProjectionFactory(
    const ProjectionConfig& config, int32_t seed_offset) {
  return ProjectionFactoryImpl<T>::Create(config, std::nullopt, seed_offset,
                                          nullptr);
}

template <typename T>
StatusOr<unique_ptr<ChunkingProjection<T>>> ChunkingProjectionFactory(
    const ProjectionConfig& config,
    MaybeDatasetOrSerializedProjection<T> dataset_or_serialized_projection =
        std::nullopt,
    int32_t seed_offset = 0, ThreadPool* parallelization_pool = nullptr) {
  ProjectionConfig canonicalized_config = config;
  if (config.projection_type() != ProjectionConfig::VARIABLE_CHUNK &&
      config.projection_type() != ProjectionConfig::IDENTITY_CHUNK &&
      config.has_num_dims_per_block() && !config.has_num_blocks()) {
    canonicalized_config.set_num_blocks(
        DivRoundUp(config.input_dim(), config.num_dims_per_block()));
  }

  unique_ptr<Projection<T>> initial_projection;
  switch (config.projection_type()) {
    case ProjectionConfig::CHUNK:
    case ProjectionConfig::VARIABLE_CHUNK:
    case ProjectionConfig::IDENTITY_CHUNK:
      break;
    default: {
      SCANN_ASSIGN_OR_RETURN(
          initial_projection,
          ProjectionFactory<T>(canonicalized_config,
                               dataset_or_serialized_projection, seed_offset,
                               parallelization_pool));
      break;
    }
  }

  if (canonicalized_config.projection_type() ==
      ProjectionConfig::EIGENVALUE_OPQ) {
    auto evopq =
        down_cast<const EigenvalueOpqProjection<T>*>(initial_projection.get());
    auto result = make_unique<ChunkingProjection<T>>(
        canonicalized_config.num_blocks(), evopq->variable_dims_per_block());
    result->set_initial_projection(std::move(initial_projection));
    return result;
  }

  return ChunkingProjection<T>::BuildFromConfig(canonicalized_config,
                                                std::move(initial_projection));
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, ProjectionFactoryImpl);

}  // namespace research_scann

#endif
