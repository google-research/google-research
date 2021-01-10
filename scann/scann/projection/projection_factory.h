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



#ifndef SCANN_PROJECTION_PROJECTION_FACTORY_H_
#define SCANN_PROJECTION_PROJECTION_FACTORY_H_

#include "scann/data_format/dataset.h"
#include "scann/projection/chunking_projection.h"
#include "scann/projection/projection_base.h"
#include "scann/proto/projection.pb.h"

namespace research_scann {

template <typename T>
class ProjectionFactoryImpl {
 public:
  static StatusOr<unique_ptr<Projection<T>>> Create(
      const ProjectionConfig& config, const TypedDataset<T>* dataset,
      int32_t seed_offset);
};

template <typename T>
StatusOr<unique_ptr<Projection<T>>> ProjectionFactory(
    const ProjectionConfig& config, const TypedDataset<T>* dataset = nullptr,
    int32_t seed_offset = 0) {
  return ProjectionFactoryImpl<T>::Create(config, dataset, seed_offset);
}

template <typename T>
StatusOr<unique_ptr<Projection<T>>> ProjectionFactory(
    const ProjectionConfig& config, int32_t seed_offset) {
  return ProjectionFactoryImpl<T>::Create(config, nullptr, seed_offset);
}

template <typename T>
inline StatusOr<unique_ptr<ChunkingProjection<T>>> ChunkingProjectionFactory(
    const ProjectionConfig& config, const TypedDataset<T>* dataset = nullptr,
    int32_t seed_offset = 0) {
  unique_ptr<Projection<T>> initial_projection;
  switch (config.projection_type()) {
    case ProjectionConfig::CHUNK:
    case ProjectionConfig::VARIABLE_CHUNK:
    case ProjectionConfig::IDENTITY_CHUNK:
      break;
    default: {
      TF_ASSIGN_OR_RETURN(initial_projection,
                          ProjectionFactory<T>(config, dataset, seed_offset));
      break;
    }
  }

  return ChunkingProjection<T>::BuildFromConfig(config,
                                                std::move(initial_projection));
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, ProjectionFactoryImpl);

}  // namespace research_scann

#endif
