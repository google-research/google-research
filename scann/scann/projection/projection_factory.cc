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

#include "scann/projection/projection_factory.h"

#include "scann/projection/chunking_projection.h"
#include "scann/projection/identity_projection.h"
#include "scann/projection/random_orthogonal_projection.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

Status FixRemainderDims(const DimensionIndex input_dim,
                        const ProjectionConfig& config,
                        DimensionIndex* projected_dim) {
  if (config.num_blocks() == 1) {
    if (input_dim < *projected_dim) {
      return InvalidArgumentError(
          "input_dim must be >= num_dims_per_block for the specified "
          "projection "
          "type if chunking is not used, i.e. if num_blocks == 1.");
    }
    return OkStatus();
  }

  *projected_dim = input_dim;
  if (static_cast<int64_t>(config.num_blocks() * config.num_dims_per_block()) -
          static_cast<int64_t>(input_dim) >=
      config.num_dims_per_block()) {
    return InvalidArgumentError(
        "num_blocks * num_dims_per_block - input_dim must be < "
        "num_dims_per_block for the specified projection type. This ensures "
        "that no block consists entirely of zero padding.");
  }

  return OkStatus();
}

template <typename T>
StatusOr<unique_ptr<Projection<T>>> ProjectionFactoryImpl<T>::Create(
    const ProjectionConfig& config, const TypedDataset<T>* dataset,
    int32_t seed_offset) {
  const int32_t effective_seed = config.seed() + seed_offset;
  if (!config.has_input_dim()) {
    return InvalidArgumentError(
        "Must set input_dim field in projection config");
  }
  const DimensionIndex input_dim = config.input_dim();

  if (!config.has_num_dims_per_block() &&
      config.projection_type() != ProjectionConfig::NONE) {
    return InvalidArgumentError(
        "num_dims_per_block must be specified for ProjectionFactory unless "
        "projection type NONE is being used.");
  }

  DimensionIndex projected_dim =
      config.num_blocks() * config.num_dims_per_block();

  auto fix_remainder_dims = [input_dim, &projected_dim, &config]() -> Status {
    return FixRemainderDims(input_dim, config, &projected_dim);
  };

  unique_ptr<Projection<T>> result;
  switch (config.projection_type()) {
    case ProjectionConfig::NONE:
      return {make_unique<IdentityProjection<T>>()};
    case ProjectionConfig::CHUNK:
      return InvalidArgumentError(
          "Cannot return projection type CHUNK from ProjectionFactory. "
          "Did you mean to call ChunkingProjectionFactory?");
    case ProjectionConfig::VARIABLE_CHUNK:
      return InvalidArgumentError(
          "Cannot return projection type VARIABLE_CHUNK from "
          "ProjectionFactory. Did you mean to call ChunkingProjectionFactory?");
    case ProjectionConfig::RANDOM_ORTHOGONAL: {
      SCANN_RETURN_IF_ERROR(fix_remainder_dims());

      auto projection = make_unique<RandomOrthogonalProjection<T>>(
          input_dim, projected_dim, effective_seed);
      projection->Create();
      return {std::move(projection)};
    }

    default:
      return UnimplementedError(
          "The specified projection type is not implemented.");
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, ProjectionFactoryImpl);

}  // namespace scann_ops
}  // namespace tensorflow
