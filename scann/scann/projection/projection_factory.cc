// Copyright 2024 The Google Research Authors.
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

#include <cstdint>
#include <memory>
#include <utility>

#include "scann/data_format/dataset.h"
#include "scann/projection/identity_projection.h"
#include "scann/projection/pca_projection.h"
#include "scann/projection/projection_base.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

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

Status ValidateDimension(ProjectionConfig::ProjectionType projection_type,
                         const DimensionIndex input_dim,
                         const DimensionIndex projected_dim) {
  constexpr DimensionIndex kMaxDimensionality = numeric_limits<int32_t>::max();
  if (projected_dim > kMaxDimensionality) {
    if (projection_type == ProjectionConfig::RANDOM_ORTHOGONAL ||
        projection_type == ProjectionConfig::RANDOM_BINARY ||
        projection_type == ProjectionConfig::RANDOM_BINARY_DYNAMIC ||
        projection_type == ProjectionConfig::RANDOM_SPARSE_BINARY ||
        projection_type == ProjectionConfig::RANDOM_GAUSS ||
        projection_type == ProjectionConfig::RANDOM_BINARY) {
      return InvalidArgumentError(
          "num_blocks * num_dims_per_block must fit in a signed 32-bit "
          "integer.");
    }
  }
  if (input_dim > kMaxDimensionality &&
      projection_type != ProjectionConfig::NONE) {
    return InvalidArgumentError(
        "input_dim must fit in a signed 32-bit integer");
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
      config.projection_type() != ProjectionConfig::NONE &&
      (config.projection_type() != ProjectionConfig::PCA &&
       !config.has_pca_significance_threshold())) {
    return InvalidArgumentError(
        "num_dims_per_block must be specified for ProjectionFactory unless "
        "projection type NONE or PCA is being used.");
  }

  DimensionIndex projected_dim =
      static_cast<DimensionIndex>(config.num_blocks()) *
      config.num_dims_per_block();
  SCANN_RETURN_IF_ERROR(
      ValidateDimension(config.projection_type(), input_dim, projected_dim));

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
    case ProjectionConfig::PCA: {
      SCANN_RETURN_IF_ERROR(fix_remainder_dims());
      if (!dataset) {
        return InvalidArgumentError(
            "A dataset must be provided when constructing a PCA projection");
      }
      unique_ptr<PcaProjection<T>> result;
      if (config.has_num_dims_per_block()) {
        result = make_unique<PcaProjection<T>>(input_dim,
                                               config.num_dims_per_block());
        result->Create(*dataset, config.build_covariance());
      } else {
        if (config.has_pca_significance_threshold()) {
          result = make_unique<PcaProjection<T>>(input_dim, input_dim);
          result->Create(*dataset, config.pca_significance_threshold(),
                         config.pca_truncation_threshold());
        } else {
          return InvalidArgumentError(
              "Must specify num_dims_per_block or pca_significance_threshold "
              "for PCA projection.");
        }
      }
      return {std::move(result)};
    }

    default:
      return UnimplementedError(
          "The specified projection type is not implemented.");
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, ProjectionFactoryImpl);

}  // namespace research_scann
