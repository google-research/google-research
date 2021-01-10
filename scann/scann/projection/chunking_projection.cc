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

#include "scann/projection/chunking_projection.h"

#include "absl/strings/substitute.h"
#include "scann/projection/identity_projection.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOr<unique_ptr<ChunkingProjection<T>>> BuildFromConfigImpl(
    const ProjectionConfig& config) {
  if (!config.has_input_dim()) {
    return InvalidArgumentError(
        "Must set input_dim field in projection config");
  }
  const int32_t input_dim = config.input_dim();

  switch (config.projection_type()) {
    case ProjectionConfig::VARIABLE_CHUNK: {
      if (config.variable_blocks_size() <= 0) {
        return InvalidArgumentError(
            "variable_blocks must be populated for projection type "
            "VARIABLE_CHUNK.");
      }
      vector<int32_t> dims_per_block;
      int32_t num_blocks = 0;
      for (const auto& vblock : config.variable_blocks()) {
        dims_per_block.insert(dims_per_block.end(), vblock.num_blocks(),
                              vblock.num_dims_per_block());
        num_blocks += vblock.num_blocks();
      }
      return make_unique<ChunkingProjection<T>>(num_blocks, dims_per_block);
    }

    case ProjectionConfig::IDENTITY_CHUNK:
      if (!config.has_num_blocks()) {
        return InvalidArgumentError(
            "Must specify num_blocks for IDENTITY_CHUNK projection");
      }
      return make_unique<ChunkingProjection<T>>(config.num_blocks());

    case ProjectionConfig::CHUNK:
    case ProjectionConfig::PCA:
    default: {
      if (!config.has_num_dims_per_block()) {
        return InvalidArgumentError(
            "num_dims_per_block must be specified for projection type CHUNK.");
      }
      const int32_t dims_per_block = config.num_dims_per_block();
      const int32_t num_blocks = config.has_num_blocks()
                                     ? config.num_blocks()
                                     : DivRoundUp(input_dim, dims_per_block);
      if (dims_per_block > input_dim) {
        return InvalidArgumentError(
            absl::Substitute("num_dims_per_block ($0) cannot be larger than "
                             "input_dim ($1) for CHUNK "
                             "projection type",
                             dims_per_block, input_dim));
      }
      if (num_blocks > DivRoundUp(input_dim, dims_per_block)) {
        return InvalidArgumentError(absl::Substitute(
            "num_blocks ($0) is too large (should be <= $1), and some blocks "
            "will consist entirely of zero-padding.",
            num_blocks, DivRoundUp(input_dim, dims_per_block)));
      }
      return make_unique<ChunkingProjection<T>>(num_blocks, dims_per_block);
    }
  }
}

template <typename T>
StatusOr<unique_ptr<ChunkingProjection<T>>>
ChunkingProjection<T>::BuildFromConfig(
    const ProjectionConfig& config,
    unique_ptr<Projection<T>> initial_projection) {
  TF_ASSIGN_OR_RETURN(auto result, BuildFromConfigImpl<T>(config));
  result->initial_projection_ = std::move(initial_projection);
  return {std::move(result)};
}

template <typename T>
ChunkingProjection<T>::ChunkingProjection(const int32_t num_blocks,
                                          const int32_t num_dims_per_block)
    : num_blocks_(num_blocks) {
  CHECK_GT(num_blocks_, 0)
      << "The number of blocks for chunking should be at least one!";
  CHECK_GT(num_dims_per_block, 0)
      << "The number of dims per block for chunking should be at least one!";

  dims_per_block_.resize(num_blocks, num_dims_per_block);

  ComputeCumulativeDims();
}

template <typename T>
ChunkingProjection<T>::ChunkingProjection(const int32_t num_blocks)
    : num_blocks_(num_blocks), is_identity_chunk_impl_(true) {}

template <typename T>
ChunkingProjection<T>::ChunkingProjection(
    const int32_t num_blocks, ConstSpan<int32_t> variable_dims_per_block)
    : num_blocks_(num_blocks),
      dims_per_block_(variable_dims_per_block.begin(),
                      variable_dims_per_block.end()) {
  CHECK_GT(num_blocks_, 0)
      << "The number of blocks for chunking should be at least one!";
  CHECK_EQ(dims_per_block_.size(), num_blocks_)
      << "The size of variable_dims_per_block must be equal to num_blocks_";

  for (size_t i = 0; i < dims_per_block_.size(); ++i) {
    CHECK_GT(dims_per_block_[i], 0)
        << "Number of dims per block for chunking should be at least one!";
  }

  ComputeCumulativeDims();
}

template <typename T>
template <typename FloatT>
StatusOr<ChunkedDatapoint<FloatT>> ChunkingProjection<T>::ProjectInputImpl(
    const DatapointPtr<T>& input) const {
  if (input.dimensionality() != input.nonzero_entries() && input.IsDense()) {
    return InvalidArgumentError(
        "ChunkingProjection does not work with binary data.");
  }

  if (is_identity_chunk_impl_) {
    Datapoint<FloatT> projected;
    IdentityProjection<T> identity_projection;
    SCANN_RETURN_IF_ERROR(identity_projection.ProjectInput(input, &projected));
    return ChunkedDatapoint<FloatT>::MakeIdentityChunk(
        std::move(*projected.mutable_values()), num_blocks_);
  }

  Datapoint<FloatT> projected;
  const DimensionIndex output_dims = cum_dims_per_block_.get()[num_blocks_];
  projected.mutable_values()->reserve(output_dims);

  if (initial_projection_) {
    SCANN_RETURN_IF_ERROR(initial_projection_->ProjectInput(input, &projected));
  } else {
    CopyToDatapoint(input, &projected);
  }

  if (num_blocks_ > input.dimensionality()) {
    return InvalidArgumentError(
        absl::Substitute("num_blocks for chunking ($0) should be less than "
                         "input dimensions ($1).",
                         num_blocks_, input.dimensionality()));
  }

  for (size_t i = 0; i < dims_per_block_.size(); ++i) {
    if (dims_per_block_[i] > input.dimensionality()) {
      return InvalidArgumentError(
          absl::Substitute("num_dims_per_block ($0) should be less than the "
                           "input dimensions ($1).",
                           dims_per_block_[i], input.dimensionality()));
    }
  }

  decltype(projected) densified_storage;
  if (!projected.IsDense()) {
    if (input.dimensionality() > 10 * 1000 * 1000) {
      return InvalidArgumentError(absl::StrCat(
          "Attempting to chunk a sparse vector with dimensionality ",
          input.dimensionality(),
          ", which is  > 10 "
          "million.  This likely indicates a misconfiguration, using "
          "asymmetric hashing for a dataset that it is very poorly suited "
          "for."));
    }
    ToDense(projected.ToPtr(), &densified_storage);
    projected = std::move(densified_storage);
  }

  if (projected.values().size() < output_dims) {
    projected.mutable_values()->resize(output_dims, 0);
  }
  return ChunkedDatapoint<FloatT>(std::move(*projected.mutable_values()),
                                  cum_dims_per_block_, num_blocks_);
}

template <typename T>
void ChunkingProjection<T>::ComputeCumulativeDims() {
  cum_dims_per_block_.reset(new uint32_t[num_blocks_ + 1],
                            [](uint32_t* p) { delete[] p; });
  cum_dims_per_block_.get()[0] = 0;
  for (size_t i = 0; i < num_blocks_; ++i) {
    cum_dims_per_block_.get()[i + 1] =
        cum_dims_per_block_.get()[i] + dims_per_block_[i];
  }
}

template <typename T>
Status ChunkingProjection<T>::ProjectInput(
    const DatapointPtr<T>& input, ChunkedDatapoint<float>* chunked) const {
  TF_ASSIGN_OR_RETURN(*chunked, ProjectInputImpl<float>(input));
  return OkStatus();
}
template <typename T>
Status ChunkingProjection<T>::ProjectInput(
    const DatapointPtr<T>& input, ChunkedDatapoint<double>* chunked) const {
  TF_ASSIGN_OR_RETURN(*chunked, ProjectInputImpl<double>(input));
  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, ChunkingProjection);

}  // namespace research_scann
