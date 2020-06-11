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



#ifndef SCANN__PROJECTION_CHUNKING_PROJECTION_H_
#define SCANN__PROJECTION_CHUNKING_PROJECTION_H_

#include <memory>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/projection/projection_base.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class ChunkingProjection;

template <typename T>
class ChunkedDatapoint {
 public:
  ChunkedDatapoint() {}

  DatapointPtr<T> operator[](size_t chunk_idx) const {
    DCHECK_LT(chunk_idx, num_blocks_);
    if (!ABSL_PREDICT_TRUE(boundaries_)) {
      return MakeDatapointPtr<T>(storage_);
    }
    const uint32_t lower_bound = boundaries_.get()[chunk_idx];
    const uint32_t chunk_size = boundaries_.get()[chunk_idx + 1] - lower_bound;
    DCHECK_LE(lower_bound + chunk_size, storage_.size());
    const T* values = storage_.data() + lower_bound;
    return DatapointPtr<T>(nullptr, values, chunk_size, chunk_size);
  }

  size_t num_blocks() const { return num_blocks_; }

  size_t size() const { return num_blocks(); }

 private:
  ChunkedDatapoint(std::vector<T> storage,
                   shared_ptr<const uint32_t> boundaries, uint32_t num_blocks)
      : storage_(std::move(storage)),
        num_blocks_(num_blocks),
        boundaries_(std::move(boundaries)) {}

  static ChunkedDatapoint<T> MakeIdentityChunk(std::vector<T> storage,
                                               uint32_t num_blocks) {
    ChunkedDatapoint result;
    result.storage_ = std::move(storage);
    result.num_blocks_ = num_blocks;
    return result;
  }

  std::vector<T> storage_;

  uint32_t num_blocks_ = 0;

  shared_ptr<const uint32_t> boundaries_ = nullptr;

  template <typename U>
  friend class ChunkingProjection;
};

class ChunkingProjectionUntyped : public VirtualDestructor {};

template <typename T>
class ChunkingProjection : public ChunkingProjectionUntyped {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(ChunkingProjection);

  static StatusOr<unique_ptr<ChunkingProjection<T>>> BuildFromConfig(
      const ProjectionConfig& config,
      unique_ptr<Projection<T>> initial_projection = nullptr);

  ChunkingProjection(const int32_t num_blocks,
                     const int32_t num_dims_per_block);

  ChunkingProjection(const int32_t num_blocks,
                     ConstSpan<int32_t> variable_dims_per_block);

  explicit ChunkingProjection(const int32_t num_blocks);

  Status ProjectInput(const DatapointPtr<T>& input,
                      ChunkedDatapoint<float>* result) const;
  Status ProjectInput(const DatapointPtr<T>& input,
                      ChunkedDatapoint<double>* result) const;

  int32_t num_blocks() const { return num_blocks_; }

  DimensionIndex input_dim() const {
    return (cum_dims_per_block_ == nullptr)
               ? 0
               : (cum_dims_per_block_.get()[num_blocks_]);
  }

  void set_initial_projection(unique_ptr<Projection<T>> p) {
    initial_projection_ = std::move(p);
  }

  Status ProjectInput(const DatapointPtr<T>& input,
                      vector<Datapoint<float>>* chunked) const {
    return BackcompatImpl<float>(input, chunked);
  }
  Status ProjectInput(const DatapointPtr<T>& input,
                      vector<Datapoint<double>>* chunked) const {
    return BackcompatImpl<double>(input, chunked);
  }

 private:
  template <typename FloatT>
  StatusOr<ChunkedDatapoint<FloatT>> ProjectInputImpl(
      const DatapointPtr<T>& input) const;

  template <typename FloatT>
  ChunkedDatapoint<FloatT> SparseChunkImpl(
      const DatapointPtr<FloatT>& input) const;

  template <typename FloatT>
  ChunkedDatapoint<FloatT> DenseChunkImpl(
      const DatapointPtr<FloatT>& input) const;

  void ComputeCumulativeDims();

  template <typename FloatT>
  Status BackcompatImpl(const DatapointPtr<T>& input,
                        vector<Datapoint<FloatT>>* chunked) const {
    ChunkedDatapoint<FloatT> raw;
    SCANN_RETURN_IF_ERROR(ProjectInput(input, &raw));
    chunked->resize(raw.num_blocks());
    for (size_t i : Seq(raw.num_blocks())) {
      CopyToDatapoint(raw[i], chunked->data() + i);
    }
    return OkStatus();
  }

  unique_ptr<Projection<T>> initial_projection_;
  uint32_t num_blocks_ = 0;
  std::vector<int32_t> dims_per_block_;
  shared_ptr<uint32_t> cum_dims_per_block_;

  bool is_identity_chunk_impl_ = false;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, ChunkingProjection);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
