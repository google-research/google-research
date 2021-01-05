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



#ifndef SCANN__HASHES_ASYMMETRIC_HASHING2_INDEXING_H_
#define SCANN__HASHES_ASYMMETRIC_HASHING2_INDEXING_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
class Indexer {
 public:
  using FloatT = FloatingTypeFor<T>;

  Indexer(shared_ptr<const ChunkingProjection<T>> projector,
          shared_ptr<const DistanceMeasure> quantization_distance,
          shared_ptr<const Model<T>> model);

  Status Hash(const DatapointPtr<T>& input, Datapoint<uint8_t>* hashed) const;

  Status Hash(const DatapointPtr<T>& input, std::string* hashed) const;

  Status Hash(const DatapointPtr<T>& input, MutableSpan<uint8_t> hashed) const;

  Status Hash(ConstSpan<T> input, MutableSpan<uint8_t> hashed) const;

  Status HashWithNoiseShaping(const DatapointPtr<T>& input,
                              Datapoint<uint8_t>* hashed,
                              double threshold) const;

  Status HashWithNoiseShaping(const DatapointPtr<T>& input,
                              MutableSpan<uint8_t> hashed,
                              double threshold) const;

  Status HashWithNoiseShaping(ConstSpan<T> input, MutableSpan<uint8_t> hashed,
                              double threshold) const;

  Status HashWithNoiseShaping(const DatapointPtr<T>& maybe_residual,
                              const DatapointPtr<T>& original,
                              Datapoint<uint8_t>* hashed,
                              double threshold) const;

  Status HashWithNoiseShaping(const DatapointPtr<T>& maybe_residual,
                              const DatapointPtr<T>& original,
                              MutableSpan<uint8_t> hashed,
                              double threshold) const;

  Status HashWithNoiseShaping(ConstSpan<T> maybe_residual,
                              ConstSpan<T> original,
                              MutableSpan<uint8_t> hashed,
                              double threshold) const;

  StatusOr<DenseDataset<uint8_t>> HashDataset(
      const TypedDataset<T>& dataset) const;

  Status Reconstruct(const DatapointPtr<uint8_t>& input,
                     Datapoint<FloatT>* reconstructed) const;

  Status Reconstruct(absl::string_view input,
                     Datapoint<FloatT>* reconstructed) const;

  Status Reconstruct(ConstSpan<uint8_t> input,
                     MutableSpan<FloatT> reconstructed) const;

  StatusOr<FloatT> DistanceBetweenOriginalAndHashed(
      ConstSpan<FloatT> original, ConstSpan<uint8_t> hashed,
      shared_ptr<const DistanceMeasure> distance_override = nullptr) const;

  DimensionIndex hash_space_dimension() const;

  DimensionIndex original_space_dimension() const;

  Status ComputeResidual(const DatapointPtr<T>& original,
                         const DatapointPtr<uint8_t>& hashed,
                         Datapoint<FloatT>* result) const;

 private:
  shared_ptr<const ChunkingProjection<T>> projector_;
  shared_ptr<const DistanceMeasure> quantization_distance_;
  shared_ptr<const Model<T>> model_;

  std::vector<FloatT> flattend_model_;

  std::vector<std::pair<uint32_t, uint32_t>> subspace_sizes_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, Indexer);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
