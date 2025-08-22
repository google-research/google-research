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



#ifndef SCANN_PROJECTION_EIGENVALUE_OPQ_PROJECTION_H_
#define SCANN_PROJECTION_EIGENVALUE_OPQ_PROJECTION_H_

#include <cstdint>
#include <optional>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/projection/projection_base.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class EigenvalueOpqProjection : public Projection<T> {
 public:
  explicit EigenvalueOpqProjection(int32_t input_dims);

  void Create(const Dataset& data, uint32_t num_blocks,
              bool build_covariance = true, ThreadPool* pool = nullptr);

  Status Create(const SerializedProjection& serialized_projection);

  StatusOr<shared_ptr<const TypedDataset<float>>> GetDirections()
      const override;

  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<float>* projected) const override;
  Status ProjectInput(const DatapointPtr<T>& input,
                      Datapoint<double>* projected) const override;

  int32_t projected_dimensionality() const override { return input_dims_; }

  ConstSpan<int32_t> variable_dims_per_block() const { return chunk_sizes_; }

  std::optional<SerializedProjection> SerializeToProto() const final;

 private:
  template <typename FloatT>
  Status ProjectInputImpl(const DatapointPtr<T>& input,
                          Datapoint<FloatT>* projected) const;

  int32_t input_dims_;
  shared_ptr<const DenseDataset<float>> rotation_matrix_;
  vector<int32_t> chunk_sizes_;
  vector<float> eigenvalue_sums_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, EigenvalueOpqProjection);

}  // namespace research_scann

#endif
