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

#ifndef SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_MODEL_H_
#define SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_MODEL_H_

#include "scann/data_format/dataset.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
class Model {
 public:
  using FloatT = FloatingTypeFor<T>;

  static StatusOr<unique_ptr<Model<T>>> FromCenters(
      std::vector<DenseDataset<FloatT>> centers,
      AsymmetricHasherConfig::QuantizationScheme quantization_scheme =
          AsymmetricHasherConfig::PRODUCT);

  static StatusOr<unique_ptr<Model<T>>> FromProto(
      const CentersForAllSubspaces& proto);

  CentersForAllSubspaces ToProto() const;

  ConstSpan<DenseDataset<FloatT>> centers() const { return centers_; }

  uint32_t num_clusters_per_block() const { return num_clusters_per_block_; }

  size_t num_blocks() const { return centers_.size(); }

  AsymmetricHasherConfig::QuantizationScheme quantization_scheme() const {
    return quantization_scheme_;
  }

  bool CentersEqual(const Model& rhs) const;

 private:
  explicit Model(
      std::vector<DenseDataset<FloatT>> centers,
      AsymmetricHasherConfig::QuantizationScheme quantization_scheme);

  std::vector<DenseDataset<FloatT>> centers_ = {};

  uint32_t num_clusters_per_block_ = numeric_limits<uint32_t>::max();

  AsymmetricHasherConfig::QuantizationScheme quantization_scheme_ =
      AsymmetricHasherConfig::PRODUCT;

  TF_DISALLOW_COPY_AND_ASSIGN(Model);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, Model);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
