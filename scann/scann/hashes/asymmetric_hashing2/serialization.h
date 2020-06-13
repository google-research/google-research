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

#ifndef SCANN__HASHES_ASYMMETRIC_HASHING2_SERIALIZATION_H_
#define SCANN__HASHES_ASYMMETRIC_HASHING2_SERIALIZATION_H_

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
CentersForAllSubspaces DatasetSpanToCentersProto(
    ConstSpan<DenseDataset<T>> dataset_span,
    AsymmetricHasherConfig::QuantizationScheme quantization_scheme) {
  CentersForAllSubspaces result;
  result.set_quantization_scheme(quantization_scheme);
  for (const auto& dataset : dataset_span) {
    auto cur_centers = result.add_subspace_centers();
    for (auto dp_ptr : dataset) {
      auto center = cur_centers->add_center();
      *center = dp_ptr.ToGfv();
    }
  }
  return result;
}

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
