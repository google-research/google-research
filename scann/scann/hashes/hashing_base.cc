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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#include "scann/hashes/hashing_base.h"

namespace tensorflow {
namespace scann_ops {

Status UntypedHashing::LoadParameters() {
  return UnimplementedError("Not implemented here.");
}

Status UntypedHashing::Reconstruct(const DatapointPtr<uint8_t>& hashed,
                                   Datapoint<double>* reconstructed) {
  return UnimplementedError("Not implemented here.");
}

double UntypedHashing::ComputeSymmetricDistance(
    const DatapointPtr<uint8_t>& hashed1,
    const DatapointPtr<uint8_t>& hashed2) {
  LOG(FATAL) << "Not implemented here";
}

std::string UntypedHashing::model_string() const {
  LOG(FATAL) << "Not implemented here";
}

void UntypedHashing::set_model_string(const std::string& model_string) {
  LOG(FATAL) << "Not implemented here";
}

UntypedHashing::~UntypedHashing() {}

template <typename T>
Status Hashing<T>::TrainSingleMachine(const TypedDataset<T>& dataset) {
  return UnimplementedError("Not implemented here.");
}

template <typename T>
Status Hashing<T>::Hash(const DatapointPtr<T>& input,
                        Datapoint<uint8_t>* hashed) {
  return UnimplementedError("Not implemented here.");
}

template <typename T>
Status Hashing<T>::GetNeighborsViaSymmetricDistance(
    const DatapointPtr<T>& query, const DenseDataset<uint8_t>& hashed_database,
    int32_t num_neighbors, double max_distance, NNResultsVector* top_items) {
  return UnimplementedError("Not implemented here.");
}

template <typename T>
Hashing<T>::~Hashing() {}

SCANN_INSTANTIATE_TYPED_CLASS(, Hashing);

}  // namespace scann_ops
}  // namespace tensorflow
