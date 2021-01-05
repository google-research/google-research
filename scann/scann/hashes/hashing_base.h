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



#ifndef SCANN__HASHES_HASHING_BASE_H_
#define SCANN__HASHES_HASHING_BASE_H_

#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

class UntypedHashing {
 public:
  virtual Status LoadParameters();

  virtual Status Reconstruct(const DatapointPtr<uint8_t>& hashed,
                             Datapoint<double>* reconstructed);

  virtual double ComputeSymmetricDistance(const DatapointPtr<uint8_t>& hashed1,
                                          const DatapointPtr<uint8_t>& hashed2);

  virtual std::string model_string() const;

  virtual void set_model_string(const std::string& model_string);

  virtual ~UntypedHashing();
};

template <typename T>
class Hashing : public UntypedHashing {
 public:
  virtual Status TrainSingleMachine(const TypedDataset<T>& dataset);

  virtual Status Hash(const DatapointPtr<T>& input, Datapoint<uint8_t>* hashed);

  virtual Status GetNeighborsViaSymmetricDistance(
      const DatapointPtr<T>& query,
      const DenseDataset<uint8_t>& hashed_database, int32_t num_neighbors,
      double max_distance, NNResultsVector* top_items);

  ~Hashing() override;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, Hashing);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
