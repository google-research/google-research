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



#ifndef SCANN__PARTITIONING_PARTITIONER_BASE_H_
#define SCANN__PARTITIONING_PARTITIONER_BASE_H_

#include <algorithm>
#include <hash_set>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {

class UntypedPartitioner {
 public:
  virtual ~UntypedPartitioner();

  virtual void CopyToProto(SerializedPartitioner* result) const = 0;

  virtual int32_t n_tokens() const = 0;

  virtual Normalization NormalizationRequired() const { return NONE; }

  enum TokenizationMode {
    DATABASE,

    QUERY
  };

  TokenizationMode tokenization_mode() const { return tokenization_mode_; }

  void set_tokenization_mode(TokenizationMode val) {
    tokenization_mode_ = val;
    OnSetTokenizationMode();
  }

  virtual int8_t TypeTag() const = 0;

  virtual void set_training_parallelization_pool(
      shared_ptr<thread::ThreadPool> pool);

 protected:
  virtual void OnSetTokenizationMode() {}

  void set_tokenization_mode_no_hook(TokenizationMode tokenization_mode) {
    tokenization_mode_ = tokenization_mode;
  }

 private:
  TokenizationMode tokenization_mode_ = DATABASE;
};

template <typename T>
class Partitioner : public UntypedPartitioner {
 public:
  virtual unique_ptr<Partitioner<T>> Clone() const = 0;

  virtual Status TokenForDatapoint(const DatapointPtr<T>& query,
                                   int32_t* result) const = 0;

  virtual Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& query, std::vector<int32_t>* result) const = 0;

  virtual Status TokenForDatapointBatched(
      const TypedDataset<T>& queries, std::vector<int32_t>* results,
      thread::ThreadPool* pool = nullptr) const;

  virtual Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries,
      MutableSpan<std::vector<int32_t>> results) const;

  virtual StatusOr<vector<std::vector<DatapointIndex>>> TokenizeDatabase(
      const TypedDataset<T>& database, thread::ThreadPool* pool_or_null) const;

  int8_t TypeTag() const final { return TagForType<T>(); }
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, Partitioner);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
