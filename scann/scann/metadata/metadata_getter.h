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



#ifndef SCANN_METADATA_METADATA_GETTER_H_
#define SCANN_METADATA_METADATA_GETTER_H_

#include <optional>
#include <string>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/features.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class MetadataGetter;

class UntypedMetadataGetter {
 public:
  virtual Status AppendMetadata(const GenericFeatureVector& gfv);

  virtual Status UpdateMetadata(DatapointIndex idx,
                                const GenericFeatureVector& gfv);

  virtual Status RemoveMetadata(DatapointIndex removed_idx);

  virtual bool needs_dataset() const;

  virtual research_scann::TypeTag TypeTag() const = 0;

  virtual ~UntypedMetadataGetter();
};

template <typename T>
class MetadataGetter : public UntypedMetadataGetter {
 public:
  MetadataGetter() = default;

  MetadataGetter(const MetadataGetter&) = delete;
  MetadataGetter& operator=(const MetadataGetter&) = delete;

  research_scann::TypeTag TypeTag() const final { return TagForType<T>(); }

  virtual std::optional<size_t> fixed_len_size(
      const TypedDataset<T>* dataset, const DatapointPtr<T>& query) const {
    return std::nullopt;
  }

  virtual Status GetMetadata(const TypedDataset<T>* dataset,
                             const DatapointPtr<T>& query,
                             DatapointIndex neighbor_index,
                             std::string* result) const = 0;

  virtual Status GetMetadatas(const TypedDataset<T>* dataset,
                              const DatapointPtr<T>& query,
                              size_t num_neighbors,
                              DpIdxGetter neighbor_dp_idx_getter,
                              StringSetter metadata_setter) const {
    for (size_t i : Seq(num_neighbors)) {
      std::string result;
      SCANN_RETURN_IF_ERROR(
          GetMetadata(dataset, query, neighbor_dp_idx_getter(i), &result));
      metadata_setter(i, result);
    }
    return OkStatus();
  }

  virtual Status TransformAndCopyMetadatas(
      const TypedDataset<T>* dataset, const DatapointPtr<T>& query,
      size_t num_neighbors, DpIdxGetter neighbor_dp_idx_getter,
      OutputStringGetter output_string_getter) const {
    for (size_t i : Seq(num_neighbors)) {
      SCANN_RETURN_IF_ERROR(GetMetadata(
          dataset, query, neighbor_dp_idx_getter(i), output_string_getter(i)));
    }
    return OkStatus();
  }

  virtual StatusOr<std::string> GetByDatapointIndex(
      const TypedDataset<T>* dataset, DatapointIndex dp_idx) const {
    return UnimplementedError(
        StrCat("Cannot get metadata by datapoint index for "
               "metadata getter type ",
               typeid(*this).name(), "."));
  }
};

}  // namespace research_scann

#endif
