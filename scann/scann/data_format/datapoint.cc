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

#include "scann/data_format/datapoint.h"

#include "scann/data_format/gfv_properties.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
bool DatapointPtr<T>::HasNonzero(DimensionIndex dimension_index) const {
  DCHECK_LT(dimension_index, dimensionality());
  DCHECK(IsSparse()) << "Can only call HasNonzero on sparse DatapointPtrs.";
  if (nonzero_entries() == 0) return false;
  const DimensionIndex* found =
      std::lower_bound(indices_, indices_ + nonzero_entries_, dimension_index);
  return found < indices_ + nonzero_entries_ && *found == dimension_index;
}

template <typename T>
T DatapointPtr<T>::GetElement(DimensionIndex dimension_index) const {
  DCHECK_LT(dimension_index, dimensionality());
  if (IsDense()) {
    if (dimensionality() == nonzero_entries()) {
      return values()[dimension_index];
    } else {
      return GetElementPacked(dimension_index);
    }
  } else {
    if (nonzero_entries() == 0) return 0;
    const DimensionIndex* found = std::lower_bound(
        indices_, indices_ + nonzero_entries_, dimension_index);
    if (found < indices_ + nonzero_entries_ && *found == dimension_index) {
      return (values() == nullptr) ? 1 : (values_[found - indices_]);
    } else {
      return 0;
    }
  }
}

template <typename T>
void DatapointPtr<T>::ToGfvIndicesAndMetadata(GenericFeatureVector* gfv) const {
  if (IsSparse()) {
    for (size_t i = 0; i < nonzero_entries(); ++i) {
      gfv->add_feature_index(indices()[i]);
    }

    gfv->set_feature_dim(dimensionality());
  }
}

template <typename T>
GenericFeatureVector Datapoint<T>::ToGfv() const {
  GenericFeatureVector result = ToPtr().ToGfv();
  result.set_norm_type(
      static_cast<GenericFeatureVector::FeatureNorm>(normalization()));
  return result;
}

template <typename T>
Status Datapoint<T>::FromGfv(const GenericFeatureVector& gfv) {
  auto status = FromGfvImpl(gfv);
  if (!status.ok()) clear();
  return status;
}

template <typename T>
Status Datapoint<T>::FromGfvImpl(const GenericFeatureVector& gfv) {
  clear();
  normalization_ = static_cast<Normalization>(gfv.norm_type());
  TF_ASSIGN_OR_RETURN(dimensionality_, GetGfvDimensionality(gfv));
  const bool is_binary = gfv.feature_type() == GenericFeatureVector::BINARY;
  if (gfv.feature_type() == GenericFeatureVector::STRING) {
    return InvalidArgumentError("GFV with feature_type == STRING");
  }

  indices_.assign(gfv.feature_index().begin(), gfv.feature_index().end());

  if (is_binary && !indices_.empty()) {
    if (!IsUint8<T>()) {
      values_.resize(indices_.size(), T(1));
    }
  } else {
    SCANN_RETURN_IF_ERROR(GfvValuesToVector(gfv, &values_));
  }

  if (indices_.empty()) return OkStatus();

  if (indices_.size() != values_.size() && !is_binary) {
    return InvalidArgumentError(
        absl::StrCat("Size of indices (", indices_.size(),
                     ") does not match size of values (", values_.size(),
                     ") as required for sparse non-binary vectors."));
  }

  bool need_check_dupes = false;
  for (size_t j : Seq(1, indices_.size())) {
    if (indices_[j - 1] >= indices_[j]) {
      SortIndices();
      need_check_dupes = true;
      break;
    }
  }

  if (indices_.back() >= dimensionality_) {
    return InvalidArgumentError(
        absl::StrCat("Largest dimension index (", indices_.back(),
                     ") is >= dimensionality (", dimensionality_, ")."));
  }

  if (need_check_dupes) {
    for (size_t j : Seq(1, indices_.size())) {
      if (indices_[j] == indices_[j - 1]) {
        SCANN_LOG_NOOP(ERROR, 10)
            << "Found duplicate indices when parsing GFV with data_id: "
            << gfv.data_id_str();
        return InvalidArgumentError(
            "Invalid sparse vector.  Found duplicate dimension index:  %d",
            indices_[j]);
      }
    }
  }

  RemoveExplicitZeroesFromSparseVector();

  return OkStatus();
}

template <typename T>
void Datapoint<T>::MakeNotBinary() {
  auto* mut_values = mutable_values();
  if (mut_values->empty()) {
    mut_values->resize(nonzero_entries(), T(1));
  } else if (IsUint8<T>() && IsDense() &&
             nonzero_entries() < dimensionality()) {
    DCHECK_EQ(nonzero_entries(), DivRoundUp(dimensionality(), 8));
    auto dptr = this->ToPtr();
    std::vector<T> new_values(dimensionality());
    for (size_t j : Seq(dimensionality())) {
      new_values[j] = dptr.GetElementPacked(j);
    }
    *mut_values = std::move(new_values);
  }
}

template <typename T>
bool Datapoint<T>::IndicesSorted() const {
  for (size_t i = 1; i < indices().size(); ++i) {
    if (indices()[i - 1] >= indices()[i]) return false;
  }

  return true;
}

template <typename T>
void Datapoint<T>::SortIndices() {
  if (indices().size() == 0) return;
  if (values().size() == 0) {
    ZipSortBranchOptimized(std::less<DimensionIndex>(), indices_.begin(),
                           indices_.end());
  } else if (values().size() == indices().size()) {
    ZipSortBranchOptimized(std::less<DimensionIndex>(), indices_.begin(),
                           indices_.end(), values_.begin(), values_.end());
  } else {
    LOG(FATAL) << "Cannot sort indices of malformed Datapoint.  values "
                  "must either be empty or of the same size as indices.";
  }
}

template <typename T>
void Datapoint<T>::RemoveExplicitZeroesFromSparseVector() {
  if (indices_.empty() || values_.empty()) {
    return;
  }

  size_t from = 0, to = 0;
  DCHECK_EQ(indices_.size(), values_.size());
  for (; from < values_.size(); ++from) {
    if (values_[from] == 0) continue;
    values_[to] = values_[from];
    indices_[to++] = indices_[from];
  }

  indices_.resize(to);
  values_.resize(to);
}

SCANN_INSTANTIATE_TYPED_CLASS(, DatapointPtr);
SCANN_INSTANTIATE_TYPED_CLASS(, Datapoint);

}  // namespace scann_ops
}  // namespace tensorflow
