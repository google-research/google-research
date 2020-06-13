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



#include "scann/data_format/dataset.h"

#include <hash_set>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "scann/data_format/docid_collection.h"
#include "scann/data_format/gfv_conversion.h"
#include "scann/data_format/gfv_properties.h"
#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/memory_logging.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {
namespace scann_ops {

void Dataset::UnusedKeyMethod() {}

shared_ptr<DocidCollectionInterface> Dataset::ReleaseDocids() {
  auto result = std::move(docids_);
  docids_ = make_unique<VariableLengthDocidCollection>(
      VariableLengthDocidCollection::CreateWithEmptyDocids(result->size()));
  return result;
}

Status Dataset::NormalizeByTag(Normalization tag) {
  if (tag == normalization()) return OkStatus();
  switch (tag) {
    case NONE:
      return OkStatus();
    case UNITL2NORM:
      return NormalizeUnitL2();
    default:
      return UnimplementedError(
          "Normalization type specified by tag not implemented yet.");
  }
}

template <typename T>
Status TypedDataset<T>::Append(const DatapointPtr<T>& dptr) {
  return Append(dptr, absl::StrCat(this->size()));
}

template <typename T>
Status TypedDataset<T>::Append(const GenericFeatureVector& gfv) {
  return Append(gfv, absl::StrCat(this->size()));
}

template <typename T>
void TypedDataset<T>::AppendOrDie(const DatapointPtr<T>& dptr) {
  AppendOrDie(dptr, absl::StrCat(this->size()));
}

template <typename T>
void TypedDataset<T>::AppendOrDie(const GenericFeatureVector& gfv) {
  AppendOrDie(gfv, absl::StrCat(this->size()));
}

template <typename T>
void TypedDataset<T>::AppendOrDie(const DatapointPtr<T>& dptr,
                                  string_view docid) {
  TF_CHECK_OK(this->Append(dptr, docid));
}

template <typename T>
void TypedDataset<T>::AppendOrDie(const GenericFeatureVector& gfv,
                                  string_view docid) {
  TF_CHECK_OK(this->Append(gfv, docid));
}

template <typename T>
Status TypedDataset<T>::MeanByDimension(Datapoint<double>* result) const {
  const size_t size = this->size();
  if (size <= 0) {
    return FailedPreconditionError(
        "Cannot compute the mean of an empty dataset.");
  }

  DCHECK(result);
  result->ZeroFill(this->dimensionality());
  std::vector<double>& values = *result->mutable_values();

  if (this->IsSparse()) {
    for (size_t i = 0; i < size; ++i) {
      auto dptr = (*this)[i];
      if (this->is_binary()) {
        for (DimensionIndex j = 0; j < dptr.nonzero_entries(); ++j) {
          ++values[dptr.indices()[j]];
        }
      } else {
        for (DimensionIndex j = 0; j < dptr.nonzero_entries(); ++j) {
          values[dptr.indices()[j]] += dptr.values()[j];
        }
      }
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      auto dptr = (*this)[i];
      if (this->is_binary()) {
        for (DimensionIndex j = 0; j < dptr.dimensionality(); ++j) {
          values[j] += dptr.GetElementPacked(j);
        }
      } else {
        PointwiseAdd(values.data(), dptr.values(), dptr.nonzero_entries());
      }
    }
  }

  const double multiplier = 1.0 / static_cast<double>(size);
  for (double& elem : values) {
    elem *= multiplier;
  }

  return OkStatus();
}

template <typename T>
Status TypedDataset<T>::MeanByDimension(ConstSpan<DatapointIndex> subset,
                                        Datapoint<double>* result) const {
  if (subset.empty()) {
    return InvalidArgumentError("Cannot compute the mean of an empty subset.");
  }

  DCHECK(result);
  result->ZeroFill(this->dimensionality());
  std::vector<double>& values = *result->mutable_values();

  if (this->IsSparse()) {
    for (DatapointIndex i : subset) {
      auto dptr = (*this)[i];
      if (this->is_binary()) {
        for (DimensionIndex j = 0; j < dptr.nonzero_entries(); ++j) {
          ++values[dptr.indices()[j]];
        }
      } else {
        for (DimensionIndex j = 0; j < dptr.nonzero_entries(); ++j) {
          values[dptr.indices()[j]] += dptr.values()[j];
        }
      }
    }
  } else {
    for (DatapointIndex i : subset) {
      auto dptr = (*this)[i];
      if (this->is_binary()) {
        for (DimensionIndex j = 0; j < dptr.dimensionality(); ++j) {
          values[j] += dptr.GetElementPacked(j);
        }
      } else {
        PointwiseAdd(values.data(), dptr.values(), dptr.nonzero_entries());
      }
    }
  }

  const double multiplier = 1.0 / static_cast<double>(subset.size());
  for (double& elem : values) {
    elem *= multiplier;
  }

  return OkStatus();
}

template <typename T>
void TypedDataset<T>::MeanVarianceByDimension(
    Datapoint<double>* means, Datapoint<double>* variances) const {
  CHECK(!this->is_binary()) << "Not implemented for binary datasets.";
  vector<DatapointIndex> subset;
  subset.reserve(dimensionality());
  for (DatapointIndex i = 0; i < size(); ++i) {
    subset.push_back(i);
  }

  return MeanVarianceByDimension(subset, means, variances);
}

template <typename T>
void TypedDataset<T>::MeanVarianceByDimension(
    ConstSpan<DatapointIndex> subset, Datapoint<double>* means,
    Datapoint<double>* variances) const {
  DCHECK(variances);
  CHECK(!this->is_binary()) << "Not implemented for binary datasets.";
  CHECK_GT(subset.size(), 0)
      << "Cannot compute MeanVarianceByDimension on empty subset.";
  const DimensionIndex dimensionality = this->dimensionality();
  using AT = AccumulatorTypeFor<T>;
  vector<AT> sums(dimensionality);
  vector<AT> squares(dimensionality);

  for (DatapointIndex index : subset) {
    auto point = (*this)[index];
    if (this->IsDense()) {
      for (DimensionIndex i = 0; i < dimensionality; ++i) {
        const T num = point.values()[i];
        sums[i] += num;
        squares[i] += static_cast<AT>(num) * static_cast<AT>(num);
      }
    } else {
      for (size_t i = 0; i < point.nonzero_entries(); ++i) {
        const T num = point.values()[i];
        const DatapointIndex index = point.indices()[i];
        sums[index] += num;
        squares[index] += static_cast<AT>(num) * static_cast<AT>(num);
      }
    }
  }

  const double one_over_n = 1.0 / subset.size();
  variances->clear();
  variances->mutable_values()->resize(dimensionality);
  if (means) {
    means->clear();
    means->mutable_values()->resize(dimensionality);
  }
  for (size_t i = 0; i < dimensionality; ++i) {
    const double mean = sums[i] * one_over_n;
    (*variances->mutable_values())[i] = squares[i] * one_over_n - mean * mean;
    if (means) {
      (*means->mutable_values())[i] = mean;
    }
  }
}

template <typename T>
Status TypedDataset<T>::NormalizeUnitL2() {
  if (this->is_binary() || IsIntegerType<T>()) {
    return FailedPreconditionError(
        "Unit L2 normalization is not supported for binary "
        "and integral datasets.");
  }

  const size_t size = this->size();
  for (size_t i = 0; i < size; ++i) {
    auto dptr = (*this)[i];
    const double norm = SquaredL2Norm(dptr);
    if (norm == 0) continue;
    const double multiplier = 1.0 / sqrt(norm);
    T* val_ptr = const_cast<T*>(dptr.values());
    for (size_t j = 0; j < dptr.nonzero_entries(); ++j) {
      val_ptr[j] *= multiplier;
    }
  }

  this->set_normalization(UNITL2NORM);
  return OkStatus();
}

template <typename T>
Status TypedDataset<T>::NormalizeZeroMeanUnitVariance() {
  if (this->is_binary() || IsIntegerType<T>()) {
    return FailedPreconditionError(
        "Zero mean/unit variance normalization is not "
        "supported for binary and integral datasets.");
  }

  const size_t size = this->size();
  for (size_t i = 0; i < size; ++i) {
    auto dptr = (*this)[i];
    double mean, variance;
    MeanVar(dptr, &mean, &variance);

    T* val_ptr = const_cast<T*>(dptr.values());
    if (variance == 0) {
      std::fill(val_ptr, val_ptr + dptr.nonzero_entries(), 0.0);
      return OkStatus();
    }

    const double inv_std = 1.0 / std::sqrt(variance);
    for (size_t j = 0; j < dptr.nonzero_entries(); ++j) {
      val_ptr[j] -= mean;
      val_ptr[j] *= inv_std;
    }
  }

  this->set_normalization(STDGAUSSNORM);
  return OkStatus();
}

namespace {

template <typename T>
inline void ToDoubleAlwaysCopy(const DatapointPtr<T>& dptr,
                               Datapoint<double>* dp) {
  ToDouble(dptr, dp);
}

template <>
inline void ToDoubleAlwaysCopy(const DatapointPtr<double>& dptr,
                               Datapoint<double>* dp) {
  CopyToDatapoint(dptr, dp);
}
}  // namespace

template <typename T>
void TypedDataset<T>::GetDatapoint(size_t index,
                                   Datapoint<double>* result) const {
  DCHECK(result);
  result->clear();
  auto unconverted = (*this)[index];
  ToDoubleAlwaysCopy(unconverted, result);
  result->set_normalization(this->normalization());
}

template <typename T>
void DenseDataset<T>::set_dimensionality(DimensionIndex dimensionality) {
  if (this->size() == 0) {
    this->set_dimensionality_no_checks(dimensionality);
    SetStride();
  } else {
    DCHECK_EQ(this->dimensionality(), dimensionality)
        << "Cannot reset the dimensionality of a non-empty Dataset.";
  }
}

template <typename T>
DimensionIndex DenseDataset<T>::NumActiveDimensions() const {
  return this->dimensionality();
}

template <typename T>
void DenseDataset<T>::ShrinkToFit() {
  this->docids()->ShrinkToFit();
  data_.shrink_to_fit();
}

template <typename T>
void DenseDataset<T>::set_is_binary(bool val) {
  this->Dataset::set_is_binary(val);
  SetStride();
}

template <typename T>
void DenseDataset<T>::SetStride() {
  if (this->packing_strategy() == HashedItem::BINARY) {
    stride_ = this->dimensionality() / 8 + (this->dimensionality() % 8 > 0);
  } else if (this->packing_strategy() == HashedItem::NIBBLE) {
    stride_ = this->dimensionality() / 2 + (this->dimensionality() % 2 > 0);
  } else {
    stride_ = this->dimensionality();
  }
}

template <typename T>
void DenseDataset<T>::GetDenseDatapoint(size_t index,
                                        Datapoint<double>* result) const {
  this->GetDatapoint(index, result);
}

template <typename T>
double DenseDataset<T>::GetDistance(const DistanceMeasure& dist,
                                    size_t vec1_index,
                                    size_t vec2_index) const {
  return dist.GetDistanceDense((*this)[vec1_index], (*this)[vec2_index]);
}

template <typename T>
Status DenseDataset<T>::Append(const DatapointPtr<T>& dptr, string_view docid) {
  if (!dptr.IsDense()) {
    if (dptr.IsSparseOrigin()) {
      return FailedPreconditionError(
          "Cannot append an empty datapoint (ie, the \"sparse origin\") to a "
          "dense dataset. This error sometimes results from datasets that have "
          "an empty GenericFeatureVector proto.");
    } else {
      return FailedPreconditionError(
          "Cannot append a sparse datapoint to a dense dataset.");
    }
  }

  const DimensionIndex dptr_dim = dptr.dimensionality();
  const bool dptr_is_binary = dptr.dimensionality() > dptr.nonzero_entries();
  if (dptr_is_binary && !std::is_same<T, uint8_t>::value) {
    return InvalidArgumentError(
        "Binary DenseDatasets may only be built with uint8_t as a template "
        "parameter.");
  }

  if (this->dimensionality() == 0) {
    this->set_dimensionality(dptr_dim);

    if (this->packing_strategy() == HashedItem::NONE) {
      set_is_binary(dptr_is_binary);
    }
  } else if (this->dimensionality() != dptr_dim) {
    return FailedPreconditionError(
        StrFormat("Dimensionality mismatch:  Appending a %u dimensional "
                  "datapoint to a %u dimensional dataset.",
                  static_cast<uint64_t>(dptr_dim),
                  static_cast<uint64_t>(this->dimensionality())));
  } else if (stride_ != dptr.nonzero_entries()) {
    return FailedPreconditionError(
        StrFormat("Cannot append a vector to a dataset with different "
                  "stride: Appending a %u dimensional datapoint to a %u "
                  "dimensional dataset.",
                  static_cast<uint64_t>(dptr.nonzero_entries()),
                  static_cast<uint64_t>(stride_)));
  }

  Datapoint<T> storage;
  DatapointPtr<T> to_insert = dptr;
  if (this->normalization() != NONE) {
    CopyToDatapoint(dptr, &storage);
    SCANN_RETURN_IF_ERROR(NormalizeByTag(this->normalization(), &storage));
    to_insert = storage.ToPtr();
  }
  SCANN_RETURN_IF_ERROR(this->AppendDocid(docid));
  data_.insert(data_.end(), to_insert.values_slice().begin(),
               to_insert.values_slice().end());
  return OkStatus();
}

template <typename T>
Status DenseDataset<T>::Append(const GenericFeatureVector& gfv,
                               string_view docid) {
  Datapoint<T> dp;
  SCANN_RETURN_IF_ERROR(dp.FromGfv(gfv));
  SCANN_RETURN_IF_ERROR(this->Append(dp.ToPtr(), docid))
      << "Docid:  " << docid << " Debug string:  " << gfv.DebugString();
  return OkStatus();
}

template <typename T>
DenseDataset<T>::DenseDataset(vector<T> datapoint_vec,
                              unique_ptr<DocidCollectionInterface> docids)
    : TypedDataset<T>(std::move(docids)), data_(std::move(datapoint_vec)) {
  if (!data_.empty()) {
    stride_ = data_.size() / this->docids()->size();

    this->set_dimensionality_no_checks(stride_);
  }
  DCHECK_EQ(this->docids()->size() * stride_, data_.size());
}

template <typename T>
DenseDataset<T>::DenseDataset(vector<T> datapoint_vec, size_t num_dp)
    : DenseDataset<T>(
          std::move(datapoint_vec),
          make_unique<VariableLengthDocidCollection>(
              VariableLengthDocidCollection::CreateWithEmptyDocids(num_dp))) {}

template <typename T>
void DenseDataset<T>::Reserve(size_t n) {
  if (mutator_) {
    mutator_->Reserve(n);
    return;
  }
  ReserveImpl(n);
}

template <typename T>
void DenseDataset<T>::ReserveImpl(size_t n) {
  data_.reserve(n * stride_);
}

template <typename T>
void DenseDataset<T>::clear() {
  this->set_dimensionality_no_checks(0);
  this->ClearDocids();
  this->set_is_binary(false);
  data_.clear();
  stride_ = 0;
  mutator_ = nullptr;
}

template <typename T>
shared_ptr<DocidCollectionInterface> DenseDataset<T>::ReleaseDocids() {
  auto result = Dataset::ReleaseDocids();
  if (mutator_) {
    mutator_ = nullptr;
    TF_CHECK_OK(GetMutator().status());
  }
  return result;
}

template <typename T>
size_t DenseDataset<T>::MemoryUsageExcludingDocids() const {
  return sizeof(*this) + sizeof(T) * data_.capacity() - sizeof(*this->docids());
}

template <typename T>
StatusOr<typename TypedDataset<T>::Mutator*> DenseDataset<T>::GetMutator()
    const {
  return UnimplementedError("No mutator supported.");
}

template <typename T>
void SparseDataset<T>::set_dimensionality(DimensionIndex dimensionality) {
  if (this->size() == 0) {
    this->set_dimensionality_no_checks(dimensionality);
  } else {
    DCHECK_EQ(this->dimensionality(), dimensionality)
        << "Cannot reset the dimensionality of a non-empty Dataset.";
  }
}

template <typename T>
void SparseDataset<T>::GetDenseDatapoint(size_t index,
                                         Datapoint<double>* result) const {
  DCHECK(result);
  result->clear();
  auto unconverted = (*this)[index];
  result->ZeroFill(unconverted.dimensionality());
  if (this->is_binary()) {
    for (size_t i = 0; i < unconverted.nonzero_entries(); ++i) {
      result->mutable_values()->at(unconverted.indices()[i]) = 1.0;
    }
  } else {
    for (size_t i = 0; i < unconverted.nonzero_entries(); ++i) {
      result->mutable_values()->at(unconverted.indices()[i]) =
          unconverted.values()[i];
    }
  }
  result->set_normalization(this->normalization());
}

template <typename T>
DimensionIndex SparseDataset<T>::NumActiveDimensions() const {
  std::unordered_set<DimensionIndex> is_active;
  for (size_t i = 0; i < this->size(); ++i) {
    const DatapointPtr<T> dptr = (*this)[i];
    for (DimensionIndex j = 0; j < dptr.nonzero_entries(); ++j) {
      is_active.insert(dptr.indices()[j]);
    }
  }

  return is_active.size();
}

template <typename T>
double SparseDataset<T>::GetDistance(const DistanceMeasure& dist,
                                     size_t vec1_index,
                                     size_t vec2_index) const {
  return dist.GetDistanceSparse((*this)[vec1_index], (*this)[vec2_index]);
}

template <typename T>
Status SparseDataset<T>::Append(const GenericFeatureVector& gfv,
                                string_view docid) {
  const auto old_dimensionality = this->dimensionality();
  const size_t old_offsets_size = repr_.start_offsets().size();
  auto result = AppendImpl(gfv, docid);
  if (!result.ok()) {
    if (repr_.start_offsets().size() > old_offsets_size) repr_.PopBack();
    this->set_dimensionality_no_checks(old_dimensionality);
    result = AnnotateStatus(
        result, absl::StrCat("  Docid:  ", docid,
                             "  Debug string:  ", gfv.DebugString()));
  }

  DCHECK(!this->is_binary() || repr_.values().empty());
  return result;
}

template <typename T>
Status SparseDataset<T>::AppendImpl(const GenericFeatureVector& gfv,
                                    string_view docid) {
  TF_ASSIGN_OR_RETURN(bool is_sparse, IsGfvSparse(gfv));
  if (!is_sparse) {
    return FailedPreconditionError(
        "Cannot append a dense GFV to a sparse dataset.");
  }

  TF_ASSIGN_OR_RETURN(DimensionIndex gfv_dim, GetGfvDimensionality(gfv));
  if (this->dimensionality() == 0) {
    this->set_dimensionality(gfv_dim);
  } else if (this->dimensionality() != gfv_dim) {
    return FailedPreconditionError(
        StrFormat("Dimensionality mismatch:  Appending a %u dimensional "
                  "datapoint to a %u dimensional dataset.",
                  static_cast<uint64_t>(gfv_dim),
                  static_cast<uint64_t>(this->dimensionality())));
  }

  const bool gfv_is_binary = gfv.feature_type() == GenericFeatureVector::BINARY;
  if (gfv_is_binary && !std::is_same<T, uint8_t>::value) {
    return InvalidArgumentError(
        "Binary SparseDatasets may only be built with uint8_t as a template "
        "parameter.");
  }

  if (this->size() == 0) {
    this->set_is_binary(gfv_is_binary);
  }

  if (this->is_binary() != gfv_is_binary) {
    return FailedPreconditionError(
        "Cannot append a binary datapoint to a non-binary dataset or "
        "vice-versa.");
  }

  Datapoint<T> dp;
  SCANN_RETURN_IF_ERROR(dp.FromGfv(gfv));
  SCANN_RETURN_IF_ERROR(NormalizeByTag(this->normalization(), &dp));
  SCANN_RETURN_IF_ERROR(this->AppendDocid(docid));
  repr_.Append(dp.indices(), dp.values());
  return OkStatus();
}

template <typename T>
Status SparseDataset<T>::Append(const DatapointPtr<T>& dptr,
                                string_view docid) {
  const auto old_dimensionality = this->dimensionality();
  const size_t old_offsets_size = repr_.start_offsets().size();
  auto result = AppendImpl(dptr, docid);
  if (!result.ok()) {
    if (repr_.start_offsets().size() > old_offsets_size) repr_.PopBack();
    this->set_dimensionality_no_checks(old_dimensionality);
  }

  DCHECK(!this->is_binary() || repr_.values().empty());
  return result;
}

template <typename T>
Status SparseDataset<T>::AppendImpl(const DatapointPtr<T>& dptr,
                                    string_view docid) {
  if (!dptr.IsSparse()) {
    return FailedPreconditionError(
        "Cannot append a dense DatapointPtr to a sparse dataset.");
  }

  const DimensionIndex dptr_dim = dptr.dimensionality();
  if (dptr_dim == 0) {
    return InvalidArgumentError("Invalid datapoint:  Zero dimensionality.");
  }

  if (this->dimensionality() == 0) {
    this->set_dimensionality(dptr_dim);
  } else if (this->dimensionality() != dptr_dim) {
    return FailedPreconditionError(
        StrFormat("Dimensionality mismatch:  Appending a %u dimensional "
                  "datapoint to a %u dimensional dataset.",
                  static_cast<uint64_t>(dptr_dim),
                  static_cast<uint64_t>(this->dimensionality())));
  }

  const bool dptr_may_be_binary = dptr.values() == nullptr;
  const bool dptr_is_definitely_binary =
      dptr_may_be_binary && dptr.nonzero_entries() > 0;
  if (dptr_is_definitely_binary && !std::is_same<T, uint8_t>::value) {
    return InvalidArgumentError(
        "Binary SparseDatasets may only be built with uint8_t as a template "
        "parameter.");
  }

  if (repr_.indices().empty()) {
    this->set_is_binary(dptr_is_definitely_binary);
  }

  if (this->is_binary() && !dptr_may_be_binary) {
    return FailedPreconditionError(
        "Cannot append a non-binary datapoint to a binary dataset.");
  } else if (!this->is_binary() && dptr_is_definitely_binary) {
    return FailedPreconditionError(
        "Cannot append a binary datapoint to a non-binary dataset.");
  }

  Datapoint<T> dp;
  CopyToDatapoint(dptr, &dp);
  SCANN_RETURN_IF_ERROR(NormalizeByTag(this->normalization(), &dp));
  SCANN_RETURN_IF_ERROR(this->AppendDocid(docid));
  repr_.Append(dp.indices(), dp.values());
  return OkStatus();
}

template <typename T>
void SparseDataset<T>::ConvertType(SparseDataset<double>* target) {
  CHECK(!this->is_binary()) << "Not implemented for binary datasets.";
  DCHECK(target);
  target->clear();
  vector<double> vals;
  LOG(INFO) << "SZ = " << repr_.start_offsets().size();
  vals.insert(vals.begin(), repr_.values().begin(), repr_.values().end());
  target->repr_ = SparseDatasetLowLevel<DimensionIndex, double>(
      {repr_.indices().begin(), repr_.indices().end()}, std::move(vals),
      {repr_.start_offsets().begin(), repr_.start_offsets().end()});
  target->set_dimensionality(this->dimensionality());
  target->set_docids_no_checks(this->docids()->Copy());
}

template <typename T>
void SparseDataset<T>::Reserve(size_t n_points) {
  repr_.Reserve(n_points);
}

template <typename T>
void SparseDataset<T>::Reserve(size_t n_points, size_t n_entries) {
  if (this->is_binary()) {
    repr_.ReserveForBinaryData(n_points, n_entries);
  } else {
    repr_.Reserve(n_points, n_entries);
  }
}

template <typename T>
void SparseDataset<T>::clear() {
  repr_.Clear();
  this->set_dimensionality_no_checks(0);
  this->ClearDocids();
  this->set_is_binary(false);
}
template <typename T>
size_t SparseDataset<T>::MemoryUsageExcludingDocids() const {
  return sizeof(*this) + repr_.MemoryUsage() - sizeof(repr_) -
         sizeof(*this->docids());
}

template <typename T>
void SparseDataset<T>::ShrinkToFit() {
  repr_.ShrinkToFit();
  this->docids()->ShrinkToFit();
}

SCANN_INSTANTIATE_TYPED_CLASS(, TypedDataset);
SCANN_INSTANTIATE_TYPED_CLASS(, SparseDataset);
SCANN_INSTANTIATE_TYPED_CLASS(, DenseDataset);

}  // namespace scann_ops
}  // namespace tensorflow
