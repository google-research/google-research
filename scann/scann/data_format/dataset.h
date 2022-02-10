// Copyright 2022 The Google Research Authors.
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



#ifndef SCANN_DATA_FORMAT_DATASET_H_
#define SCANN_DATA_FORMAT_DATASET_H_

#include <memory>
#include <type_traits>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/docid_collection.h"
#include "scann/data_format/features.pb.h"
#include "scann/data_format/sparse_low_level.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/iterators.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/platform/prefetch.h"

namespace research_scann {

template <typename T>
class TypedDataset;
template <typename T>
class DenseDataset;
template <typename T>
class SparseDataset;

class Dataset : public VirtualDestructor {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(Dataset);

  Dataset() : docids_(make_shared<VariableLengthDocidCollection>()) {}

  explicit Dataset(unique_ptr<DocidCollectionInterface> docids)
      : docids_(std::move(docids)) {
    DCHECK(docids_);
  }

  DatapointIndex size() const { return docids_->size(); }

  bool empty() const { return size() == 0; }

  DimensionIndex dimensionality() const { return dimensionality_; }

  virtual DimensionIndex NumActiveDimensions() const = 0;

  virtual bool IsDense() const = 0;

  bool IsSparse() const { return !IsDense(); }

  virtual void Reserve(size_t n_points) {}

  void ReserveDocids(size_t n_docids) { docids_->Reserve(n_docids); }

  string_view GetDocid(size_t index) const { return docids_->Get(index); }

  const shared_ptr<DocidCollectionInterface>& docids() const { return docids_; }

  virtual shared_ptr<DocidCollectionInterface> ReleaseDocids();

  virtual void clear() = 0;

  virtual research_scann::TypeTag TypeTag() const = 0;

  virtual void GetDatapoint(size_t index, Datapoint<double>* result) const = 0;

  virtual void GetDenseDatapoint(size_t index,
                                 Datapoint<double>* result) const = 0;

  virtual void Prefetch(size_t index) const = 0;

  virtual double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                             size_t vec2_index) const = 0;

  virtual Status MeanByDimension(Datapoint<double>* result) const = 0;

  virtual Status MeanByDimension(ConstSpan<DatapointIndex> subset,
                                 Datapoint<double>* result) const = 0;

  virtual void MeanVarianceByDimension(Datapoint<double>* means,
                                       Datapoint<double>* variances) const = 0;

  virtual void MeanVarianceByDimension(ConstSpan<DatapointIndex> subset,
                                       Datapoint<double>* means,
                                       Datapoint<double>* variances) const = 0;

  virtual Status NormalizeUnitL2() = 0;

  virtual Status NormalizeZeroMeanUnitVariance() = 0;

  Normalization normalization() const { return normalization_; }

  Status NormalizeByTag(Normalization tag);

  void set_normalization_tag(Normalization tag) { normalization_ = tag; }

  HashedItem::PackingStrategy packing_strategy() const {
    return packing_strategy_;
  }

  virtual void set_packing_strategy(
      HashedItem::PackingStrategy packing_strategy) {
    packing_strategy_ = packing_strategy;
  }

  virtual bool is_float() const = 0;

  bool is_binary() const { return packing_strategy_ == HashedItem::BINARY; }

  virtual void set_is_binary(bool val) {
    packing_strategy_ = val ? HashedItem::BINARY : HashedItem::NONE;
  }

  virtual void ShrinkToFit() {}

  size_t DocidArrayCapacity() const { return docids_->capacity(); }

  virtual size_t MemoryUsageExcludingDocids() const = 0;

  size_t DocidMemoryUsage() const { return docids_->MemoryUsage(); }

  class Mutator;
  virtual StatusOr<typename Dataset::Mutator*> GetUntypedMutator() const = 0;

 protected:
  void set_dimensionality_no_checks(DimensionIndex dim) {
    dimensionality_ = dim;
  }

  void set_docids_no_checks(shared_ptr<DocidCollectionInterface> docids) {
    docids_ = std::move(docids);
  }

  void set_normalization(Normalization norm) { normalization_ = norm; }

  Status AppendDocid(string_view docid) { return docids_->Append(docid); }

 private:
  shared_ptr<DocidCollectionInterface> docids_;

  DimensionIndex dimensionality_ = 0;

  Normalization normalization_ = NONE;

  HashedItem::PackingStrategy packing_strategy_ = HashedItem::NONE;

  virtual void UnusedKeyMethod();
};

class Dataset::Mutator : public VirtualDestructor {
 public:
  virtual Status RemoveDatapoint(string_view docid) = 0;

  virtual bool LookupDatapointIndex(string_view docid,
                                    DatapointIndex* index) const = 0;

  virtual void Reserve(size_t size) = 0;

  virtual Status RemoveDatapoint(DatapointIndex index) = 0;
};

template <typename T>
class TypedDataset : public Dataset {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(TypedDataset);

  TypedDataset() {}

  explicit TypedDataset(unique_ptr<DocidCollectionInterface> docids)
      : Dataset(std::move(docids)) {}

  research_scann::TypeTag TypeTag() const final { return TagForType<T>(); }

  bool is_float() const final { return std::is_floating_point<T>::value; }

  using const_iterator = RandomAccessIterator<const TypedDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  virtual DatapointPtr<T> operator[](size_t datapoint_index) const = 0;

  DatapointPtr<T> at(size_t datapoint_index) const {
    CHECK_LT(datapoint_index, size());
    return operator[](datapoint_index);
  }

  virtual void set_dimensionality(DimensionIndex dimensionality) = 0;

  virtual Status Append(const DatapointPtr<T>& dptr, string_view docid) = 0;
  Status Append(const DatapointPtr<T>& dptr);

  virtual Status Append(const GenericFeatureVector& gfv, string_view docid) = 0;
  Status Append(const GenericFeatureVector& gfv);

  void AppendOrDie(const DatapointPtr<T>& dptr, string_view docid);
  void AppendOrDie(const GenericFeatureVector& gfv, string_view docid);
  void AppendOrDie(const DatapointPtr<T>& dptr);
  void AppendOrDie(const GenericFeatureVector& gfv);

  void GetDatapoint(size_t index, Datapoint<double>* result) const final;
  Status MeanByDimension(Datapoint<double>* result) const final;
  Status MeanByDimension(ConstSpan<DatapointIndex> subset,
                         Datapoint<double>* result) const final;
  void MeanVarianceByDimension(Datapoint<double>* means,
                               Datapoint<double>* variances) const final;
  void MeanVarianceByDimension(ConstSpan<DatapointIndex> subset,
                               Datapoint<double>* means,
                               Datapoint<double>* variances) const final;
  Status NormalizeUnitL2() final;
  Status NormalizeZeroMeanUnitVariance() final;

  class Mutator;
  virtual StatusOr<typename TypedDataset::Mutator*> GetMutator() const = 0;
  StatusOr<typename Dataset::Mutator*> GetUntypedMutator() const override {
    TF_ASSIGN_OR_RETURN(Dataset::Mutator * result, GetMutator());
    return result;
  }
};

template <typename T>
class TypedDataset<T>::Mutator : public Dataset::Mutator {
 public:
  virtual Status AddDatapoint(const DatapointPtr<T>& dptr,
                              string_view docid) = 0;

  virtual Status RemoveDatapoint(string_view docid) = 0;

  virtual Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                                 string_view docid) = 0;

  virtual bool LookupDatapointIndex(string_view docid,
                                    DatapointIndex* index) const = 0;

  virtual void Reserve(size_t size) = 0;

  virtual Status RemoveDatapoint(DatapointIndex index) = 0;
  virtual Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                                 DatapointIndex index) = 0;
};

template <typename T>
class DenseDataset final : public TypedDataset<T> {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(DenseDataset);

  DenseDataset() {}

  explicit DenseDataset(unique_ptr<DocidCollectionInterface> docids)
      : TypedDataset<T>(std::move(docids)) {}

  DenseDataset(std::vector<T> datapoint_vec,
               unique_ptr<DocidCollectionInterface> docids);

  DenseDataset(std::vector<T> datapoint_vec, size_t num_dp);

  DenseDataset<T> Copy() const {
    auto result = DenseDataset<T>(data_, this->docids()->Copy());
    result.set_normalization_tag(this->normalization());

    result.set_dimensionality(this->dimensionality());
    return result;
  }

  bool IsDense() const final { return true; }

  using const_iterator = RandomAccessIterator<const DenseDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  size_t n_elements() const {
    return static_cast<size_t>(this->size()) *
           static_cast<size_t>(this->dimensionality());
  }

  void Reserve(size_t n) final;
  void ReserveImpl(size_t n);

  void Resize(size_t n);

  template <typename Real>
  void ConvertType(DenseDataset<Real>* target) const;

  ConstSpan<T> data() const { return data_; }
  ConstSpan<T> data(size_t index) const {
    return MakeConstSpan(data_.data() + index * stride_, stride_);
  }
  MutableSpan<T> mutable_data() { return MakeMutableSpan(data_); }
  MutableSpan<T> mutable_data(size_t index) {
    return MakeMutableSpan(data_.data() + index * stride_, stride_);
  }

  vector<T> ClearRecyclingDataVector() {
    vector<T> result = std::move(data_);
    this->clear();
    return result;
  }

  void clear() final;
  DimensionIndex NumActiveDimensions() const final;
  void ShrinkToFit() final;
  size_t MemoryUsageExcludingDocids() const final;
  inline DatapointPtr<T> operator[](size_t i) const final;
  void set_dimensionality(DimensionIndex dimensionality) final;
  void set_is_binary(bool val) final;
  void GetDenseDatapoint(size_t index, Datapoint<double>* result) const final;
  inline void Prefetch(size_t index) const final;
  double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                     size_t vec2_index) const final;
  using TypedDataset<T>::Append;
  Status Append(const DatapointPtr<T>& dptr, string_view docid) final;
  Status Append(const GenericFeatureVector& gfv, string_view docid) final;
  shared_ptr<DocidCollectionInterface> ReleaseDocids() final;

  using TypedDataset<T>::AppendOrDie;

  void AppendOrDie(ConstSpan<T> values, string_view docid) {
    AppendOrDie(MakeDatapointPtr<T>(values), docid);
  }
  void AppendOrDie(ConstSpan<T> values) {
    AppendOrDie(MakeDatapointPtr<T>(values), absl::StrCat(this->size()));
  }

  StatusOr<typename TypedDataset<T>::Mutator*> GetMutator() const final;

 private:
  void SetStride();

  std::vector<T> data_;

  DimensionIndex stride_ = 0;

  mutable unique_ptr<typename DenseDataset<T>::Mutator> mutator_;

  template <typename U>
  friend class DenseDataset;
};

template <typename T>
class DenseDatasetSubView;

template <typename T>
class DenseDatasetView : VirtualDestructor {
 public:
  DenseDatasetView() {}

  virtual const T* GetPtr(size_t i) const = 0;

  virtual size_t dimensionality() const = 0;

  virtual size_t size() const = 0;

  virtual research_scann::TypeTag TypeTag() const { return TagForType<T>(); }

  virtual bool IsConsecutiveStorage() const { return false; }

  virtual std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                                       size_t size) const {
    return absl::make_unique<DenseDatasetSubView<T>>(this, offset, size);
  }
};

template <typename T>
class DefaultDenseDatasetView : public DenseDatasetView<T> {
 public:
  DefaultDenseDatasetView() {}

  explicit DefaultDenseDatasetView(const DenseDataset<T>& ds)
      : ptr_(ds.data().data()), size_(ds.size()) {
    if (ds.packing_strategy() == HashedItem::BINARY) {
      dims_ = ds.dimensionality() / 8 + (ds.dimensionality() % 8 > 0);
    } else if (ds.packing_strategy() == HashedItem::NIBBLE) {
      dims_ = ds.dimensionality() / 2 + (ds.dimensionality() % 2 > 0);
    } else {
      dims_ = ds.dimensionality();
    }
  }

  explicit DefaultDenseDatasetView(ConstSpan<T> span, size_t dimensionality)
      : ptr_(span.data()),
        dims_(dimensionality),
        size_(span.size() / dimensionality) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return ptr_ + i * dims_;
  }

  SCANN_INLINE size_t dimensionality() const final { return dims_; }

  SCANN_INLINE size_t size() const final { return size_; }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    return absl::WrapUnique(
        new DefaultDenseDatasetView<T>(ptr_ + offset * dims_, dims_, size));
  }

  bool IsConsecutiveStorage() const override { return true; }

 private:
  DefaultDenseDatasetView(const T* ptr, size_t dim, size_t size)
      : ptr_(ptr), dims_(dim), size_(size) {}

  const T* __restrict__ ptr_ = nullptr;
  size_t dims_ = 0;
  size_t size_ = 0;
};

template <typename T>
class DenseDatasetSubView : public DenseDatasetView<T> {
 public:
  DenseDatasetSubView(const DenseDatasetView<T>* parent, size_t offset,
                      size_t size)
      : parent_view_(parent), offset_(offset), size_(size) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return parent_view_->GetPtr(offset_ + i);
  }

  SCANN_INLINE size_t dimensionality() const final {
    return parent_view_->dimensionality();
  };

  SCANN_INLINE size_t size() const final { return size_; }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    return absl::make_unique<DenseDatasetSubView<T>>(parent_view_,
                                                     offset + offset_, size);
  }

  bool IsConsecutiveStorage() const override {
    return parent_view_->IsConsecutiveStorage();
  }

 private:
  const DenseDatasetView<T>* __restrict__ parent_view_ = nullptr;
  const size_t offset_ = 0;
  const size_t size_ = 0;
};

template <typename T>
class SparseDataset final : public TypedDataset<T> {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(SparseDataset);

  SparseDataset() {}

  explicit SparseDataset(unique_ptr<DocidCollectionInterface> docids)
      : TypedDataset<T>(std::move(docids)) {}

  explicit SparseDataset(DimensionIndex dimensionality) : SparseDataset() {
    this->set_dimensionality(dimensionality);
  }

  bool IsDense() const final { return false; }

  using const_iterator = RandomAccessIterator<const SparseDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  using TypedDataset<T>::Append;

  Status Append(const GenericFeatureVector& gfv, string_view docid) final;
  Status Append(const DatapointPtr<T>& dptr, string_view docid) final;

  using TypedDataset<T>::AppendOrDie;

  void AppendOrDie(ConstSpan<DimensionIndex> indices, ConstSpan<T> values,
                   string_view docid = "") {
    AppendOrDie(MakeDatapointPtr<T>(indices, values, this->dimensionality()),
                docid);
  }

  void Reserve(size_t n_points) final;

  void Reserve(size_t n_points, size_t n_entries);

  DimensionIndex NonzeroEntriesForDatapoint(DatapointIndex i) const {
    return repr_.NonzeroEntriesForDatapoint(i);
  }

  size_t num_entries() const { return repr_.indices().size(); }

  bool AllValuesNonNegative() const {
    return std::is_unsigned<T>::value || repr_.values().empty() ||
           *std::min_element(repr_.values().begin(), repr_.values().end()) >= 0;
  }

  void clear() final;

  void ConvertType(SparseDataset<double>* target);

  inline DatapointPtr<T> operator[](size_t i) const final;
  void set_dimensionality(DimensionIndex dimensionality) final;
  DimensionIndex NumActiveDimensions() const final;
  void GetDenseDatapoint(size_t index, Datapoint<double>* result) const final;
  inline void Prefetch(size_t index) const final;
  double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                     size_t vec2_index) const final;
  size_t MemoryUsageExcludingDocids() const final;
  void ShrinkToFit() final;

  StatusOr<typename TypedDataset<T>::Mutator*> GetMutator() const final {
    return UnimplementedError("Sparse dataset does not support mutation.");
  }

 private:
  Status AppendImpl(const GenericFeatureVector& gfv, string_view docid);
  Status AppendImpl(const DatapointPtr<T>& dptr, string_view docid);

  mutable SparseDatasetLowLevel<DimensionIndex, T> repr_;

  template <typename U>
  friend class SparseDataset;
};

template <typename T>
DatapointPtr<T> DenseDataset<T>::operator[](size_t i) const {
  DCHECK_LT(i, this->size());
  return MakeDatapointPtr(nullptr, data_.data() + i * stride_, stride_,
                          this->dimensionality());
}

template <typename T>
void DenseDataset<T>::Prefetch(size_t i) const {
  DCHECK_LT(i, this->size());
  ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
      reinterpret_cast<const char*>(data_.data() + i * stride_));
}

template <typename T>
template <typename Real>
void DenseDataset<T>::ConvertType(DenseDataset<Real>* target) const {
  static_assert(std::is_floating_point<Real>(),
                "Real template parameter must be either float or double for "
                "DenseDataset::ConvertType.");
  CHECK(!this->is_binary()) << "Not implemented for binary datasets.";
  DCHECK(target);
  target->clear();
  target->set_dimensionality_no_checks(this->dimensionality());
  target->stride_ = stride_;
  target->set_docids_no_checks(this->docids()->Copy());
  target->data_.insert(target->data_.begin(), data_.begin(), data_.end());
}

template <typename T>
DatapointPtr<T> SparseDataset<T>::operator[](size_t i) const {
  DCHECK_LT(i, this->size());
  auto low_level_result = repr_.Get(i);
  return MakeDatapointPtr(low_level_result.indices, low_level_result.values,
                          low_level_result.nonzero_entries,
                          this->dimensionality());
}

template <typename T>
void SparseDataset<T>::Prefetch(size_t i) const {
  DCHECK_LT(i, this->size());
  repr_.Prefetch(i);
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, TypedDataset);
SCANN_INSTANTIATE_TYPED_CLASS(extern, SparseDataset);
SCANN_INSTANTIATE_TYPED_CLASS(extern, DenseDataset);

}  // namespace research_scann

#endif
