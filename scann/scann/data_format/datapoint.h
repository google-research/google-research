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



#ifndef SCANN_DATA_FORMAT_DATAPOINT_H_
#define SCANN_DATA_FORMAT_DATAPOINT_H_

#include <cstdint>

#include "scann/data_format/features.pb.h"
#include "scann/data_format/gfv_conversion.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/infinite_one_array.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
class DatapointPtr;
template <typename T>
class Datapoint;

template <>
class DatapointPtr<NoValue> final {
 public:
  DatapointPtr() {}

  DatapointPtr(const DimensionIndex* indices, const NoValue* values,
               DimensionIndex nonzero_entries, DimensionIndex dimensionality)
      : indices_(indices),
        nonzero_entries_(nonzero_entries),
        dimensionality_(dimensionality) {
    CHECK(!values) << "values must be nullptr";
  }

  const DimensionIndex* indices() const { return indices_; }
  ConstSpan<DimensionIndex> indices_slice() const {
    return ConstSpan<DimensionIndex>(indices_, indices_ ? nonzero_entries_ : 0);
  }
  bool has_values() const { return false; }
  InfiniteOneArray<NoValue> values() const {
    return InfiniteOneArray<NoValue>();
  }
  InfiniteOneArray<NoValue> values_slice() const {
    return InfiniteOneArray<NoValue>();
  }
  bool IsDense() const { return false; }
  bool IsSparse() const { return true; }
  bool IsAllOnes() const { return true; }
  bool IsFinite() const { return true; }
  DimensionIndex nonzero_entries() const { return nonzero_entries_; }
  DatapointPtr<NoValue> ToSparseBinary() const { return *this; }

 private:
  const DimensionIndex* indices_ = nullptr;
  DimensionIndex nonzero_entries_ = 0;
  DimensionIndex dimensionality_ = 0;
};

template <typename T>
class DatapointPtr final {
 public:
  DatapointPtr() {}

  DatapointPtr(const DimensionIndex* indices, const T* values,
               DimensionIndex nonzero_entries, DimensionIndex dimensionality)
      : indices_(indices),
        values_(values),
        nonzero_entries_(nonzero_entries),
        dimensionality_(dimensionality) {}

  const DimensionIndex* indices() const { return indices_; }

  ConstSpan<DimensionIndex> indices_slice() const {
    return ConstSpan<DimensionIndex>(indices_, indices_ ? nonzero_entries_ : 0);
  }

  const T* values() const { return values_; }

  bool has_values() const { return values_; }

  ConstSpan<T> values_slice() const {
    return ConstSpan<T>(values_, values_ ? nonzero_entries_ : 0);
  }

  DimensionIndex nonzero_entries() const { return nonzero_entries_; }

  DimensionIndex dimensionality() const { return dimensionality_; }

  bool IsDense() const { return nonzero_entries_ > 0 && indices_ == nullptr; }

  bool IsSparse() const { return !IsDense(); }

  bool IsSparseOrigin() const { return nonzero_entries_ == 0; }

  bool IsAllOnes() const {
    return std::all_of(values_slice().begin(), values_slice().end(),
                       [](T val) { return val == 1; });
  }

  bool IsFinite() const {
    if constexpr (IsFloatingType<T>()) {
      for (T val : values_slice()) {
        if (!std::isfinite(val)) return false;
      }
    }
    return true;
  }

  bool HasNonzero(DimensionIndex dimension_index) const;

  T GetElement(DimensionIndex dimension_index) const;

  T GetElementPacked(DimensionIndex dimension_index) const;

  GenericFeatureVector ToGfv() const;

  DatapointPtr<NoValue> ToSparseBinary() const {
    return DatapointPtr<NoValue>(indices_, nullptr, nonzero_entries_,
                                 dimensionality_);
  }

 private:
  void ToGfvIndicesAndMetadata(GenericFeatureVector* gfv) const;

  const DimensionIndex* indices_ = nullptr;

  const T* values_ = nullptr;

  DimensionIndex nonzero_entries_ = 0;

  DimensionIndex dimensionality_ = 0;
};

constexpr DimensionIndex kSparseDimensionality =
    numeric_limits<DimensionIndex>::max();

template <typename T>
inline DatapointPtr<T> MakeDatapointPtr(const DimensionIndex* indices,
                                        const T* values,
                                        DimensionIndex nonzero_entries,
                                        DimensionIndex dimensionality) {
  return DatapointPtr<T>(indices, values, nonzero_entries, dimensionality);
}

template <typename T>
inline DatapointPtr<T> MakeDatapointPtr(ConstSpan<T> values) {
  return MakeDatapointPtr(nullptr, values.data(), values.size(), values.size());
}
template <typename T>
inline DatapointPtr<T> MakeDatapointPtr(const T* ptr, DimensionIndex size) {
  return MakeDatapointPtr(nullptr, ptr, size, size);
}

template <int&... ExplicitArgumentBarrier, typename CollectionT,
          typename EnableIfConvertibleToConstSpan_SFINAE =
              decltype(std::declval<const CollectionT&>().data())>
inline auto MakeDatapointPtr(const CollectionT& c)
    -> decltype(MakeDatapointPtr(MakeConstSpan(c))) {
  return MakeDatapointPtr(MakeConstSpan(c));
}

inline DatapointPtr<uint8_t> MakeDenseBinaryDatapointPtr(
    ConstSpan<uint8_t> values, DimensionIndex dimensionality) {
  return MakeDatapointPtr(nullptr, values.data(), values.size(),
                          dimensionality);
}

inline DatapointPtr<uint8_t> MakeSparseBinaryDatapointPtr(
    ConstSpan<DimensionIndex> indices,
    DimensionIndex dimensionality = kSparseDimensionality) {
  return MakeDatapointPtr<uint8_t>(indices.data(), nullptr, indices.size(),
                                   dimensionality);
}

template <typename T>
inline DatapointPtr<T> MakeDatapointPtr(
    ConstSpan<DimensionIndex> indices, ConstSpan<T> values,
    DimensionIndex dimensionality = kSparseDimensionality) {
  constexpr DimensionIndex* kIndicesNullptr = nullptr;
  constexpr T* kValuesNullptr = nullptr;
  const bool has_indices = !indices.empty();
  const bool has_values = !values.empty();
  if (has_indices && has_values) {
    CHECK_EQ(values.size(), indices.size());
    return MakeDatapointPtr(indices.data(), values.data(), indices.size(),
                            dimensionality);
  }
  if (has_indices && !has_values) {
    return MakeDatapointPtr(indices.data(), kValuesNullptr, indices.size(),
                            dimensionality);
  }
  if (!has_indices && has_values) {
    if (dimensionality != kSparseDimensionality) {
      CHECK_EQ(values.size(), dimensionality);
    }
    return MakeDatapointPtr(values);
  }
  DCHECK(!has_indices && !has_values);

  return MakeDatapointPtr<T>(kIndicesNullptr, kValuesNullptr, 0,
                             dimensionality);
}

template <int&... ExplicitArgumentBarrier, typename CollectionT>
inline auto MakeDatapointPtr(ConstSpan<DimensionIndex> indices,
                             const CollectionT& values)
    -> decltype(MakeDatapointPtr(indices, MakeConstSpan(values))) {
  return MakeDatapointPtr(indices, MakeConstSpan(values));
}

inline DatapointPtr<uint8_t> MakeDatapointPtr(const HashedItem& item) {
  return MakeDatapointPtr(
      nullptr, reinterpret_cast<const uint8_t*>(item.indicator_vars().data()),
      item.indicator_vars().size(), item.indicator_vars().size());
}

template <typename T>
class Datapoint final {
 public:
  Datapoint() : dimensionality_(0), normalization_(NONE) {}

  Datapoint(ConstSpan<DimensionIndex> indices, ConstSpan<T> values,
            DimensionIndex dimensionality)
      : indices_(indices.begin(), indices.end()),
        values_(values.begin(), values.end()),
        dimensionality_(dimensionality),
        normalization_(NONE) {}

  Datapoint(std::vector<DimensionIndex> indices, std::vector<T> values,
            DimensionIndex dimensionality)
      : indices_(std::move(indices)),
        values_(std::move(values)),
        dimensionality_(dimensionality),
        normalization_(NONE) {}

  Status FromGfv(const GenericFeatureVector& gfv);

  std::vector<DimensionIndex>* mutable_indices() { return &indices_; }
  MutableSpan<DimensionIndex> mutable_indices_slice() {
    return MutableSpan<DimensionIndex>(indices_);
  }

  const std::vector<DimensionIndex>& indices() const { return indices_; }
  ConstSpan<DimensionIndex> indices_slice() const { return indices_; }

  std::vector<T>* mutable_values() { return &values_; }
  MutableSpan<T> mutable_values_slice() { return MutableSpan<T>(values_); }

  const std::vector<T>& values() const { return values_; }
  ConstSpan<T> values_slice() const { return values_; }

  DimensionIndex nonzero_entries() const {
    return (IsDense()) ? values_.size() : indices_.size();
  }

  DimensionIndex dimensionality() const {
    return (dimensionality_ == 0) ? nonzero_entries() : dimensionality_;
  }

  void set_dimensionality(DimensionIndex new_value) {
    dimensionality_ = new_value;
  }

  bool IsDense() const { return indices_.empty() && !values_.empty(); }

  bool IsSparse() const { return !IsDense(); }

  bool IsSparseBinary() const { return values_.empty(); }

  void MakeNotBinary();

  DatapointPtr<T> ToPtr() const {
    const DimensionIndex* indices_data =
        indices_.empty() ? nullptr : indices_.data();
    const T* values_data = values_.empty() ? nullptr : values_.data();
    return MakeDatapointPtr(indices_data, values_data, nonzero_entries(),
                            dimensionality());
  }

  GenericFeatureVector ToGfv() const;

  void clear() {
    indices_.clear();
    values_.clear();
    dimensionality_ = 0;
    normalization_ = NONE;
  }

  void ZeroFill(size_t dimensionality) {
    clear();
    values_.resize(dimensionality);
  }

  Normalization normalization() const { return normalization_; }

  void set_normalization(Normalization val) { normalization_ = val; }

  void Swap(Datapoint<T>* rhs) { std::swap(*this, *rhs); }

  bool IndicesSorted() const;

  void SortIndices();

  void RemoveExplicitZeroesFromSparseVector();

 private:
  Status FromGfvImpl(const GenericFeatureVector& gfv);

  std::vector<DimensionIndex> indices_;

  std::vector<T> values_;

  DimensionIndex dimensionality_;

  Normalization normalization_;
};

template <typename T>
Datapoint<T> MakeDenseDatapoint(ConstSpan<T> values) {
  ConstSpan<DimensionIndex> indices(nullptr, 0);
  return Datapoint<T>(indices, values, values.size());
}

template <typename T>
Datapoint<T> MakeSparseDatapoint(ConstSpan<DimensionIndex> indices,
                                 ConstSpan<T> values,
                                 DimensionIndex dimensionality) {
  CHECK_EQ(indices.size(), values.size());
  return Datapoint<T>(indices, values, dimensionality);
}

template <typename T>
T DatapointPtr<T>::GetElementPacked(DimensionIndex dimension_index) const {
  DCHECK_GT(dimensionality(), nonzero_entries());
  const auto array_offset = dimension_index / 8;
  const auto bit_offset = dimension_index % 8;
  return (values()[array_offset] >> bit_offset) & 1;
}

template <>
inline float DatapointPtr<float>::GetElementPacked(
    DimensionIndex dimension_index) const {
  LOG(FATAL) << "Can't happen.";
}

template <>
inline double DatapointPtr<double>::GetElementPacked(
    DimensionIndex dimension_index) const {
  LOG(FATAL) << "Can't happen.";
}

template <typename T>
GenericFeatureVector DatapointPtr<T>::ToGfv() const {
  GenericFeatureVector result;
  ToGfvIndicesAndMetadata(&result);

  if (dimensionality() == nonzero_entries() || IsSparse()) {
    if (values() == nullptr) {
      result.set_feature_type(GenericFeatureVector::BINARY);
    } else {
      result.set_feature_type(GenericFeatureVector::INT64);
      for (size_t i = 0; i < nonzero_entries(); ++i) {
        result.add_feature_value_int64(values()[i]);
      }
    }
  } else {
    DCHECK_GT(dimensionality(), nonzero_entries());
    DCHECK(IsUint8<T>());
    result.set_feature_type(GenericFeatureVector::BINARY);
    UnpackBinaryToInt64<T>(ConstSpan<T>(values(), nonzero_entries()),
                           dimensionality(), &result);
  }

  return result;
}

template <>
inline GenericFeatureVector DatapointPtr<float>::ToGfv() const {
  GenericFeatureVector result;
  result.set_feature_type(GenericFeatureVector::FLOAT);
  ToGfvIndicesAndMetadata(&result);
  for (size_t i = 0; i < nonzero_entries(); ++i) {
    result.add_feature_value_float(values()[i]);
  }

  return result;
}
template <>
inline GenericFeatureVector DatapointPtr<double>::ToGfv() const {
  GenericFeatureVector result;
  result.set_feature_type(GenericFeatureVector::DOUBLE);
  ToGfvIndicesAndMetadata(&result);
  for (size_t i = 0; i < nonzero_entries(); ++i) {
    result.add_feature_value_double(values()[i]);
  }

  return result;
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, DatapointPtr);
SCANN_INSTANTIATE_TYPED_CLASS(extern, Datapoint);

}  // namespace research_scann

#endif
