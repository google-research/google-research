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



#ifndef SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_IMPL_H_
#define SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_IMPL_H_

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/training_options_base.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

template <typename FloatT>
StatusOr<double> ComputeNormBiasCorrection(
    const DenseDataset<FloatT>& db, DatapointPtr<double> center,
    ConstSpan<DatapointIndex> cluster_members) {
  if (cluster_members.empty()) return 1.0;
  SCANN_RETURN_IF_ERROR(VerifyAllFinite(center.values_span()));
  double mean_norm = 0.0;
  for (DatapointIndex idx : cluster_members) {
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(db[idx].values_span()));
    mean_norm += std::sqrt(SquaredL2Norm(db[idx]));
  }
  SCANN_RET_CHECK(std::isfinite(mean_norm)) << mean_norm;
  mean_norm /= cluster_members.size();
  SCANN_RET_CHECK(std::isfinite(mean_norm))
      << mean_norm << " " << cluster_members.size();
  const double center_norm = std::sqrt(SquaredL2Norm(center));
  SCANN_RET_CHECK(std::isfinite(center_norm)) << center_norm;
  return (center_norm == 0.0) ? 1.0 : (mean_norm / center_norm);
}

template <typename T>
inline vector<DenseDataset<FloatingTypeFor<T>>> ConvertCentersIfNecessary(
    std::vector<DenseDataset<double>> double_centers) {
  std::vector<DenseDataset<float>> result(double_centers.size());
  for (size_t i = 0; i < double_centers.size(); ++i) {
    double_centers[i].ConvertType(&result[i]);
  }

  return result;
}

template <>
inline std::vector<DenseDataset<double>> ConvertCentersIfNecessary<double>(
    std::vector<DenseDataset<double>> double_centers) {
  return double_centers;
}

template <typename T>
struct AhImpl {
  using FloatT = FloatingTypeFor<T>;
  using TrainingOptionsT = asymmetric_hashing2::TrainingOptionsTyped<T>;

  static StatusOr<std::vector<DenseDataset<double>>> TrainAsymmetricHashing(
      const TypedDataset<T>& dataset, const TrainingOptionsT& opts,
      shared_ptr<ThreadPool> pool);

  static Status IndexDatapoint(const DatapointPtr<T>& input,
                               const ChunkingProjection<T>& projection,
                               const DistanceMeasure& quantization_distance,
                               ConstSpan<DenseDataset<FloatT>> centers,
                               Datapoint<uint8_t>* result);

  static Status IndexDatapoint(const DatapointPtr<T>& input,
                               const ChunkingProjection<T>& projection,
                               const DistanceMeasure& quantization_distance,
                               ConstSpan<DenseDataset<FloatT>> centers,
                               MutableSpan<uint8_t> result);

  static Status IndexDatapointNoiseShaped(
      const DatapointPtr<T>& maybe_residual_dptr,
      const DatapointPtr<T>& original_dptr,
      const ChunkingProjection<T>& projection,
      ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers, double threshold,
      double eta, MutableSpan<uint8_t> result);

  static StatusOr<std::vector<float>> CreateRawFloatLookupTable(
      const DatapointPtr<T>& query, const ChunkingProjection<T>& projection,
      const DistanceMeasure& lookup_distance,
      ConstSpan<DenseDataset<FloatT>> centers,
      ConstSpan<FloatT> block_transposed_centers,
      int32_t num_clusters_per_block);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, AhImpl);

template <typename T>
StatusOr<std::vector<DenseDataset<double>>> TrainAsymmetricHashing(
    const TypedDataset<T>& dataset,
    const asymmetric_hashing2::TrainingOptionsTyped<T>& opts,
    shared_ptr<ThreadPool> pool) {
  return AhImpl<T>::TrainAsymmetricHashing(dataset, opts, std::move(pool));
}

template <typename T>
Status IndexDatapoint(const DatapointPtr<T>& input,
                      const ChunkingProjection<T>& projection,
                      const DistanceMeasure& quantization_distance,
                      ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
                      Datapoint<uint8_t>* result) {
  return AhImpl<T>::IndexDatapoint(input, projection, quantization_distance,
                                   centers, result);
}

template <typename T>
Status IndexDatapoint(const DatapointPtr<T>& input,
                      const ChunkingProjection<T>& projection,
                      const DistanceMeasure& quantization_distance,
                      ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
                      MutableSpan<uint8_t> result) {
  return AhImpl<T>::IndexDatapoint(input, projection, quantization_distance,
                                   centers, result);
}

template <typename T>
StatusOr<std::vector<float>> CreateRawFloatLookupTable(
    const DatapointPtr<T>& query, const ChunkingProjection<T>& projection,
    const DistanceMeasure& lookup_distance,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
    ConstSpan<FloatingTypeFor<T>> block_transposed_centers,
    int32_t num_clusters_per_block) {
  return AhImpl<T>::CreateRawFloatLookupTable(
      query, projection, lookup_distance, centers, block_transposed_centers,
      num_clusters_per_block);
}

template <typename Uint>
inline constexpr Uint FixedPointBias() {
  return static_cast<Uint>(1) << ((sizeof(Uint) * 8) - 1);
}

template <typename T>
std::vector<T> ConvertLookupToFixedPoint(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&
        conversion_options,
    float* multiplier);

extern template vector<uint8_t> ConvertLookupToFixedPoint<uint8_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);
extern template vector<uint16_t> ConvertLookupToFixedPoint<uint16_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);

template <typename T>
vector<T> ConvertLookupToFixedPoint(ConstSpan<float> raw_lookup,
                                    float* multiplier) {
  return ConvertLookupToFixedPoint<T>(
      raw_lookup, AsymmetricHasherConfig::FixedPointLUTConversionOptions(),
      multiplier);
}

bool CanUseInt16Accumulator(ConstSpan<uint8_t> lookup_table, size_t num_blocks);

template <typename LookupElement>
inline enable_if_t<IsIntegerType<LookupElement>(), int32_t>
ComputePossiblyFixedPointMaxDistance(float float_max_distance,
                                     float multiplier) {
  if (float_max_distance == numeric_limits<float>::infinity()) {
    return numeric_limits<int32_t>::max();
  } else if (float_max_distance * multiplier >=
             static_cast<float>(numeric_limits<int32_t>::max())) {
    return numeric_limits<int32_t>::max();
  } else {
    return static_cast<int32_t>(std::floor(float_max_distance * multiplier));
  }
}

template <typename LookupElement>
inline enable_if_t<IsFloatingType<LookupElement>(), float>
ComputePossiblyFixedPointMaxDistance(float float_max_distance,
                                     float multiplier) {
  return float_max_distance;
}

std::vector<uint8_t> CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database);

template <typename LookupElement>
struct DistanceType {
  using type = float;
};

template <>
struct DistanceType<uint8_t> {
  using type = int32_t;
};

template <>
struct DistanceType<uint16_t> {
  using type = int32_t;
};

template <typename TopN, typename Distance, typename DistancePostprocess>
class AddPostprocessedValueToTopN {
 public:
  AddPostprocessedValueToTopN(TopN* top_n, Distance max_distance,
                              DistancePostprocess postprocess)
      : top_n_(top_n), max_distance_(max_distance), postprocess_(postprocess) {}

  template <typename RawDistance>
  SCANN_INLINE void Postprocess(RawDistance distance, DatapointIndex dp_index) {
    Distance postprocessed = postprocess_.Postprocess(distance, dp_index);
    if (ABSL_PREDICT_FALSE(postprocessed <= max_distance_)) {
      PostprocessImpl(postprocessed, dp_index);
    }
  }

 private:
  void PostprocessImpl(Distance postprocessed, DatapointIndex dp_index) {
    top_n_->push(std::make_pair(dp_index, postprocessed));
    if (top_n_->full()) {
      max_distance_ = top_n_->approx_bottom().second;
    }
  }

  TopN* top_n_;

  Distance max_distance_;

  DistancePostprocess postprocess_;
};

template <size_t kUnrollFactorParam, typename Functor>
class UnrestrictedIndexIterator {
 public:
  static constexpr size_t kUnrollFactor = kUnrollFactorParam;

  explicit UnrestrictedIndexIterator(DatapointIndex n, Functor functor)
      : n_(n), functor_(functor) {}

  void Advance() { i_ += kUnrollFactor; }

  bool FullUnrollLeft() const { return i_ + kUnrollFactor <= n_; }

  size_t num_left() const { return n_ - i_; }

  DatapointIndex GetOffsetIndex(DatapointIndex offset) const {
    return i_ + offset;
  }

  template <typename T>
  void Postprocess(T score, DatapointIndex offset) {
    functor_.Postprocess(score, GetOffsetIndex(offset));
  }

 private:
  size_t i_ = 0;

  const size_t n_;

  Functor functor_;
};

template <size_t kUnrollFactorParam, typename Functor>
class PopulateDistancesIterator {
 public:
  static constexpr size_t kUnrollFactor = kUnrollFactorParam;

  PopulateDistancesIterator(MutableSpan<pair<DatapointIndex, float>> results,
                            Functor functor)
      : results_(results), functor_(functor) {}

  void Advance() { i_ += kUnrollFactor; }

  bool FullUnrollLeft() const { return i_ + kUnrollFactor <= results_.size(); }

  size_t num_left() const { return results_.size() - i_; }

  DatapointIndex GetOffsetIndex(DatapointIndex offset) const {
    return results_[i_ + offset].first;
  }

  template <typename T>
  void Postprocess(T score, DatapointIndex offset) {
    results_[i_ + offset].second =
        functor_.Postprocess(score, GetOffsetIndex(offset));
  }

 private:
  MutableSpan<pair<DatapointIndex, float>> results_;

  size_t i_ = 0;

  Functor functor_;
};

extern template class UnrestrictedIndexIterator<6, IdentityPostprocessFunctor>;
extern template class UnrestrictedIndexIterator<6, AddBiasFunctor>;
extern template class UnrestrictedIndexIterator<6, LimitedInnerFunctor>;
extern template class PopulateDistancesIterator<6, IdentityPostprocessFunctor>;
extern template class PopulateDistancesIterator<6, AddBiasFunctor>;
extern template class PopulateDistancesIterator<6, LimitedInnerFunctor>;

template <typename T>
uint32_t ComputeTotalBias(size_t num_blocks) {
  return 0;
}

template <>
inline uint32_t ComputeTotalBias<uint16_t>(size_t num_blocks) {
  return num_blocks * FixedPointBias<uint16_t>();
}

template <>
inline uint32_t ComputeTotalBias<uint8_t>(size_t num_blocks) {
  return num_blocks * FixedPointBias<uint8_t>();
}

template <typename T>
struct make_unsigned_if_int_struct {
  using type = make_unsigned_t<T>;
};

template <>
struct make_unsigned_if_int_struct<float> {
  using type = float;
};

template <typename T>
using make_unsigned_if_int_t = typename make_unsigned_if_int_struct<T>::type;

template <typename DatasetView, typename LookupElement,
          size_t kCompileTimeNumCenters, typename IndexIterator, bool kPrefetch>
SCANN_OUTLINE void
GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCentersImpl(
    ConstSpan<LookupElement> lookup, size_t runtime_num_centers,
    const DatasetView* __restrict__ hashed_database, IndexIterator it) {
  using Dist = typename DistanceType<LookupElement>::type;
  using UnsignedDist = make_unsigned_if_int_t<Dist>;
  static_assert(IndexIterator::kUnrollFactor == 6,
                "Mismatch between unroll factor of IndexIterator and "
                "unroll factor used in "
                "GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters.");

  const size_t num_blocks = hashed_database->dimensionality();

  const size_t num_centers = (kCompileTimeNumCenters == 0)
                                 ? runtime_num_centers
                                 : kCompileTimeNumCenters;

  const UnsignedDist total_bias = ComputeTotalBias<LookupElement>(num_blocks);
  auto remove_bias = [total_bias](UnsignedDist d) -> Dist {
    return d - total_bias;
  };

  DCHECK_GT(num_blocks, 0);

  constexpr size_t kCacheLineSize = 64;
  const size_t num_cache_lines_per_dp =
      kPrefetch ? DivRoundUp(num_blocks, kCacheLineSize) : 0;

  for (; it.FullUnrollLeft(); it.Advance()) {
    if constexpr (kPrefetch) {
      const size_t num_prefetch =
          std::min(it.kUnrollFactor, it.num_left() - it.kUnrollFactor);
      for (size_t prefetch_idx : Seq(num_prefetch)) {
        const size_t offset =
            it.GetOffsetIndex(prefetch_idx + it.kUnrollFactor);
        const uint8_t* hashed_database_point = hashed_database->GetPtr(offset);
        for (size_t cache_line_idx : Seq(num_cache_lines_per_dp)) {
          absl::PrefetchToLocalCache(hashed_database_point +
                                     cache_line_idx * kCacheLineSize);
        }
      }
    }
    const uint8_t* hashed_database_point0 =
        hashed_database->GetPtr(it.GetOffsetIndex(0));
    const uint8_t* hashed_database_point1 =
        hashed_database->GetPtr(it.GetOffsetIndex(1));
    const uint8_t* hashed_database_point2 =
        hashed_database->GetPtr(it.GetOffsetIndex(2));
    const uint8_t* hashed_database_point3 =
        hashed_database->GetPtr(it.GetOffsetIndex(3));
    const uint8_t* hashed_database_point4 =
        hashed_database->GetPtr(it.GetOffsetIndex(4));
    const uint8_t* hashed_database_point5 =
        hashed_database->GetPtr(it.GetOffsetIndex(5));
    const LookupElement* cur_lookup_row =
        lookup.data() + num_centers * (num_blocks - 1);
    ssize_t j = num_blocks - 1;
    DCHECK_LT(hashed_database_point0[j], num_centers);
    UnsignedDist sum0 = cur_lookup_row[hashed_database_point0[j]];
    DCHECK_LT(hashed_database_point1[j], num_centers);
    UnsignedDist sum1 = cur_lookup_row[hashed_database_point1[j]];
    DCHECK_LT(hashed_database_point2[j], num_centers);
    UnsignedDist sum2 = cur_lookup_row[hashed_database_point2[j]];
    DCHECK_LT(hashed_database_point3[j], num_centers);
    UnsignedDist sum3 = cur_lookup_row[hashed_database_point3[j]];
    DCHECK_LT(hashed_database_point4[j], num_centers);
    UnsignedDist sum4 = cur_lookup_row[hashed_database_point4[j]];
    DCHECK_LT(hashed_database_point5[j], num_centers);
    UnsignedDist sum5 = cur_lookup_row[hashed_database_point5[j]];
    cur_lookup_row -= num_centers;
    --j;

    for (; j >= 0; --j) {
      DCHECK_LT(hashed_database_point0[j], num_centers);
      sum0 += cur_lookup_row[hashed_database_point0[j]];
      DCHECK_LT(hashed_database_point1[j], num_centers);
      sum1 += cur_lookup_row[hashed_database_point1[j]];
      DCHECK_LT(hashed_database_point2[j], num_centers);
      sum2 += cur_lookup_row[hashed_database_point2[j]];
      DCHECK_LT(hashed_database_point3[j], num_centers);
      sum3 += cur_lookup_row[hashed_database_point3[j]];
      DCHECK_LT(hashed_database_point4[j], num_centers);
      sum4 += cur_lookup_row[hashed_database_point4[j]];
      DCHECK_LT(hashed_database_point5[j], num_centers);
      sum5 += cur_lookup_row[hashed_database_point5[j]];
      cur_lookup_row -= num_centers;
    }

    it.Postprocess(remove_bias(sum0), 0);
    it.Postprocess(remove_bias(sum1), 1);
    it.Postprocess(remove_bias(sum2), 2);
    it.Postprocess(remove_bias(sum3), 3);
    it.Postprocess(remove_bias(sum4), 4);
    it.Postprocess(remove_bias(sum5), 5);
  }

  for (DatapointIndex offset = 0; offset < it.num_left(); ++offset) {
    const LookupElement* cur_lookup_row = lookup.data();
    const uint8_t* hashed_database_point =
        hashed_database->GetPtr(it.GetOffsetIndex(offset));
    DCHECK_LT(hashed_database_point[0], num_centers);
    UnsignedDist sum = cur_lookup_row[hashed_database_point[0]];
    cur_lookup_row += num_centers;
    for (size_t j = 1; j < num_blocks; ++j) {
      DCHECK_LT(hashed_database_point[j], num_centers);
      sum += cur_lookup_row[hashed_database_point[j]];
      cur_lookup_row += num_centers;
    }

    it.Postprocess(remove_bias(sum), offset);
  }
}

template <typename DatasetView, typename LookupElement, typename IndexIterator,
          bool kPrefetch = false>
SCANN_INLINE void GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters(
    ConstSpan<LookupElement> lookup, size_t runtime_num_centers,
    const DatasetView* __restrict__ hashed_database, IndexIterator it) {
  if constexpr (std::is_same_v<LookupElement, uint8_t>) {
    if (ABSL_PREDICT_TRUE(runtime_num_centers == 256)) {
      return GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCentersImpl<
          DatasetView, LookupElement, 256, IndexIterator, kPrefetch>(
          lookup, runtime_num_centers, hashed_database, it);
    }
  }
  return GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCentersImpl<
      DatasetView, LookupElement, 0, IndexIterator, kPrefetch>(
      lookup, runtime_num_centers, hashed_database, it);
}

#define SCANN_SINGLE_ARG(...) __VA_ARGS__

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL0(                                   \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, IndexIterator)   \
  extern_or_nothing template void                                              \
  GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCentersImpl<              \
      DefaultDenseDatasetView<uint8_t>, LookupElement, kCompileTimeNumCenters, \
      IndexIterator, false>(                                                   \
      ConstSpan<LookupElement> lookup, size_t runtime_num_centers,             \
      const DefaultDenseDatasetView<uint8_t>* __restrict__ hashed_database,    \
      IndexIterator it);

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN_RESTRICTS( \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess)

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN(                          \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess) \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL0(                                     \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,            \
      SCANN_SINGLE_ARG(UnrestrictedIndexIterator<6, Postprocess>));        \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN_RESTRICTS(                      \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,            \
      SCANN_SINGLE_ARG(Postprocess));

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_POPULATE(                          \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess)     \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL0(                                         \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,                \
      SCANN_SINGLE_ARG(PopulateDistancesIterator<6, Postprocess>));            \
  extern_or_nothing template void                                              \
  GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCentersImpl<              \
      DefaultDenseDatasetView<uint8_t>, LookupElement, kCompileTimeNumCenters, \
      PopulateDistancesIterator<6, Postprocess>, true>(                        \
      ConstSpan<LookupElement> lookup, size_t runtime_num_centers,             \
      const DefaultDenseDatasetView<uint8_t>* __restrict__ hashed_database,    \
      PopulateDistancesIterator<6, Postprocess> it);

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL2_CROWDING( \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess)

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL2(                                   \
    extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess)     \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN(                                    \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,                \
      SCANN_SINGLE_ARG(AddPostprocessedValueToTopN<TopNeighbors<float>, float, \
                                                   Postprocess>));             \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN(                                    \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,                \
      SCANN_SINGLE_ARG(AddPostprocessedValueToTopN<TopNeighbors<int32_t>,      \
                                                   int32_t, Postprocess>));    \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_TOPN(                                    \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters,                \
      SCANN_SINGLE_ARG(AddPostprocessedValueToTopN<                            \
                       TopNeighbors<float>, float,                             \
                       ConvertToFloatAndPostprocess<Postprocess>>));           \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL2_CROWDING(                                \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess);  \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL1_POPULATE(                                \
      extern_or_nothing, LookupElement, kCompileTimeNumCenters, Postprocess);

#define SCANN_INSTANTIATE_AH_FUNCTION_IMPL3(extern_or_nothing, LookupElement,  \
                                            kCompileTimeNumCenters)            \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL2(extern_or_nothing, LookupElement,        \
                                      kCompileTimeNumCenters,                  \
                                      IdentityPostprocessFunctor);             \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL2(extern_or_nothing, LookupElement,        \
                                      kCompileTimeNumCenters, AddBiasFunctor); \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL2(extern_or_nothing, LookupElement,        \
                                      kCompileTimeNumCenters,                  \
                                      LimitedInnerFunctor);

#define SCANN_INSTANTIATE_AH_FUNCTION(extern_or_nothing)               \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL3(extern_or_nothing, float, 0);    \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL3(extern_or_nothing, uint16_t, 0); \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL3(extern_or_nothing, uint8_t, 0);  \
  SCANN_INSTANTIATE_AH_FUNCTION_IMPL3(extern_or_nothing, uint8_t, 256);

SCANN_INSTANTIATE_AH_FUNCTION(extern);

template <typename TopNsrc, typename TopNdst>
inline void DistanceTranslateWithMultiplier(const TopNsrc& top_n_src,
                                            const float multiplier,
                                            TopNdst* top_n_dst) {
  top_n_dst->resize(top_n_src.size());
  const float inv_mul = 1.0f / multiplier;
  auto dst_ptr = top_n_dst->begin();
  auto src_ptr = top_n_src.begin();
  auto src_size = top_n_src.size();
  for (size_t i = 0; i < src_size; ++i) {
    dst_ptr[i].first = src_ptr[i].first;
    dst_ptr[i].second = src_ptr[i].second * inv_mul;
  }
}

extern template class UnrestrictedIndexIterator<6, IdentityPostprocessFunctor>;
extern template class UnrestrictedIndexIterator<6, AddBiasFunctor>;
extern template class UnrestrictedIndexIterator<6, LimitedInnerFunctor>;
extern template class PopulateDistancesIterator<6, IdentityPostprocessFunctor>;
extern template class PopulateDistancesIterator<6, AddBiasFunctor>;
extern template class PopulateDistancesIterator<6, LimitedInnerFunctor>;

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
