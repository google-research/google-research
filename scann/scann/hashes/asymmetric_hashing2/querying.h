// Copyright 2024 The Google Research Authors.
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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_QUERYING_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_QUERYING_H_

#include <math.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/base/search_parameters.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/hashes/internal/asymmetric_hashing_lut16.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/hashes/internal/lut16_args.h"
#include "scann/hashes/internal/lut16_interface.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/restricts/restrict_allowlist.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {
namespace asymmetric_hashing2 {

struct LookupTable {
  bool empty() const {
    return float_lookup_table.empty() && int16_lookup_table.empty() &&
           int8_lookup_table.empty();
  }

  std::vector<float> float_lookup_table = {};
  std::vector<uint16_t> int16_lookup_table = {};
  std::vector<uint8_t> int8_lookup_table = {};

  float fixed_point_multiplier = NAN;

  bool can_use_int16_accumulator = false;

  absl::StatusOr<std::vector<uint8_t>> ToBytes() const;
  static absl::StatusOr<LookupTable> FromBytes(absl::Span<const uint8_t> bytes);
};

struct PackedDataset {
  std::vector<uint8_t> bit_packed_data = {};

  DatapointIndex num_datapoints = 0;

  uint32_t num_blocks = 0;
};

PackedDataset CreatePackedDataset(const DenseDataset<uint8_t>& hashed_database);

struct PackedDatasetView {
  ConstSpan<uint8_t> bit_packed_data = {};

  DatapointIndex num_datapoints = 0;

  DimensionIndex num_blocks = 0;
};

DenseDataset<uint8_t> UnpackDataset(const PackedDatasetView& packed);

PackedDatasetView CreatePackedDatasetView(const PackedDataset& packed_dataset);

template <typename PostprocessFunctor =
              asymmetric_hashing_internal::IdentityPostprocessFunctor,
          typename DatasetView = DefaultDenseDatasetView<uint8_t>>
struct QueryerOptions {
  std::shared_ptr<DatasetView> hashed_dataset;

  std::optional<PackedDatasetView> lut16_packed_dataset;

  PostprocessFunctor postprocessing_functor;
};

namespace ai = ::research_scann::asymmetric_hashing_internal;

template <typename T>
class AsymmetricQueryer {
 public:
  using IdentityPostprocessFunctor =
      asymmetric_hashing_internal::IdentityPostprocessFunctor;

  AsymmetricQueryer(shared_ptr<const ChunkingProjection<T>> projector,
                    shared_ptr<const DistanceMeasure> lookup_distance,
                    shared_ptr<const Model<T>> model);

  using FixedPointLUTConversionOptions =
      AsymmetricHasherConfig::FixedPointLUTConversionOptions;

  template <typename LookupElement>
  StatusOr<LookupTable> CreateLookupTable(
      const DatapointPtr<T>& query, const DistanceMeasure& lookup_distance,
      FixedPointLUTConversionOptions float_int_conversion_options =
          FixedPointLUTConversionOptions()) const;

  template <typename LookupElement>
  StatusOr<LookupTable> CreateLookupTable(
      const DatapointPtr<T>& query,
      FixedPointLUTConversionOptions float_int_conversion_options =
          FixedPointLUTConversionOptions()) const {
    DCHECK(lookup_distance_);
    return CreateLookupTable<LookupElement>(query, *lookup_distance_,
                                            float_int_conversion_options);
  }

  StatusOr<LookupTable> CreateLookupTable(
      const DatapointPtr<T>& query,
      AsymmetricHasherConfig::LookupType lookup_type,
      FixedPointLUTConversionOptions float_int_conversion_options =
          FixedPointLUTConversionOptions()) const;

  template <typename TopN, typename Functor = IdentityPostprocessFunctor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status FindApproximateNeighbors(
      const LookupTable& lookup_table, const SearchParameters& params,
      QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n);

  template <size_t kNumQueries, typename TopN, typename Functor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status FindApproximateNeighborsBatched(
      array<const LookupTable*, kNumQueries> lookup_tables,
      array<const SearchParameters*, kNumQueries> params,
      QueryerOptions<Functor, DatasetView> querying_options,
      array<TopN*, kNumQueries> top_ns);

  template <typename Functor = IdentityPostprocessFunctor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status PopulateDistances(
      const LookupTable& lookup_table,
      QueryerOptions<Functor, DatasetView> querying_options,
      MutableSpan<pair<DatapointIndex, float>> results);

  shared_ptr<const DistanceMeasure> lookup_distance() const {
    return lookup_distance_;
  }

  AsymmetricHasherConfig::QuantizationScheme quantization_scheme() const {
    return model_->quantization_scheme();
  }

  size_t num_clusters_per_block() const {
    return model_->num_clusters_per_block();
  }

  size_t num_blocks() const { return model_->centers().size(); }

  shared_ptr<const Model<T>> model() const { return model_; }

 private:
  template <typename LookupElement, typename TopN,
            typename Functor = IdentityPostprocessFunctor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status FindApproximateNeighborsNoLUT16(
      const LookupTable& lookup_table, const SearchParameters& params,
      QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n);

  template <typename LookupElement, typename MaxDist, typename TopN,
            typename Functor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status FindApproximateNeighborsNoLUT16Impl(
      const DatasetView* __restrict__ hashed_dataset,
      DimensionIndex num_clusters_per_block,
      ConstSpan<LookupElement> lookup_raw, MaxDist max_dist,
      const RestrictAllowlist* whitelist_or_null, Functor postprocess,
      TopN* top_n);

  template <typename TopN, typename Functor = IdentityPostprocessFunctor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status FindApproximateNeighborsForceLUT16(
      const LookupTable& lookup_table, const SearchParameters& params,
      QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n);

  template <typename LookupElement, typename Functor,
            typename DatasetView = DefaultDenseDatasetView<uint8_t>>
  static Status PopulateDistancesImpl(
      const LookupTable& lookup_table,
      QueryerOptions<Functor, DatasetView> querying_options,
      MutableSpan<pair<DatapointIndex, float>> results);

  shared_ptr<const ChunkingProjection<T>> projector_;
  shared_ptr<const DistanceMeasure> lookup_distance_;
  shared_ptr<const Model<T>> model_;
};

template <typename T>
inline ConstSpan<T> GetRawLookupTable(const LookupTable& lookup_table) {
  LOG(FATAL) << "INVALID TYPE";
}

template <>
inline ConstSpan<float> GetRawLookupTable<float>(
    const LookupTable& lookup_table) {
  return ConstSpan<float>(lookup_table.float_lookup_table);
}

template <>
inline ConstSpan<uint8_t> GetRawLookupTable<uint8_t>(
    const LookupTable& lookup_table) {
  return ConstSpan<uint8_t>(lookup_table.int8_lookup_table);
}

template <>
inline ConstSpan<uint16_t> GetRawLookupTable<uint16_t>(
    const LookupTable& lookup_table) {
  return ConstSpan<uint16_t>(lookup_table.int16_lookup_table);
}

template <typename T>
template <typename LookupElement>
StatusOr<LookupTable> AsymmetricQueryer<T>::CreateLookupTable(
    const DatapointPtr<T>& query, const DistanceMeasure& lookup_distance,
    AsymmetricHasherConfig::FixedPointLUTConversionOptions
        float_int_conversion_options) const {
  const DatapointPtr<T> query_no_bias = [&] {
    if (quantization_scheme() == AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
      return MakeDatapointPtr(query.indices(), query.values(),
                              query.nonzero_entries() - 1,
                              query.dimensionality() - 1);
    } else {
      return query;
    }
  }();
  SCANN_ASSIGN_OR_RETURN(
      auto raw_float_lookup,
      asymmetric_hashing_internal::CreateRawFloatLookupTable(
          query_no_bias, *projector_, lookup_distance, model_->centers(),
          model_->num_clusters_per_block()));

  LookupTable result;
  if (IsIntegerType<LookupElement>() &&
      (float_int_conversion_options.multiplier_quantile() > 1.0 ||
       float_int_conversion_options.multiplier_quantile() <= 0.0)) {
    return InvalidArgumentError(
        "FixedPointLUTConversionOptions::multiplier_quantile must be in (0.0, "
        "1.0].");
  }
  if (IsSame<LookupElement, float>()) {
    result.float_lookup_table = std::move(raw_float_lookup);
  } else if (IsSameAny<LookupElement, int16_t, uint16_t>()) {
    result.int16_lookup_table = ai::ConvertLookupToFixedPoint<uint16_t>(
        raw_float_lookup, float_int_conversion_options,
        &result.fixed_point_multiplier);
  } else {
    result.int8_lookup_table = ai::ConvertLookupToFixedPoint<uint8_t>(
        raw_float_lookup, float_int_conversion_options,
        &result.fixed_point_multiplier);
    result.can_use_int16_accumulator = ai::CanUseInt16Accumulator(
        result.int8_lookup_table,
        result.int8_lookup_table.size() / model_->num_clusters_per_block());
  }
  return std::move(result);
}

template <typename T>
template <typename TopN, typename Functor, typename DatasetView>
Status AsymmetricQueryer<T>::FindApproximateNeighbors(
    const LookupTable& lookup_table, const SearchParameters& params,
    QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n) {
  DCHECK(top_n);
  static_assert(std::is_same_v<float, decltype(top_n->approx_bottom().second)>,
                "The distance type for TopN must be float for "
                "AsymmetricQueryer::FindApproximateNeighbors.");
  if (!top_n->empty()) {
    return FailedPreconditionError(
        "TopN must be empty for "
        "AsymmetricQueryer::FindApproximateNeighbors.");
  }

  if (static_cast<int>(lookup_table.float_lookup_table.empty()) +
          static_cast<int>(lookup_table.int16_lookup_table.empty()) +
          static_cast<int>(lookup_table.int8_lookup_table.empty()) !=
      2) {
    return InvalidArgumentError(
        "Exactly one of float/int8/int16 lookup table must be populated.");
  }

  if (!querying_options.hashed_dataset &&
      !querying_options.lut16_packed_dataset.has_value()) {
    return InvalidArgumentError(
        "Either hashed_dataset or lut16_packed_dataset must be provided to "
        "AsymmetricQueryer::FindApproximateNeighbors.");
  }

  if ((querying_options.hashed_dataset &&
       querying_options.hashed_dataset->size() == 0) ||
      (querying_options.lut16_packed_dataset.has_value() &&
       querying_options.lut16_packed_dataset->num_blocks == 0)) {
    return OkStatus();
  }

  const bool can_use_lut16 =
      querying_options.lut16_packed_dataset.has_value() &&
      !lookup_table.int8_lookup_table.empty() &&
      lookup_table.int8_lookup_table.size() /
              querying_options.lut16_packed_dataset->num_blocks ==
          16;

  if (can_use_lut16) {
    return FindApproximateNeighborsForceLUT16<TopN, Functor>(
        lookup_table, params, querying_options, top_n);
  } else if (querying_options.hashed_dataset) {
    auto in_memory_ptr =
        (!lookup_table.float_lookup_table.empty())
            ? &FindApproximateNeighborsNoLUT16<float, TopN, Functor,
                                               DatasetView>
            : (!lookup_table.int8_lookup_table.empty())
                  ? &FindApproximateNeighborsNoLUT16<uint8_t, TopN, Functor,
                                                     DatasetView>
                  : &FindApproximateNeighborsNoLUT16<uint16_t, TopN, Functor,
                                                     DatasetView>;
    return (*in_memory_ptr)(lookup_table, params, querying_options, top_n);
  } else {
    return InvalidArgumentError(
        "LUT16 querying not possible.  Could not fall back to in-memory "
        "querying because no hashed_dataset provided.");
  }

  return OkStatus();
}

namespace asymmetric_hashing2_internal {

template <size_t kNumQueries>
Status FindApproxNeighborsFastTopNeighbors(
    array<const LookupTable*, kNumQueries> lookup_tables,
    array<const SearchParameters*, kNumQueries> params,
    const PackedDatasetView& packed_dataset,
    array<TopNeighbors<float>*, kNumQueries> top_ns) {
  array<FastTopNeighbors<int16_t>, kNumQueries> ftns;
  array<FastTopNeighbors<int16_t>*, kNumQueries> ftn_ptrs;
  array<const uint8_t*, kNumQueries> raw_luts;
  array<RestrictAllowlistConstView, kNumQueries> restricts;
  for (size_t batch_idx : Seq(kNumQueries)) {
    int32_t fixed_point_max_distance =
        ai::ComputePossiblyFixedPointMaxDistance<int8_t>(
            params[batch_idx]->pre_reordering_epsilon(),
            lookup_tables[batch_idx]->fixed_point_multiplier);

    fixed_point_max_distance =
        std::min<int32_t>(fixed_point_max_distance,
                          numeric_limits<int16_t>::max() - 1) +
        1;
    ftns[batch_idx] = FastTopNeighbors<int16_t>(top_ns[batch_idx]->limit(),
                                                fixed_point_max_distance);
    ftn_ptrs[batch_idx] = &ftns[batch_idx];
    raw_luts[batch_idx] = lookup_tables[batch_idx]->int8_lookup_table.data();
    if (params[batch_idx]->restricts_enabled()) {
      restricts[batch_idx] =
          RestrictAllowlistConstView(*params[batch_idx]->restrict_whitelist());
    } else {
      restricts[batch_idx] = RestrictAllowlistConstView();
    }
  }
  asymmetric_hashing_internal::LUT16ArgsTopN<int16_t> args;
  args.packed_dataset = packed_dataset.bit_packed_data.data();
  args.num_32dp_simd_iters = DivRoundUp(packed_dataset.num_datapoints, 32);
  args.num_blocks = packed_dataset.num_blocks;
  args.lookups = {raw_luts.data(), kNumQueries};
  args.first_dp_index = 0;
  args.num_datapoints = packed_dataset.num_datapoints;
  args.fast_topns = {ftn_ptrs.data(), kNumQueries};
  args.restrict_whitelists = restricts;
  asymmetric_hashing_internal::LUT16Interface::GetTopDistances(std::move(args));

  for (size_t batch_idx : Seq(kNumQueries)) {
    ConstSpan<DatapointIndex> ii;
    ConstSpan<int16_t> vv;
    std::tie(ii, vv) = ftns[batch_idx].FinishUnsorted();

    NNResultsVector v(ii.size());
    const float inv_fixed_point_multiplier =
        1.0f / lookup_tables[batch_idx]->fixed_point_multiplier;
    for (size_t j : Seq(ii.size())) {
      const float dist = vv[j] * inv_fixed_point_multiplier;
      v[j] = {ii[j], dist};
    }

    top_ns[batch_idx]->OverwriteContents(std::move(v),
                                         {numeric_limits<DatapointIndex>::max(),
                                          numeric_limits<float>::infinity()});
  }
  return OkStatus();
}

}  // namespace asymmetric_hashing2_internal

template <typename T>
template <size_t kNumQueries, typename TopN, typename Functor,
          typename DatasetView>
Status AsymmetricQueryer<T>::FindApproximateNeighborsBatched(
    array<const LookupTable*, kNumQueries> lookup_tables,
    array<const SearchParameters*, kNumQueries> params,
    QueryerOptions<Functor, DatasetView> querying_options,
    array<TopN*, kNumQueries> top_ns) {
  static_assert(kNumQueries <= 9,
                "Only batch sizes up to 9 are supported in "
                "FindApproximateNeighborsBatched.");
  static_assert(
      std::is_same_v<float, decltype(top_ns[0]->approx_bottom().second)>,
      "The distance type for TopN must be float for "
      "AsymmetricQueryer::FindApproximateNeighborsBatched.");
  for (TopN* top_n : top_ns) {
    DCHECK(top_n);
    if (!top_n->empty()) {
      return FailedPreconditionError(
          "TopNs must be empty for "
          "AsymmetricQueryer::FindApproximateNeighborsBatched.");
    }
  }

  if (!querying_options.hashed_dataset &&
      !querying_options.lut16_packed_dataset.has_value()) {
    return InvalidArgumentError(
        "Either hashed_dataset or lut16_packed_dataset must be provided to "
        "AsymmetricQueryer::FindApproximateNeighborsBatched.");
  }

  if ((querying_options.hashed_dataset &&
       querying_options.hashed_dataset->size() == 0) ||
      (querying_options.lut16_packed_dataset.has_value() &&
       querying_options.lut16_packed_dataset->num_blocks == 0)) {
    return OkStatus();
  }

  const bool can_use_lut16_for_all = [&] {
    if (!std::is_same_v<Functor, IdentityPostprocessFunctor>) return false;
    if (!querying_options.lut16_packed_dataset.has_value()) return false;
    for (const LookupTable* lt : lookup_tables) {
      if (lt->int8_lookup_table.empty()) return false;
      if (lt->int8_lookup_table.size() /
              querying_options.lut16_packed_dataset->num_blocks !=
          16) {
        return false;
      }
    }
    return true;
  }();

  if (!can_use_lut16_for_all) {
    for (size_t i = 0; i < kNumQueries; ++i) {
      SCANN_RETURN_IF_ERROR(FindApproximateNeighbors(
          *lookup_tables[i], *params[i], querying_options, top_ns[i]));
    }
    return OkStatus();
  }

  const bool can_use_int16_accumulator_for_all = [&] {
    for (const LookupTable* lt : lookup_tables) {
      if (!lt->can_use_int16_accumulator) return false;
    }
    return true;
  }();

  auto& packed_dataset = *querying_options.lut16_packed_dataset;
  std::array<ConstSpan<uint8_t>, kNumQueries> lookup_spans;
  std::array<int32_t, kNumQueries> max_dists;
  std::array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null;
  for (size_t i = 0; i < kNumQueries; ++i) {
    lookup_spans[i] = lookup_tables[i]->int8_lookup_table;
    restrict_whitelists_or_null[i] = params[i]->restrict_whitelist();
    max_dists[i] = ai::ComputePossiblyFixedPointMaxDistance<int8_t>(
        params[i]->pre_reordering_epsilon(),
        lookup_tables[i]->fixed_point_multiplier);
  }

  using RawTopN =
      decltype(top_ns[0]->template CloneWithAlternateDistanceType<int32_t>());
  array<RawTopN, kNumQueries> raw_top_ns_storage;
  array<RawTopN*, kNumQueries> raw_top_ns;
  for (size_t i = 0; i < kNumQueries; ++i) {
    raw_top_ns_storage[i] =
        top_ns[i]->template CloneWithAlternateDistanceType<int32_t>();
    raw_top_ns[i] = &raw_top_ns_storage[i];
  }
  if (can_use_int16_accumulator_for_all) {
    if constexpr (std::is_same_v<TopN, TopNeighbors<float>>) {
      auto& top_ns_casted =
          *reinterpret_cast<array<TopNeighbors<float>*, kNumQueries>*>(&top_ns);
      return asymmetric_hashing2_internal::FindApproxNeighborsFastTopNeighbors<
          kNumQueries>(lookup_tables, params, packed_dataset, top_ns_casted);
    } else {
      ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt16AccumulatorBatched2(
          lookup_spans, packed_dataset.num_datapoints,
          packed_dataset.bit_packed_data, restrict_whitelists_or_null,
          max_dists, querying_options.postprocessing_functor, raw_top_ns);
    }
  } else {
    ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt32AccumulatorBatched2(
        lookup_spans, packed_dataset.num_datapoints,
        packed_dataset.bit_packed_data, restrict_whitelists_or_null, max_dists,
        querying_options.postprocessing_functor, raw_top_ns);
  }
  for (size_t i = 0; i < kNumQueries; ++i) {
    const float inv_fixed_point_multiplier =
        1.0f / lookup_tables[i]->fixed_point_multiplier;
    top_ns[i]->OverwriteFromClone(raw_top_ns[i],
                                  [inv_fixed_point_multiplier](int32_t x) {
                                    return x * inv_fixed_point_multiplier;
                                  });
  }
  return OkStatus();
}

namespace asymmetric_hashing2_internal {

template <typename TopN>
void MoveOrOverwriteFromClone(TopN* dst, TopN* src,
                              float fixed_point_multiplier) {
  static_assert(std::is_same_v<float, decltype(src->approx_bottom().second)>,
                "The single-template parameter instantiation of "
                "MoveOrOverwriteFromClone should only be "
                "called with float distance.");
  *dst = std::move(*src);
}

template <typename TopN0, typename TopN1>
void MoveOrOverwriteFromClone(TopN0* dst, TopN1* src,
                              float fixed_point_multiplier) {
  static_assert(!std::is_same_v<float, decltype(src->approx_bottom().second)>,
                "The dual-template parameter instantiation of "
                "MoveOrOverwriteFromClone should only be "
                "called with non-float src distance.");
  const float inv_fixed_point_multiplier = 1.0f / fixed_point_multiplier;
  dst->OverwriteFromClone(src, [inv_fixed_point_multiplier](int32_t x) {
    return x * inv_fixed_point_multiplier;
  });
}

}  // namespace asymmetric_hashing2_internal

template <typename T>
template <typename LookupElement, typename TopN, typename Functor,
          typename DatasetView>
Status AsymmetricQueryer<T>::FindApproximateNeighborsNoLUT16(
    const LookupTable& lookup_table, const SearchParameters& params,
    QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n) {
  const DatasetView* __restrict__ hashed_dataset =
      querying_options.hashed_dataset.get();
  const ConstSpan<LookupElement> lookup_raw =
      GetRawLookupTable<LookupElement>(lookup_table);

  constexpr DimensionIndex kMaxInt8Blocks =
      numeric_limits<int32_t>::min() / numeric_limits<int8_t>::min();
  constexpr DimensionIndex kMaxInt16Blocks =
      numeric_limits<int32_t>::min() / numeric_limits<int16_t>::min();

  const size_t num_database_points = hashed_dataset->size();
  if (num_database_points == 0) {
    return OkStatus();
  }

  const size_t num_hashes = hashed_dataset->dimensionality();

  if (std::is_same_v<LookupElement, int8_t> && num_hashes > kMaxInt8Blocks) {
    return InvalidArgumentError(absl::StrCat(
        "Number of AH blocks (", num_hashes,
        ") may produce overflow.  (Max blocks for int8 lookup table = ",
        kMaxInt8Blocks, ")."));
  } else if (std::is_same_v<LookupElement, int16_t> &&
             num_hashes > kMaxInt16Blocks) {
    return InvalidArgumentError(absl::StrCat(
        "Number of AH blocks (", num_hashes,
        ") may produce overflow.  (Max blocks for int16 lookup table = ",
        kMaxInt16Blocks, ")."));
  }

  const int32_t num_clusters_per_block = lookup_raw.size() / num_hashes;
  if (num_hashes * num_clusters_per_block != lookup_raw.size()) {
    return InvalidArgumentError(
        absl::StrCat("Mismatch between number of hashes in database (",
                     num_hashes, ") and number implied by lookup table size (",
                     lookup_raw.size() / num_clusters_per_block, "."));
  }

  const RestrictAllowlist* whitelist_or_null = params.restrict_whitelist();
  if (std::is_same_v<Functor, IdentityPostprocessFunctor> ||
      std::is_same_v<LookupElement, float>) {
    auto possibly_fixed_point_max_distance =
        ai::ComputePossiblyFixedPointMaxDistance<LookupElement>(
            params.pre_reordering_epsilon(),
            lookup_table.fixed_point_multiplier);
    using PossiblyFixedDist = decltype(possibly_fixed_point_max_distance);
    using PossiblyFixedTopN = decltype(
        top_n->template CloneWithAlternateDistanceType<PossiblyFixedDist>());
    PossiblyFixedTopN raw_top_items =
        top_n->template CloneWithAlternateDistanceType<PossiblyFixedDist>();
    SCANN_RETURN_IF_ERROR(
        AsymmetricQueryer<T>::FindApproximateNeighborsNoLUT16Impl(
            hashed_dataset, num_clusters_per_block, lookup_raw,
            possibly_fixed_point_max_distance, whitelist_or_null,
            querying_options.postprocessing_functor, &raw_top_items));
    asymmetric_hashing2_internal::MoveOrOverwriteFromClone(
        top_n, &raw_top_items, lookup_table.fixed_point_multiplier);
  } else {
    asymmetric_hashing_internal::ConvertToFloatAndPostprocess<Functor>
        postprocess_with_float_conversion(
            querying_options.postprocessing_functor,
            1.0f / lookup_table.fixed_point_multiplier);
    SCANN_RETURN_IF_ERROR(
        AsymmetricQueryer<T>::FindApproximateNeighborsNoLUT16Impl(
            hashed_dataset, num_clusters_per_block, lookup_raw,
            params.pre_reordering_epsilon(), whitelist_or_null,
            postprocess_with_float_conversion, top_n));
  }
  return OkStatus();
}

template <typename T>
template <typename LookupElement, typename MaxDist, typename TopN,
          typename Functor, typename DatasetView>
Status AsymmetricQueryer<T>::FindApproximateNeighborsNoLUT16Impl(
    const DatasetView* __restrict__ hashed_dataset,
    DimensionIndex num_clusters_per_block, ConstSpan<LookupElement> lookup_raw,
    MaxDist max_dist, const RestrictAllowlist* whitelist_or_null,
    Functor postprocess, TopN* top_n) {
  using TopNFunctor = ai::AddPostprocessedValueToTopN<TopN, MaxDist, Functor>;
  TopNFunctor top_n_functor(top_n, max_dist, postprocess);
  if (!whitelist_or_null) {
    ai::UnrestrictedIndexIterator<6, TopNFunctor> it(hashed_dataset->size(),
                                                     top_n_functor);
    auto unrestricted_ptr =
        &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
            DatasetView, LookupElement, 0, decltype(it)>;
    if (num_clusters_per_block == 256) {
      unrestricted_ptr =
          &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
              DatasetView, LookupElement, 256, decltype(it)>;
    } else if (num_clusters_per_block == 128) {
      unrestricted_ptr =
          &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
              DatasetView, LookupElement, 128, decltype(it)>;
    } else if (num_clusters_per_block == 16) {
      unrestricted_ptr =
          &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
              DatasetView, LookupElement, 16, decltype(it)>;
    }

    (*unrestricted_ptr)(lookup_raw, num_clusters_per_block, hashed_dataset, it);
  } else {
    return UnimplementedError("Restricts aren't supported.");
  }
  return OkStatus();
}

template <typename T>
template <typename TopN, typename Functor, typename DatasetView>
Status AsymmetricQueryer<T>::FindApproximateNeighborsForceLUT16(
    const LookupTable& lookup_table, const SearchParameters& params,
    QueryerOptions<Functor, DatasetView> querying_options, TopN* top_n) {
  DCHECK(!lookup_table.int8_lookup_table.empty());
  DCHECK(querying_options.lut16_packed_dataset.has_value());
  auto& packed_dataset = *querying_options.lut16_packed_dataset;

  if constexpr (std::is_same_v<Functor, IdentityPostprocessFunctor>) {
    int32_t fixed_point_max_distance =
        ai::ComputePossiblyFixedPointMaxDistance<int8_t>(
            params.pre_reordering_epsilon(),
            lookup_table.fixed_point_multiplier);
    const float inv_fixed_point_multiplier =
        1.0f / lookup_table.fixed_point_multiplier;
    if (std::is_same_v<TopN, TopNeighbors<float>> &&
        lookup_table.can_use_int16_accumulator) {
      if (fixed_point_max_distance < numeric_limits<int16_t>::min()) {
        return OkStatus();
      }

      return asymmetric_hashing2_internal::FindApproxNeighborsFastTopNeighbors<
          1>({&lookup_table}, {&params}, packed_dataset,
             {reinterpret_cast<TopNeighbors<float>*>(top_n)});
    }

    using FixedTopN =
        decltype(top_n->template CloneWithAlternateDistanceType<int32_t>());
    FixedTopN raw_top_items =
        top_n->template CloneWithAlternateDistanceType<int32_t>();
    auto lut16_function =
        (lookup_table.can_use_int16_accumulator)
            ? &ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt16Accumulator2<
                  FixedTopN, int32_t, Functor>
            : &ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt32Accumulator2<
                  FixedTopN, int32_t, Functor>;
    (*lut16_function)(lookup_table.int8_lookup_table,
                      packed_dataset.num_datapoints,
                      packed_dataset.bit_packed_data,
                      params.restrict_whitelist(), fixed_point_max_distance,
                      querying_options.postprocessing_functor, &raw_top_items);
    top_n->OverwriteFromClone(&raw_top_items,
                              [inv_fixed_point_multiplier](int32_t x) {
                                return x * inv_fixed_point_multiplier;
                              });
  } else {
    using FloatPostprocessFunctor =
        asymmetric_hashing_internal::ConvertToFloatAndPostprocess<Functor>;
    FloatPostprocessFunctor postprocess_with_float_conversion(
        querying_options.postprocessing_functor,
        1.0f / lookup_table.fixed_point_multiplier);
    auto lut16_function =
        &ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt32Accumulator2<
            TopN, float, FloatPostprocessFunctor>;
    if (lookup_table.can_use_int16_accumulator) {
      lut16_function =
          &ai::GetNeighborsViaAsymmetricDistanceLUT16WithInt16Accumulator2<
              TopN, float, FloatPostprocessFunctor>;
    }
    (*lut16_function)(
        lookup_table.int8_lookup_table, packed_dataset.num_datapoints,
        packed_dataset.bit_packed_data, params.restrict_whitelist(),
        params.pre_reordering_epsilon(), postprocess_with_float_conversion,
        top_n);
  }
  return OkStatus();
}

template <typename T>
template <typename Functor, typename DatasetView>
Status AsymmetricQueryer<T>::PopulateDistances(
    const LookupTable& lookup_table,
    QueryerOptions<Functor, DatasetView> querying_options,
    MutableSpan<pair<DatapointIndex, float>> results) {
  if (static_cast<int>(lookup_table.float_lookup_table.empty()) +
          static_cast<int>(lookup_table.int16_lookup_table.empty()) +
          static_cast<int>(lookup_table.int8_lookup_table.empty()) !=
      2) {
    return InvalidArgumentError(
        "Exactly one of float/int8/int16 lookup table must be populated.");
  }

  auto impl_ptr =
      (!lookup_table.float_lookup_table.empty())
          ? &PopulateDistancesImpl<float, Functor, DatasetView>
          : (!lookup_table.int8_lookup_table.empty())
                ? &PopulateDistancesImpl<uint8_t, Functor, DatasetView>
                : &PopulateDistancesImpl<uint16_t, Functor, DatasetView>;
  return (*impl_ptr)(lookup_table, querying_options, results);
}

template <typename T>
template <typename LookupElement, typename Functor, typename DatasetView>
Status AsymmetricQueryer<T>::PopulateDistancesImpl(
    const LookupTable& lookup_table,
    QueryerOptions<Functor, DatasetView> querying_options,
    MutableSpan<pair<DatapointIndex, float>> results) {
  const ConstSpan<LookupElement> lookup_raw =
      GetRawLookupTable<LookupElement>(lookup_table);
  const DatasetView* __restrict__ hashed_dataset =
      querying_options.hashed_dataset.get();

  constexpr DimensionIndex kMaxInt8Blocks =
      numeric_limits<int32_t>::min() / numeric_limits<int8_t>::min();
  constexpr DimensionIndex kMaxInt16Blocks =
      numeric_limits<int32_t>::min() / numeric_limits<int16_t>::min();

  const size_t num_database_points = hashed_dataset->size();
  if (num_database_points == 0) {
    DCHECK(results.empty());
    return OkStatus();
  }

  const size_t num_hashes = hashed_dataset->dimensionality();

  if (IsSame<LookupElement, uint8_t>() && num_hashes > kMaxInt8Blocks) {
    return InvalidArgumentError(absl::StrCat(
        "Number of AH blocks (", num_hashes,
        ") may produce overflow.  (Max blocks for int8 lookup table = ",
        kMaxInt8Blocks, ")."));
  } else if (IsSame<LookupElement, uint16_t>() &&
             num_hashes > kMaxInt16Blocks) {
    return InvalidArgumentError(absl::StrCat(
        "Number of AH blocks (", num_hashes,
        ") may produce overflow.  (Max blocks for int16 lookup table = ",
        kMaxInt16Blocks, ")."));
  }

  const int32_t num_clusters_per_block = lookup_raw.size() / num_hashes;
  if (num_hashes * num_clusters_per_block != lookup_raw.size()) {
    return InvalidArgumentError(
        absl::StrCat("Mismatch between number of hashes in database (",
                     num_hashes, ") and number implied by lookup table size (",
                     lookup_raw.size() / num_clusters_per_block, "."));
  }

  ai::PopulateDistancesIterator<6, Functor> it(
      results, querying_options.postprocessing_functor);
  auto fp = &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
      DatasetView, LookupElement, 0, decltype(it)>;
  if (num_clusters_per_block == 256) {
    fp = &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
        DatasetView, LookupElement, 256, decltype(it)>;
  } else if (num_clusters_per_block == 128) {
    fp = &ai::GetNeighborsViaAsymmetricDistanceWithCompileTimeNumCenters<
        DatasetView, LookupElement, 128, decltype(it)>;
  }

  (*fp)(lookup_raw, num_clusters_per_block, hashed_dataset, it);
  if (!IsFloatingType<LookupElement>() && !results.empty()) {
    const float inv_mul = 1.0 / lookup_table.fixed_point_multiplier;
    for (auto& elem : results) {
      elem.second *= inv_mul;
    }
  }

  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, AsymmetricQueryer);

}  // namespace asymmetric_hashing2
}  // namespace research_scann

#endif
