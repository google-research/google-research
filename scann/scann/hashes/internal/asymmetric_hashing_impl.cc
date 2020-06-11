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

#include "scann/hashes/internal/asymmetric_hashing_impl.h"

#include <numeric>

#include "absl/random/distributions.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/projection/chunking_projection.h"
#include "scann/utils/common.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/noise_shaping_utils.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {

class ParallelPerpendicularDistance : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(NOT_SPECIALLY_OPTIMIZED);

  void set_parallel_cost_multiplier(double x) { parallel_cost_multiplier_ = x; }

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(
      const DatapointPtr<T>& x_dptr, const DatapointPtr<T>& y_dptr) const {
    DCHECK(x_dptr.IsDense());
    DCHECK(y_dptr.IsDense());
    auto x = x_dptr.values_slice();
    auto y = y_dptr.values_slice();
    DCHECK_EQ(x.size(), y.size());
    double parallel_error = 0.0;
    double residual_squared_norm = 0.0;
    for (size_t i : IndicesOf(x)) {
      const double residual_coord =
          static_cast<double>(y[i]) - static_cast<double>(x[i]);
      parallel_error += residual_coord * x[i];
      residual_squared_norm += Square(residual_coord);
    }
    parallel_error = Square(parallel_error);
    const double perpendicular_error = residual_squared_norm - parallel_error;
    return parallel_cost_multiplier_ * parallel_error + perpendicular_error;
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(
      const DatapointPtr<T>& x_dptr, const DatapointPtr<T>& y_dptr) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(
      const DatapointPtr<T>& x_dptr, const DatapointPtr<T>& y_dptr) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
  }

  double parallel_cost_multiplier_ = 1.0;
};

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(ParallelPerpendicularDistance,
                                              numeric_limits<size_t>::max());

string_view ParallelPerpendicularDistance::name() const {
  return "ParallelPerpendicularDistance";
}

namespace {

double ComputeNormBiasCorrection(const DenseDataset<double>& db,
                                 DatapointPtr<double> center,
                                 ConstSpan<DatapointIndex> cluster_members) {
  double mean_norm = 0.0;
  for (DatapointIndex idx : cluster_members) {
    mean_norm += std::sqrt(SquaredL2Norm(db[idx]));
  }
  mean_norm /= cluster_members.size();
  const double center_norm = std::sqrt(SquaredL2Norm(center));
  return (center_norm == 0.0) ? 1.0 : (mean_norm / center_norm);
}

}  // namespace

template <typename T>
StatusOr<vector<DenseDataset<double>>> AhImpl<T>::TrainAsymmetricHashing(
    const TypedDataset<T>& dataset, const TrainingOptionsT& opts,
    shared_ptr<thread::ThreadPool> pool) {
  if (dataset.empty()) {
    return InvalidArgumentError("Cannot train AH on an empty dataset.");
  }

  ChunkedDatapoint<double> chunked_vec;

  if (opts.preprocessing_function()) {
    TF_ASSIGN_OR_RETURN(Datapoint<T> preprocessed,
                        opts.preprocessing_function()(dataset[0]));
    SCANN_RETURN_IF_ERROR(
        opts.projector()->ProjectInput(preprocessed.ToPtr(), &chunked_vec));
  } else {
    SCANN_RETURN_IF_ERROR(
        opts.projector()->ProjectInput(dataset[0], &chunked_vec));
  }
  int32_t num_blocks = chunked_vec.size();
  vector<DenseDataset<double>> chunked_dataset(num_blocks);

  const float sampling_fraction =
      opts.config().has_expected_sample_size()
          ? std::min(1.0,
                     static_cast<double>(opts.config().expected_sample_size()) /
                         static_cast<double>(dataset.size()))
          : opts.config().sampling_fraction();

  if (sampling_fraction == 1.0) {
    for (int32_t i = 0; i < num_blocks; ++i) {
      DenseDataset<double>& ds = chunked_dataset[i];
      ds.set_dimensionality(chunked_vec[i].dimensionality());
      ds.Reserve(dataset.size());
    }
  }

  MTRandom rng(kDeterministicSeed * (opts.config().sampling_seed() + 1));
  vector<DatapointIndex> sample;
  for (DatapointIndex i = 0; i < dataset.size(); ++i) {
    if (absl::Uniform<double>(rng, 0, 1.0) < sampling_fraction) {
      sample.push_back(i);
    }
  }

  if (sample.size() > opts.config().max_sample_size()) {
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(opts.config().max_sample_size());
    std::sort(sample.begin(), sample.end());
  }

  if (sample.size() < opts.config().num_clusters_per_block()) {
    return InvalidArgumentError(absl::StrCat(
        "Number of clusters per block (",
        opts.config().num_clusters_per_block(),
        ") is greater than asymmetric hashing training data size (",
        sample.size(), ")."));
  }

  auto append_chunked_blocks = [&] {
    for (size_t j = 0; j < num_blocks; ++j) {
      chunked_dataset[j].AppendOrDie(chunked_vec[j], "");
    }
  };

  if (opts.preprocessing_function()) {
    for (DatapointIndex i : sample) {
      TF_ASSIGN_OR_RETURN(Datapoint<T> preprocessed,
                          opts.preprocessing_function()(dataset[i]));
      SCANN_RETURN_IF_ERROR(
          opts.projector()->ProjectInput(preprocessed.ToPtr(), &chunked_vec));
      append_chunked_blocks();
    }
  } else {
    for (DatapointIndex i : sample) {
      SCANN_RETURN_IF_ERROR(
          opts.projector()->ProjectInput(dataset[i], &chunked_vec));
      append_chunked_blocks();
    }
  }

  auto quantization_distance = opts.quantization_distance();
  GmmUtils::Options gmm_opts;
  gmm_opts.seed = opts.config().clustering_seed();
  gmm_opts.max_iterations = opts.config().max_clustering_iterations();
  gmm_opts.epsilon = opts.config().clustering_convergence_tolerance();
  gmm_opts.parallelization_pool = std::move(pool);
  if (!std::isnan(opts.config().noise_shaping_threshold()) &&
      opts.config().use_noise_shaped_training()) {
    gmm_opts.parallel_cost_multiplier = ComputeParallelCostMultiplier(
        opts.config().noise_shaping_threshold(), 1.0, dataset.dimensionality());
    auto d = make_shared<ParallelPerpendicularDistance>();
    d->set_parallel_cost_multiplier(gmm_opts.parallel_cost_multiplier);
    quantization_distance = d;
  }
  GmmUtils gmm(quantization_distance, gmm_opts);

  vector<DenseDataset<double>> all_centers(num_blocks);
  for (size_t i : Seq(num_blocks)) {
    DenseDataset<double> centers;
    vector<vector<DatapointIndex>> subpartitions;
    SCANN_RETURN_IF_ERROR(gmm.GenericKmeans(
        chunked_dataset[i], opts.config().num_clusters_per_block(), &centers,
        &subpartitions));

    if (opts.config().use_norm_biasing_correction()) {
      for (size_t center_idx : IndicesOf(centers)) {
        const double norm_bias_correction = ComputeNormBiasCorrection(
            chunked_dataset[i], centers[center_idx], subpartitions[center_idx]);
        for (double& d : centers.mutable_data(center_idx)) {
          d *= norm_bias_correction;
        }
      }
    }

    chunked_dataset[i].clear();
    chunked_dataset[i].ShrinkToFit();

    vector<uint32_t> centers_permutation(centers.size());
    std::iota(centers_permutation.begin(), centers_permutation.end(), 0U);
    std::sort(centers_permutation.begin(), centers_permutation.end(),
              [&subpartitions](uint32_t a, uint32_t b) {
                return subpartitions[a].size() > subpartitions[b].size();
              });

    constexpr size_t kAssumedCacheLineSize = 64;
    constexpr size_t kFloatsPerCacheLine =
        kAssumedCacheLineSize / sizeof(float);
    const unsigned long long_t cache_lines_per_row =
        std::max(static_cast<size_t>(1), centers.size() / kFloatsPerCacheLine);
    const size_t num_rotate =
        ((i / 2) % cache_lines_per_row) * kFloatsPerCacheLine;
    std::rotate(centers_permutation.begin(),
                centers_permutation.begin() + num_rotate,
                centers_permutation.end());

    if (i & 1) {
      std::reverse(centers_permutation.begin(), centers_permutation.end());
    }

    for (uint32_t j : centers_permutation) {
      all_centers[i].AppendOrDie(centers[j], "");
    }
  }

  return std::move(all_centers);
}

template <typename T>
Status AhImpl<T>::IndexDatapoint(const DatapointPtr<T>& input,
                                 const ChunkingProjection<T>& projection,
                                 const DistanceMeasure& quantization_distance,
                                 ConstSpan<DenseDataset<FloatT>> centers,
                                 MutableSpan<uint8_t> result) {
  DCHECK(!centers.empty());
  ChunkedDatapoint<FloatT> projected;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(input, &projected));

  DCHECK_LE(projected.size(), result.size());

  vector<float> distances(centers[0].size());
  DCHECK_GE(distances.size(), 1);
  DCHECK_LE(distances.size(), 256);
  for (size_t i = 0; i < projected.size(); ++i) {
    DCHECK_EQ(centers[0].size(), centers[i].size());
    size_t closest = 0;
    const DatapointPtr<FloatT> projected_ptr = projected[i];
    const DenseDataset<FloatT>& cur_centers = centers[i];

    const size_t centers_size = cur_centers.size();

    if (projected_ptr.IsSparse()) {
      double closest_distance = numeric_limits<double>::infinity();
      for (size_t j = 0; j < centers_size; ++j) {
        const double distance = quantization_distance.GetDistanceHybrid(
            projected_ptr, cur_centers[j]);
        if (ABSL_PREDICT_FALSE(distance < closest_distance)) {
          closest_distance = distance;
          closest = j;
        }
      }
    } else {
      DCHECK_EQ(distances.size(), cur_centers.size());
      DenseDistanceOneToMany(quantization_distance, projected_ptr, cur_centers,
                             MutableSpan<float>(distances));
      auto min_it = std::min_element(distances.begin(), distances.end());
      closest = min_it - distances.begin();
    }
    result[i] = closest;
  }

  return OkStatus();
}

template <typename T>
Status AhImpl<T>::IndexDatapoint(const DatapointPtr<T>& input,
                                 const ChunkingProjection<T>& projection,
                                 const DistanceMeasure& quantization_distance,
                                 ConstSpan<DenseDataset<FloatT>> centers,
                                 Datapoint<uint8_t>* result) {
  DatapointIndex result_size = centers.size();
  DCHECK_EQ(result_size, projection.num_blocks());
  result->clear();
  result->mutable_values()->resize(result_size, 0);

  return AhImpl<T>::IndexDatapoint(input, projection, quantization_distance,
                                   centers, result->mutable_values_slice());
}

namespace {

template <typename T>
T Square(T x) {
  return x * x;
}

double ComputeParallelCostMultiplier(double t, double squared_l2_norm,
                                     DimensionIndex dims) {
  const double parallel_cost = Square(t) / squared_l2_norm;
  const double perpendicular_cost =
      (1.0 - Square(t) / squared_l2_norm) / (dims - 1.0);
  return parallel_cost / perpendicular_cost;
}

struct SubspaceResidualStats {
  double residual_norm = 0.0;

  double parallel_residual_component = 0.0;
};

template <typename T>
SubspaceResidualStats ComputeResidualStatsForCluster(
    ConstSpan<T> maybe_residual_dptr, ConstSpan<T> original_dptr,
    double inv_norm, ConstSpan<FloatingTypeFor<T>> quantized) {
  DCHECK_EQ(maybe_residual_dptr.size(), quantized.size());
  const size_t dims = maybe_residual_dptr.size();
  SubspaceResidualStats result;
  for (size_t i : Seq(dims)) {
    const double residual_coordinate =
        static_cast<double>(maybe_residual_dptr[i]) -
        static_cast<double>(quantized[i]);
    result.residual_norm += Square(residual_coordinate);
    result.parallel_residual_component +=
        residual_coordinate * original_dptr[i] * inv_norm;
  }
  return result;
}

template <typename T>
StatusOr<vector<std::vector<SubspaceResidualStats>>> ComputeResidualStats(
    DatapointPtr<T> maybe_residual_dptr, DatapointPtr<T> original_dptr,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
    const ChunkingProjection<T>& projection) {
  const size_t num_subspaces = centers.size();
  DCHECK_GE(num_subspaces, 1);
  vector<std::vector<SubspaceResidualStats>> result(num_subspaces);
  vector<std::vector<SubspaceResidualStats>> residual_stats(num_subspaces);
  const size_t num_clusters_per_block = centers[0].size();

  using FloatT = FloatingTypeFor<T>;
  ChunkedDatapoint<FloatT> maybe_residual_dptr_chunked;
  ChunkedDatapoint<FloatT> original_dptr_chunked;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(maybe_residual_dptr,
                                                &maybe_residual_dptr_chunked));
  SCANN_RETURN_IF_ERROR(
      projection.ProjectInput(original_dptr, &original_dptr_chunked));
  SCANN_RET_CHECK_EQ(maybe_residual_dptr_chunked.size(), num_subspaces);
  SCANN_RET_CHECK_EQ(original_dptr_chunked.size(), num_subspaces);
  double chunked_norm = 0.0;
  for (size_t subspace_idx : Seq(num_subspaces)) {
    for (FloatT x : original_dptr_chunked[subspace_idx].values_slice()) {
      chunked_norm += Square<double>(x);
    }
  }
  chunked_norm = std::sqrt(chunked_norm);
  double inverse_chunked_norm = 1.0 / chunked_norm;

  for (size_t subspace_idx : Seq(num_subspaces)) {
    auto& cur_subspace_residual_stats = residual_stats[subspace_idx];
    cur_subspace_residual_stats.resize(num_clusters_per_block);
    const DenseDataset<FloatingTypeFor<T>>& cur_subspace_centers =
        centers[subspace_idx];
    for (size_t cluster_idx : Seq(num_clusters_per_block)) {
      ConstSpan<FloatingTypeFor<T>> center =
          cur_subspace_centers[cluster_idx].values_slice();
      ConstSpan<FloatT> maybe_residual_dptr_span =
          maybe_residual_dptr_chunked[subspace_idx].values_slice();
      ConstSpan<FloatT> original_dptr_span =
          original_dptr_chunked[subspace_idx].values_slice();
      cur_subspace_residual_stats[cluster_idx] = ComputeResidualStatsForCluster(
          maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
          center);
    }
  }
  return residual_stats;
}

void InitializeToMinResidualNorm(
    ConstSpan<std::vector<SubspaceResidualStats>> residual_stats,
    MutableSpan<uint8_t> result) {
  DCHECK_EQ(result.size(), residual_stats.size());
  for (size_t subspace_idx : IndicesOf(residual_stats)) {
    auto it = std::min_element(
        residual_stats[subspace_idx].begin(),
        residual_stats[subspace_idx].end(),
        [](const SubspaceResidualStats& a, const SubspaceResidualStats& b) {
          return a.residual_norm < b.residual_norm;
        });
    result[subspace_idx] = it - residual_stats[subspace_idx].begin();
  }
}

double ComputeParallelResidualComponent(
    ConstSpan<uint8_t> quantized,
    ConstSpan<std::vector<SubspaceResidualStats>> residual_stats) {
  double result = 0.0;
  for (size_t subspace_idx : IndicesOf(quantized)) {
    const uint8_t cluster_idx = quantized[subspace_idx];
    result +=
        residual_stats[subspace_idx][cluster_idx].parallel_residual_component;
  }
  return result;
}

struct CoordinateDescentResult {
  uint8_t new_center_idx = 0;
  double cost_delta = 0.0;
  double new_parallel_residual_component = 0.0;
};

CoordinateDescentResult OptimizeSingleSubspace(
    ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats,
    const uint8_t cur_center_idx, const double parallel_residual_component,
    const double parallel_cost_multiplier) {
  CoordinateDescentResult result;
  result.new_center_idx = cur_center_idx;
  result.new_parallel_residual_component = parallel_residual_component;
  const double old_subspace_residual_norm =
      cur_subspace_residual_stats[cur_center_idx].residual_norm;
  const double old_subspace_parallel_component =
      cur_subspace_residual_stats[cur_center_idx].parallel_residual_component;
  for (size_t new_center_idx : IndicesOf(cur_subspace_residual_stats)) {
    if (new_center_idx == cur_center_idx) continue;
    const SubspaceResidualStats& rs =
        cur_subspace_residual_stats[new_center_idx];
    const double new_parallel_residual_component =
        parallel_residual_component - old_subspace_parallel_component +
        rs.parallel_residual_component;
    const double parallel_norm_delta = Square(new_parallel_residual_component) -
                                       Square(parallel_residual_component);
    if (parallel_norm_delta > 0.0) continue;
    const double residual_norm_delta =
        rs.residual_norm - old_subspace_residual_norm;
    const double perpendicular_norm_delta =
        residual_norm_delta - parallel_norm_delta;
    const double cost_delta = parallel_cost_multiplier * parallel_norm_delta +
                              perpendicular_norm_delta;
    if (cost_delta < result.cost_delta) {
      result.new_center_idx = new_center_idx;
      result.cost_delta = cost_delta;
      result.new_parallel_residual_component = new_parallel_residual_component;
    }
  }
  return result;
}

template <typename T>
Status CoordinateDescentAHQuantize(
    DatapointPtr<T> maybe_residual_dptr, DatapointPtr<T> original_dptr,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
    const ChunkingProjection<T>& projection, double threshold,
    MutableSpan<uint8_t> result, int* num_changes = nullptr,
    double* residual_ptr = nullptr, double* parallel_residual_ptr = nullptr) {
  SCANN_RET_CHECK_EQ(result.size(), centers.size());
  SCANN_RET_CHECK_EQ(maybe_residual_dptr.dimensionality(),
                     original_dptr.dimensionality());
  TF_ASSIGN_OR_RETURN(auto residual_stats,
                      ComputeResidualStats(maybe_residual_dptr, original_dptr,
                                           centers, projection));

  const double parallel_cost_multiplier = ComputeParallelCostMultiplier(
      threshold, SquaredL2Norm(original_dptr), original_dptr.dimensionality());
  InitializeToMinResidualNorm(residual_stats, result);
  double parallel_residual_component =
      ComputeParallelResidualComponent(result, residual_stats);

  vector<uint16_t> subspace_idxs(result.size());
  std::iota(subspace_idxs.begin(), subspace_idxs.end(), 0U);
  vector<double> subspace_residual_norms(result.size());
  for (size_t subspace_idx : IndicesOf(result)) {
    const uint8_t cluster_idx = result[subspace_idx];
    subspace_residual_norms[subspace_idx] =
        residual_stats[subspace_idx][cluster_idx].residual_norm;
  }
  std::vector<uint8_t> result_sorted(result.begin(), result.end());
  ZipSortBranchOptimized(
      std::greater<double>(), subspace_residual_norms.begin(),
      subspace_residual_norms.end(), result_sorted.begin(), result_sorted.end(),
      subspace_idxs.begin(), subspace_idxs.end());

  enum { kMaxRounds = 10 };
  bool cur_round_changes = true;
  if (num_changes) *num_changes = 0;
  for (int round = 0; cur_round_changes && round < kMaxRounds; ++round) {
    cur_round_changes = false;
    for (size_t i : IndicesOf(subspace_idxs)) {
      const size_t subspace_idx = subspace_idxs[i];
      ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats =
          residual_stats[subspace_idx];
      const uint8_t cur_center_idx = result_sorted[i];
      auto subspace_result = OptimizeSingleSubspace(
          cur_subspace_residual_stats, cur_center_idx,
          parallel_residual_component, parallel_cost_multiplier);
      if (subspace_result.new_center_idx != cur_center_idx) {
        if (num_changes) ++*num_changes;
        parallel_residual_component =
            subspace_result.new_parallel_residual_component;
        result_sorted[i] = subspace_result.new_center_idx;
        cur_round_changes = true;
      }
    }
  }

  double final_residual_norm = 0.0;
  for (size_t i : IndicesOf(result_sorted)) {
    const size_t subspace_idx = subspace_idxs[i];
    const uint8_t center_idx = result_sorted[i];
    result[subspace_idx] = center_idx;
    final_residual_norm +=
        residual_stats[subspace_idx][center_idx].residual_norm;
  }
  if (residual_ptr) *residual_ptr = final_residual_norm;
  if (parallel_residual_ptr) {
    *parallel_residual_ptr = Square(parallel_residual_component);
  }
  return OkStatus();
}

}  // namespace

template <typename T>
Status AhImpl<T>::IndexDatapointNoiseShaped(
    const DatapointPtr<T>& maybe_residual_dptr,
    const DatapointPtr<T>& original_dptr,
    const ChunkingProjection<T>& projection,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers, double threshold,
    MutableSpan<uint8_t> result) {
  return CoordinateDescentAHQuantize<T>(maybe_residual_dptr, original_dptr,
                                        centers, projection, threshold, result);
}

template <typename T>
StatusOr<vector<float>> AhImpl<T>::CreateRawFloatLookupTable(
    const DatapointPtr<T>& query, const ChunkingProjection<T>& projection,
    const DistanceMeasure& lookup_distance,
    ConstSpan<DenseDataset<FloatT>> centers, int32_t num_clusters_per_block) {
  ChunkedDatapoint<FloatT> projected;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(query, &projected));
  SCANN_RET_CHECK_EQ(centers.size(), projected.size());

  vector<float> result(num_clusters_per_block * projected.size());
  float* result_row_start = result.data();

  for (size_t i = 0; i < centers.size();
       ++i, result_row_start += num_clusters_per_block) {
    const DatapointPtr<FloatT> projected_ptr = projected[i];
    const DenseDataset<FloatT>& cur_centers = centers[i];
    if (projected_ptr.IsSparse()) {
      for (size_t j = 0; j < num_clusters_per_block; ++j) {
        result_row_start[j] = static_cast<float>(
            lookup_distance.GetDistanceHybrid(projected_ptr, cur_centers[j]));
      }
    } else {
      if (lookup_distance.specially_optimized_distance_tag() ==
          DistanceMeasure::LIMITED_INNER_PRODUCT) {
        DenseDistanceOneToMany(
            DotProductDistance(), projected_ptr, cur_centers,
            MutableSpan<float>(result_row_start, num_clusters_per_block));
      } else {
        DenseDistanceOneToMany(
            lookup_distance, projected_ptr, cur_centers,
            MutableSpan<float>(result_row_start, num_clusters_per_block));
      }
    }
  }

  return std::move(result);
}

namespace {
float ComputeMultiplierByQuantile(ConstSpan<float> raw_lookup, float quantile,
                                  int32_t max_integer_value) {
  const size_t k = raw_lookup.size() * (1.0 - quantile) + 1;
  if (k == 1) {
    const float max_abs_lookup_element = std::max(
        std::sqrt(numeric_limits<float>::epsilon()), MaxAbsValue(raw_lookup));
    return max_integer_value / max_abs_lookup_element;
  } else {
    DCHECK_LT(quantile, 1.0f);
    TopNAmortizedConstant<float> tn(k);
    for (auto& elem : raw_lookup) {
      tn.push(std::abs(elem));
    }
    return max_integer_value / tn.exact_bottom();
  }
}

template <typename T, typename Lambda>
inline vector<T> ConvertLookupToFixedPointImpl(ConstSpan<float> raw_lookup,
                                               Lambda convert_to_int_lambda,
                                               float multiplier) {
  constexpr T kBias = FixedPointBias<T>();
  vector<T> result(raw_lookup.size());
  for (size_t i = 0; i < raw_lookup.size(); ++i) {
    result[i] = convert_to_int_lambda(raw_lookup[i] * multiplier) + kBias;
  }
  return result;
}

}  // namespace

template <typename T>
vector<T> ConvertLookupToFixedPoint(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&
        conversion_options,
    float* multiplier) {
  DCHECK_GT(conversion_options.multiplier_quantile(), 0.0f);
  DCHECK_LE(conversion_options.multiplier_quantile(), 1.0f);
  using SignedT = make_signed_t<T>;
  *multiplier = ComputeMultiplierByQuantile(
      raw_lookup, conversion_options.multiplier_quantile(),
      numeric_limits<SignedT>::max());
  constexpr int kRound =
      AsymmetricHasherConfig::FixedPointLUTConversionOptions::ROUND;
  if (conversion_options.multiplier_quantile() == 1.0f) {
    if (conversion_options.float_to_int_conversion_method() == kRound) {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return std::lround(f); }, *multiplier);
    } else {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return static_cast<SignedT>(f); },
          *multiplier);
    }
  } else {
    auto compress_to_bounds = [](float f) {
      f = std::min<float>(f, numeric_limits<SignedT>::max());
      return std::max<float>(f, numeric_limits<SignedT>::min());
    };
    if (conversion_options.float_to_int_conversion_method() == kRound) {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup,
          [&](float f) {
            return static_cast<SignedT>(std::lround(compress_to_bounds(f)));
          },
          *multiplier);
    } else {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup,
          [&](float f) { return static_cast<SignedT>(compress_to_bounds(f)); },
          *multiplier);
    }
  }
}

template vector<uint8_t> ConvertLookupToFixedPoint<uint8_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);
template vector<uint16_t> ConvertLookupToFixedPoint<uint16_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);

bool CanUseInt16Accumulator(ConstSpan<uint8_t> lookup_table,
                            size_t num_blocks) {
  const size_t num_centers_per_block = lookup_table.size() / num_blocks;
  DCHECK_EQ(lookup_table.size() % num_blocks, 0);
  if (num_centers_per_block != 16) return false;
  constexpr uint8_t kBias = FixedPointBias<uint8_t>();

  constexpr size_t kMinMin =
      numeric_limits<int16_t>::min() / numeric_limits<int8_t>::min();
  constexpr size_t kMaxMax =
      numeric_limits<int16_t>::max() / numeric_limits<int8_t>::max();
  constexpr size_t kGuaranteedToWork = (kMaxMax < kMinMin) ? kMaxMax : kMinMin;
  if (num_blocks <= kGuaranteedToWork) return true;

  int32_t sum_of_maxes = 0, sum_of_mins = 0;
  auto block_start = lookup_table.begin();
  for (size_t block = num_blocks; block != 0;
       --block, block_start += num_centers_per_block) {
    int8_t max_val = numeric_limits<int8_t>::min();
    int8_t min_val = numeric_limits<int8_t>::max();
    for (size_t i = 0; i < 16; ++i) {
      const int8_t unbiased = block_start[i] - kBias;
      max_val = std::max<int8_t>(unbiased, max_val);
      min_val = std::min<int8_t>(unbiased, min_val);
    }

    sum_of_mins += static_cast<int32_t>(min_val);
    sum_of_maxes += static_cast<int32_t>(max_val);
  }

  return sum_of_maxes <= numeric_limits<int16_t>::max() &&
         sum_of_mins >= numeric_limits<int16_t>::min();
}

vector<uint8_t> CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database) {
  vector<uint8_t> packed_dataset;
  if (hashed_database.empty()) {
    return packed_dataset;
  }

  DimensionIndex num_blocks = hashed_database[0].nonzero_entries();
  packed_dataset.resize(num_blocks * ((hashed_database.size() + 31) & (~31)) /
                        2);
  DatapointIndex k = 0;
  for (; k < hashed_database.size() / 32; ++k) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        uint8_t u0 = hashed_database[k * 32 + m].values()[j];
        uint8_t u1 = hashed_database[k * 32 + m + 16].values()[j];
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }

  if (k * 32 < hashed_database.size()) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        DatapointIndex dp_idx = k * 32 + m;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u0 = hashed_database[dp_idx].values()[j];

        dp_idx = k * 32 + m + 16;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u1 = hashed_database[dp_idx].values()[j];
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }

  return packed_dataset;
}

template class UnrestrictedIndexIterator<6, IdentityPostprocessFunctor>;
template class UnrestrictedIndexIterator<6, AddBiasFunctor>;
template class UnrestrictedIndexIterator<6, LimitedInnerFunctor>;
template class PopulateDistancesIterator<6, IdentityPostprocessFunctor>;
template class PopulateDistancesIterator<6, AddBiasFunctor>;
template class PopulateDistancesIterator<6, LimitedInnerFunctor>;

SCANN_INSTANTIATE_TYPED_CLASS(, AhImpl);

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow
