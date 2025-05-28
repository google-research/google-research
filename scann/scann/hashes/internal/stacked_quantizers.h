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



#ifndef SCANN_HASHES_INTERNAL_STACKED_QUANTIZERS_H_
#define SCANN_HASHES_INTERNAL_STACKED_QUANTIZERS_H_

#include <cmath>
#include <cstdint>

#include "absl/log/log.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_options_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

class CodesList;

template <typename T>
class StackedQuantizers {
 public:
  using FloatT = FloatingTypeFor<T>;
  using TrainingOptions = asymmetric_hashing2::TrainingOptionsTyped<T>;

  template <typename U>
  using CodebookList = vector<DenseDataset<U>>;
  template <typename U>
  using CodebookListView = ConstSpan<DenseDataset<U>>;

  static Status Hash(const DatapointPtr<T>& input,
                     const ChunkingProjection<T>& projector,
                     const DistanceMeasure& quantization_distance,
                     CodebookListView<FloatT> codebook_list,
                     Datapoint<uint8_t>* output);

  static Status Hash(const DatapointPtr<T>& input,
                     const ChunkingProjection<T>& projector,
                     const DistanceMeasure& quantization_distance,
                     CodebookListView<FloatT> codebook_list,
                     MutableSpan<uint8_t> output);

  static Datapoint<FloatT> Reconstruct(const DatapointPtr<uint8_t>& input,
                                       CodebookListView<FloatT> codebook_list);

  static void Reconstruct(ConstSpan<uint8_t> input,
                          CodebookListView<FloatT> codebook_list,
                          MutableSpan<FloatingTypeFor<T>> output);

  static StatusOr<CodebookList<FloatT>> Train(const DenseDataset<T>& dataset,
                                              const TrainingOptions& opts,
                                              shared_ptr<ThreadPool> pool);

  template <typename U>
  static void NoiseShapeQuantizedVector(
      const DatapointPtr<T>& maybe_partitioning_residual_dptr,
      const DatapointPtr<T>& original_dptr, CodebookListView<U> codebook_list,
      double threshold, double eta, MutableSpan<uint8_t> mutable_codes);

 private:
  static double Square(double x) { return x * x; }
  static double ComputeParallelCostMultiplier(double t, double squared_l2_norm,
                                              DimensionIndex dims) {
    const double parallel_cost = Square(t) / squared_l2_norm;
    const double perpendicular_cost =
        (1.0 - Square(t) / squared_l2_norm) / (dims - 1.0);
    return parallel_cost / perpendicular_cost;
  }

  static StatusOr<const DenseDataset<double>*> SampledDataset(
      const DenseDataset<T>& dataset, const TrainingOptions& opts,
      DenseDataset<double>* buffer);

  static StatusOr<CodebookList<double>> HierarchicalKMeans(
      const DenseDataset<double>& dataset, const TrainingOptions& opts,
      int num_codebooks, shared_ptr<ThreadPool> pool);

  template <typename U>
  static void GreedilyAssignCodes(const DatapointPtr<U>& input,
                                  const DistanceMeasure& quantization_distance,
                                  CodebookListView<U> codebook_list,
                                  MutableSpan<uint8_t> mutable_codes,
                                  Datapoint<U>* output_residual = nullptr);

  template <typename U>
  static bool NoiseShapeQuantizedVectorOneBlock(
      const DatapointPtr<T>& maybe_partitioning_residual_dptr,
      const DatapointPtr<T>& original_dptr, CodebookListView<U> codebook_list,
      size_t block_idx, double eta, MutableSpan<uint8_t> mutable_codes,
      double* old_cost_ptr, double* new_cost_ptr);

  static double ComputeAnisotropicCost(DatapointPtr<FloatT> ah_residual_dptr,
                                       DatapointPtr<T> original_dptr,
                                       double eta);

  static vector<Datapoint<double>> ComputeUpdatesToCodebook(
      int codebook_i, DimensionIndex dim, int num_centers,
      const DistanceMeasure& quantization_distance, const CodesList& codes_list,
      const DenseDataset<double>& residual);

  static Status InitializeCodes(const DenseDataset<double>& dataset,
                                const DistanceMeasure& quantization_distance,
                                CodebookListView<double> codebook_list,
                                CodesList* codes_list,
                                DenseDataset<double>* residual,
                                ThreadPool* pool);
};

template <typename T>
double StackedQuantizers<T>::ComputeAnisotropicCost(
    const DatapointPtr<FloatT> ah_residual_dptr,
    const DatapointPtr<T> original_dptr, double eta) {
  const double inv_original_norm =
      1.0 / std::sqrt(SquaredL2Norm(original_dptr));
  const double parallel_component =
      Square(DotProduct(ah_residual_dptr, original_dptr) * inv_original_norm);
  double perpendicular_component = 0.0;
  for (size_t dim_idx : Seq(ah_residual_dptr.nonzero_entries())) {
    const double residual_coordinate = ah_residual_dptr.values()[dim_idx];
    const double residual_perp_dim =
        residual_coordinate - original_dptr.values()[dim_idx] *
                                  inv_original_norm * residual_coordinate;
    perpendicular_component += Square(residual_perp_dim);
  }
  return eta * parallel_component + perpendicular_component;
}

template <typename T>
template <typename U>
bool StackedQuantizers<T>::NoiseShapeQuantizedVectorOneBlock(
    const DatapointPtr<T>& maybe_partitioning_residual_dptr,
    const DatapointPtr<T>& original_dptr, CodebookListView<U> codebook_list,
    size_t block_idx, double eta, MutableSpan<uint8_t> mutable_codes,
    double* old_cost_ptr, double* new_cost_ptr) {
  Datapoint<FloatT> reconstructed =
      Reconstruct(MakeDatapointPtr<uint8_t>(mutable_codes), codebook_list);
  vector<FloatT> ah_residual(reconstructed.nonzero_entries());
  for (size_t dim_idx : IndicesOf(ah_residual)) {
    ah_residual[dim_idx] = maybe_partitioning_residual_dptr.values()[dim_idx] -
                           reconstructed.values()[dim_idx];
  }
  const DenseDataset<U>& codebook = codebook_list[block_idx];
  const double old_cost = ComputeAnisotropicCost(
      MakeDatapointPtr<FloatT>(ah_residual), original_dptr, eta);
  double cur_cost = old_cost;
  const uint8_t old_code = mutable_codes[block_idx];
  uint8_t cur_code = old_code;
  const DatapointPtr<FloatT> old_centroid = codebook[old_code];
  for (uint8_t new_code : IndicesOf(codebook)) {
    if (new_code == old_code) continue;
    const DatapointPtr<FloatT> new_centroid = codebook[new_code];
    vector<FloatT> new_ah_residual = ah_residual;
    for (size_t dim_idx : IndicesOf(new_ah_residual)) {
      new_ah_residual[dim_idx] += old_centroid.values()[dim_idx];
      new_ah_residual[dim_idx] -= new_centroid.values()[dim_idx];
    }
    const double new_cost = ComputeAnisotropicCost(
        MakeDatapointPtr<FloatT>(new_ah_residual), original_dptr, eta);
    if (new_cost < cur_cost) {
      cur_code = new_code;
      cur_cost = new_cost;
    }
  }
  mutable_codes[block_idx] = cur_code;
  if (cur_cost < old_cost) {
    VLOG(2) << "Old cost = " << old_cost << ", new cost = " << cur_cost;
  }
  if (old_cost_ptr) *old_cost_ptr = old_cost;
  if (new_cost_ptr) *new_cost_ptr = cur_cost;
  return cur_code != old_code;
}

template <typename T>
template <typename U>
void StackedQuantizers<T>::NoiseShapeQuantizedVector(
    const DatapointPtr<T>& maybe_partitioning_residual_dptr,
    const DatapointPtr<T>& original_dptr, CodebookListView<U> codebook_list,
    double threshold, double eta, MutableSpan<uint8_t> mutable_codes) {
  if (std::isnan(eta)) {
    eta = ComputeParallelCostMultiplier(threshold, SquaredL2Norm(original_dptr),
                                        original_dptr.dimensionality());
  }
  double init_cost = 0.0, final_cost = 0.0;
  bool changes = true;
  constexpr int kMaxIters = 10;
  for (int iter = 0; iter < kMaxIters && changes; ++iter) {
    changes = false;
    for (size_t block_idx : IndicesOf(codebook_list)) {
      double* old_cost_ptr =
          (iter == 0 && block_idx == 0) ? &init_cost : nullptr;
      changes |= NoiseShapeQuantizedVectorOneBlock(
          maybe_partitioning_residual_dptr, original_dptr, codebook_list,
          block_idx, eta, mutable_codes, old_cost_ptr, &final_cost);
    }
  }
  VLOG(1) << "Initial cost = " << init_cost << ", final cost = " << final_cost;
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, StackedQuantizers);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
