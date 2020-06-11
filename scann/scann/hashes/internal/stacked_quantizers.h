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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__HASHES_INTERNAL_STACKED_QUANTIZERS_H_
#define SCANN__HASHES_INTERNAL_STACKED_QUANTIZERS_H_

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_options_base.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {
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

  static StatusOr<CodebookList<FloatT>> Train(
      const DenseDataset<T>& dataset, const TrainingOptions& opts,
      shared_ptr<thread::ThreadPool> pool);

 private:
  static StatusOr<const DenseDataset<double>*> SampledDataset(
      const DenseDataset<T>& dataset, const TrainingOptions& opts,
      DenseDataset<double>* buffer);

  static StatusOr<CodebookList<double>> HierarchicalKMeans(
      const DenseDataset<double>& dataset, const TrainingOptions& opts,
      int num_codebooks, shared_ptr<thread::ThreadPool> pool);

  template <typename U>
  static void GreedilyAssignCodes(const DatapointPtr<U>& input,
                                  const DistanceMeasure& quantization_distance,
                                  CodebookListView<U> codebook_list,
                                  MutableSpan<uint8_t> mutable_codes,
                                  Datapoint<U>* output_residual = nullptr);

  static vector<Datapoint<double>> ComputeUpdatesToCodebook(
      int codebook_i, DimensionIndex dim, int num_centers,
      const DistanceMeasure& quantization_distance, const CodesList& codes_list,
      const DenseDataset<double>& residual);

  static Status InitializeCodes(const DenseDataset<double>& dataset,
                                const DistanceMeasure& quantization_distance,
                                CodebookListView<double> codebook_list,
                                CodesList* codes_list,
                                DenseDataset<double>* residual,
                                thread::ThreadPool* pool);
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, StackedQuantizers);

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
