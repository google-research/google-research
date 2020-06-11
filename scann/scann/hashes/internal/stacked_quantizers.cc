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

#include "scann/hashes/internal/stacked_quantizers.h"

#include <functional>
#include <limits>
#include <numeric>

#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/dataset_sampling.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {

namespace {

template <typename O, typename T>
DenseDataset<O> CopyDenseDatasetIntoNewType(const DenseDataset<T>& input) {
  const auto input_size = input.size();

  DenseDataset<O> output;
  output.set_dimensionality(input.dimensionality());
  output.Reserve(input_size);

  Datapoint<O> buffer;
  for (DatapointIndex i = 0; i < input_size; ++i)
    output.AppendOrDie(MaybeConvertDatapoint(input[i], &buffer), "");
  return output;
}

template <typename O, typename T>
enable_if_t<std::is_same<O, T>::value, const DenseDataset<O>&>
ConvertDenseDatasetIntoNewType(const DenseDataset<T>& input,
                               DenseDataset<O>* buffer) {
  return input;
}

template <typename O, typename T>
enable_if_t<!std::is_same<O, T>::value, const DenseDataset<O>&>
ConvertDenseDatasetIntoNewType(const DenseDataset<T>& input,
                               DenseDataset<O>* buffer) {
  *buffer = CopyDenseDatasetIntoNewType<O>(input);
  return *buffer;
}

template <typename O, typename T>
enable_if_t<std::is_same<O, T>::value, DenseDataset<O>*>
ConvertMutableDenseDatasetIntoNewType(DenseDataset<T>* input,
                                      DenseDataset<O>* buffer) {
  return input;
}

template <typename O, typename T>
enable_if_t<!std::is_same<O, T>::value, DenseDataset<O>*>
ConvertMutableDenseDatasetIntoNewType(DenseDataset<T>* input,
                                      DenseDataset<O>* buffer) {
  *buffer = CopyDenseDatasetIntoNewType<O>(*input);
  return buffer;
}

template <typename BinaryOp, typename T, typename O>
void UpdateSpanByScalar(BinaryOp op, T arg, MutableSpan<O> result) {
  const auto d = result.size();
  for (DimensionIndex i = 0; i < d; ++i) result[i] = op(result[i], arg);
}

template <typename BinaryOp, typename T, typename O>
void UpdateSpanByVec(BinaryOp op, const DatapointPtr<T>& arg,
                     MutableSpan<O> result) {
  const auto d = arg.dimensionality();
  DCHECK_EQ(d, result.size());
  const auto* arg_values = arg.values();
  for (DimensionIndex i = 0; i < d; ++i)
    result[i] = op(result[i], arg_values[i]);
}

template <typename BinaryOp, typename T, typename O>
void InplaceUpdateDenseDatapoint(BinaryOp op, T arg, Datapoint<O>* result) {
  UpdateSpanByScalar(op, arg, result->mutable_values_slice());
}

template <typename BinaryOp, typename T, typename O>
void InplaceUpdateDenseDatapoint(BinaryOp op, const DatapointPtr<T>& arg,
                                 Datapoint<O>* result) {
  UpdateSpanByVec(op, arg, result->mutable_values_slice());
}

template <typename T>
double AverageSquaredL2Norm(const DenseDataset<T>& matrix) {
  const auto num_rows = matrix.size();
  if (!num_rows) return 0.0;

  double result = 0.0;
  for (DatapointIndex i = 0; i < num_rows; ++i)
    result += SquaredL2Norm(matrix[i]);
  return result / num_rows;
}

}  // namespace

class CodesList {
 public:
  CodesList(DatapointIndex num_datapoints, int num_codebooks)
      : num_datapoints_(num_datapoints),
        num_codebooks_(num_codebooks),
        codes_(new uint8_t[num_datapoints * num_codebooks]) {}

  ConstSpan<uint8_t> GetCodes(DatapointIndex i) const {
    DCHECK_LT(i, num_datapoints_);
    return ConstSpan<uint8_t>(codes_.get() + i * num_codebooks_,
                              num_codebooks_);
  }

  MutableSpan<uint8_t> GetMutableCodes(DatapointIndex i) {
    DCHECK_LT(i, num_datapoints_);
    return MutableSpan<uint8_t>(codes_.get() + i * num_codebooks_,
                                num_codebooks_);
  }

  DatapointIndex NumDatapoints() const { return num_datapoints_; }

  int NumCodebooks() const { return num_codebooks_; }

 private:
  const DatapointIndex num_datapoints_;
  const int num_codebooks_;
  unique_ptr<uint8_t[]> codes_;
};

template <typename T>
Status StackedQuantizers<T>::Hash(const DatapointPtr<T>& input,
                                  const ChunkingProjection<T>& projector,
                                  const DistanceMeasure& quantization_distance,
                                  CodebookListView<FloatT> codebook_list,
                                  MutableSpan<uint8_t> output) {
  std::fill(output.begin(), output.end(), 0);
  ChunkedDatapoint<FloatT> projected;
  SCANN_RETURN_IF_ERROR(projector.ProjectInput(input, &projected));
  DCHECK_EQ(output.size(), projected.size());
  GreedilyAssignCodes(projected[0], quantization_distance, codebook_list,
                      output);
  return OkStatus();
}

template <typename T>
Status StackedQuantizers<T>::Hash(const DatapointPtr<T>& input,
                                  const ChunkingProjection<T>& projector,
                                  const DistanceMeasure& quantization_distance,
                                  CodebookListView<FloatT> codebook_list,
                                  Datapoint<uint8_t>* output) {
  DCHECK(output);
  const auto num_codebooks = codebook_list.size();
  DCHECK(num_codebooks);
  output->mutable_values()->resize(num_codebooks);
  return StackedQuantizers<T>::Hash(input, projector, quantization_distance,
                                    codebook_list,
                                    output->mutable_values_slice());
}

template <typename T>
Datapoint<FloatingTypeFor<T>> StackedQuantizers<T>::Reconstruct(
    const DatapointPtr<uint8_t>& input,
    CodebookListView<FloatT> codebook_list) {
  const auto num_codebooks = codebook_list.size();
  DCHECK(num_codebooks);
  DCHECK_EQ(num_codebooks, input.dimensionality());
  const auto d = codebook_list[0].dimensionality();

  Datapoint<FloatT> output;
  output.mutable_values()->resize(d, 0);
  const auto* codes = input.values();
  for (int i = 0; i < num_codebooks; ++i)
    InplaceUpdateDenseDatapoint(std::plus<FloatT>(), codebook_list[i][codes[i]],
                                &output);
  return output;
}

template <typename T>
void StackedQuantizers<T>::Reconstruct(ConstSpan<uint8_t> input,
                                       CodebookListView<FloatT> codebook_list,
                                       MutableSpan<FloatingTypeFor<T>> output) {
  const auto num_codebooks = codebook_list.size();
  DCHECK_EQ(num_codebooks, input.size());
  DCHECK_EQ(output.size(), codebook_list[0].dimensionality());
  for (int i = 0; i < num_codebooks; ++i)
    UpdateSpanByVec(std::plus<FloatT>(), codebook_list[i][input[i]], output);
}

template <typename T>

StatusOr<
    typename StackedQuantizers<T>::template CodebookList<FloatingTypeFor<T>>>
StackedQuantizers<T>::Train(const DenseDataset<T>& dataset,
                            const TrainingOptions& opts,
                            shared_ptr<thread::ThreadPool> pool) {
  const auto num_datapoints = dataset.size();
  const auto num_codebooks = opts.projector()->num_blocks();
  const auto num_centers = opts.config().num_clusters_per_block();
  const auto quantization_distance = opts.quantization_distance();
  const auto& sq_config = opts.config().stacked_quantizers_config();

  DenseDataset<double> buffer;
  TF_ASSIGN_OR_RETURN(const auto* converted,
                      SampledDataset(dataset, opts, &buffer));
  VLOG(1) << "SQ training, sampled training set size = " << converted->size();
  TF_ASSIGN_OR_RETURN(
      auto codebook_list,
      HierarchicalKMeans(*converted, opts, num_codebooks, pool));
  CodesList codes_list(num_datapoints, num_codebooks);
  DenseDataset<double> residual;
  SCANN_RETURN_IF_ERROR(InitializeCodes(*converted, *quantization_distance,
                                        codebook_list, &codes_list, &residual,
                                        pool.get()));
  TF_ASSIGN_OR_RETURN(auto* residual_mutator, residual.GetMutator());

  auto mse = AverageSquaredL2Norm(residual);
  VLOG(1) << "SQ training, initial mse = " << mse;

  for (int iter = 0; iter < sq_config.max_num_iterations(); ++iter) {
    for (int codebook_i = 0; codebook_i < num_codebooks; ++codebook_i) {
      auto& codebook = codebook_list[codebook_i];
      TF_ASSIGN_OR_RETURN(auto* codebook_mutator, codebook.GetMutator());
      const auto deltas = ComputeUpdatesToCodebook(
          codebook_i, dataset.dimensionality(), num_centers,
          *quantization_distance, codes_list, residual);
      Datapoint<double> tmp;

      for (DatapointIndex datapoint_i = 0; datapoint_i < num_datapoints;
           ++datapoint_i) {
        const auto code = codes_list.GetCodes(datapoint_i)[codebook_i];
        PointSum(residual[datapoint_i], codebook[code], &tmp);
        SCANN_RETURN_IF_ERROR(
            residual_mutator->UpdateDatapoint(tmp.ToPtr(), datapoint_i));
      }

      for (int center_i = 0; center_i < num_centers; ++center_i) {
        PointSum(codebook[center_i], deltas[center_i].ToPtr(), &tmp);
        SCANN_RETURN_IF_ERROR(
            codebook_mutator->UpdateDatapoint(tmp.ToPtr(), center_i));
      }

      auto neighbors = DenseDistanceManyToManyTop1<double>(
          *quantization_distance, residual, codebook, pool.get());
      for (DatapointIndex datapoint_i = 0; datapoint_i < num_datapoints;
           ++datapoint_i) {
        const auto code = neighbors[datapoint_i].first;
        codes_list.GetMutableCodes(datapoint_i)[codebook_i] = code;
        PointDiff(residual[datapoint_i], codebook[code], &tmp);
        SCANN_RETURN_IF_ERROR(
            residual_mutator->UpdateDatapoint(tmp.ToPtr(), datapoint_i));
      }
    }

    const auto mse_iter = AverageSquaredL2Norm(residual);
    VLOG(1) << "SQ training, at iteration " << iter
            << ", after improving both codebooks and codes, mse = " << mse_iter;
    if (iter >= sq_config.min_num_iterations() - 1) {
      if (mse_iter > mse) {
        LOG(WARNING) << "SQ training, at iteration " << iter
                     << ", mse increases from " << mse << " to " << mse_iter;
        break;
      }
      const auto relative_improvement = (mse - mse_iter) / mse;
      if (relative_improvement < sq_config.relative_improvement_threshold())
        break;
    }
    mse = mse_iter;
  }

  VLOG(1) << "SQ training, final mse = " << mse;

  CodebookList<FloatT> result;
  result.reserve(codebook_list.size());
  DenseDataset<FloatT> tmp;
  for (auto& codebook : codebook_list)
    result.emplace_back(std::move(
        *ConvertMutableDenseDatasetIntoNewType<FloatT>(&codebook, &tmp)));

  return std::move(result);
}

template <typename T>
StatusOr<const DenseDataset<double>*> StackedQuantizers<T>::SampledDataset(
    const DenseDataset<T>& dataset, const TrainingOptions& opts,
    DenseDataset<double>* buffer) {
  const auto dataset_size = dataset.size();
  if (opts.config().sampling_fraction() == 1.0 &&
      !opts.config().max_sample_size()) {
    return &ConvertDenseDatasetIntoNewType(dataset, buffer);
  } else {
    const auto max_sample_size = opts.config().max_sample_size()
                                     ? opts.config().max_sample_size()
                                     : dataset_size;
    TF_ASSIGN_OR_RETURN(
        auto sampled_indices,
        internal::CreateSampledIndexList<DatapointIndex>(
            dataset_size, opts.config().sampling_seed(),
            opts.config().sampling_fraction(), 0, max_sample_size,
            SubsamplingStrategy::kWithoutReplacement));
    buffer->clear();
    Datapoint<double> tmp;
    DatapointIndex i;
    while (sampled_indices.GetNextIndex(&i))
      buffer->AppendOrDie(MaybeConvertDatapoint(dataset[i], &tmp), "");
    return buffer;
  }
}

template <typename T>
StatusOr<typename StackedQuantizers<T>::template CodebookList<double>>
StackedQuantizers<T>::HierarchicalKMeans(const DenseDataset<double>& dataset,
                                         const TrainingOptions& opts,
                                         int num_codebooks,
                                         shared_ptr<thread::ThreadPool> pool) {
  const auto num_centers = opts.config().num_clusters_per_block();

  GmmUtils::Options gmm_opts;
  gmm_opts.seed = opts.config().clustering_seed();
  gmm_opts.max_iterations = opts.config().max_clustering_iterations();
  gmm_opts.epsilon = opts.config().clustering_convergence_tolerance();
  gmm_opts.parallelization_pool = std::move(pool);
  GmmUtils gmm(opts.quantization_distance(), gmm_opts);

  CodebookList<double> codebook_list;
  codebook_list.reserve(num_codebooks);
  auto residual = CopyDenseDatasetIntoNewType<double>(dataset);
  TF_ASSIGN_OR_RETURN(auto* residual_mutator, residual.GetMutator());

  for (auto _ : Seq(num_codebooks)) {
    DenseDataset<double> centers;
    vector<vector<DatapointIndex>> labels;
    SCANN_RETURN_IF_ERROR(
        gmm.GenericKmeans(residual, num_centers, &centers, &labels));
    DCHECK_EQ(labels.size(), num_centers);

    DenseDataset<double> buffer;
    codebook_list.emplace_back(
        std::move(*ConvertMutableDenseDatasetIntoNewType(&centers, &buffer)));
    const auto& codebook = codebook_list.back();
    Datapoint<double> new_residual;
    for (size_t center_i : Seq(num_centers)) {
      for (const auto datapoint_i : labels[center_i]) {
        PointDiff(residual[datapoint_i], codebook[center_i], &new_residual);
        SCANN_RETURN_IF_ERROR(residual_mutator->UpdateDatapoint(
            new_residual.ToPtr(), datapoint_i));
      }
    }
  }

  return std::move(codebook_list);
}

template <typename T>
template <typename U>
void StackedQuantizers<T>::GreedilyAssignCodes(
    const DatapointPtr<U>& input, const DistanceMeasure& quantization_distance,
    CodebookListView<U> codebook_list, MutableSpan<uint8_t> mutable_codes,
    Datapoint<U>* output_residual) {
  const auto num_codebooks = codebook_list.size();
  DCHECK_EQ(num_codebooks, mutable_codes.size());
  const auto num_centers = codebook_list[0].size();
  DCHECK_GE(num_centers, 1);
  DCHECK_LE(num_centers, 256);

  Datapoint<U> residual;
  CopyToDatapoint(input, &residual);
  vector<double> distances(num_centers);
  for (int i = 0; i < num_codebooks; ++i) {
    const auto& codebook = codebook_list[i];
    DCHECK_EQ(num_centers, codebook.size());

    DenseDistanceOneToMany(quantization_distance, residual.ToPtr(), codebook,
                           MutableSpan<double>(distances));
    const auto min_it = std::min_element(distances.begin(), distances.end());
    const auto code = min_it - distances.begin();
    mutable_codes[i] = code;
    InplaceUpdateDenseDatapoint(std::minus<U>(), codebook[code], &residual);
  }

  if (output_residual) *output_residual = std::move(residual);
}

template <typename T>
vector<Datapoint<double>> StackedQuantizers<T>::ComputeUpdatesToCodebook(
    int codebook_i, DimensionIndex dim, int num_centers,
    const DistanceMeasure& quantization_distance, const CodesList& codes_list,
    const DenseDataset<double>& residual) {
  const auto dataset_size = codes_list.NumDatapoints();
  const auto num_codebooks = codes_list.NumCodebooks();
  DCHECK(num_codebooks);

  vector<Datapoint<double>> deltas(num_centers);
  for (auto& delta : deltas) delta.mutable_values()->resize(dim, 0);
  vector<double> num_assigned(num_centers, 0);
  for (DatapointIndex datapoint_i = 0; datapoint_i < dataset_size;
       ++datapoint_i) {
    const auto code = codes_list.GetCodes(datapoint_i)[codebook_i];
    InplaceUpdateDenseDatapoint(std::plus<double>(), residual[datapoint_i],
                                &deltas[code]);
    ++num_assigned[code];
  }
  for (int center_i = 0; center_i < num_centers; ++center_i) {
    if (num_assigned[center_i])
      InplaceUpdateDenseDatapoint(std::divides<double>(),
                                  num_assigned[center_i], &deltas[center_i]);
  }

  return deltas;
}

template <typename T>
Status StackedQuantizers<T>::InitializeCodes(
    const DenseDataset<double>& dataset,
    const DistanceMeasure& quantization_distance,
    CodebookListView<double> codebook_list, CodesList* codes_list,
    DenseDataset<double>* residual, thread::ThreadPool* pool) {
  DCHECK(codes_list && residual);
  const auto dataset_size = dataset.size();
  const auto num_codebooks = codebook_list.size();

  DenseDataset<double> prev_residuals =
      CopyDenseDatasetIntoNewType<double>(dataset);
  DenseDataset<double> updated_residuals;
  Datapoint<double> tmp;
  for (size_t codebook_idx : Seq(num_codebooks)) {
    const auto& codebook = codebook_list[codebook_idx];
    auto neighbors = DenseDistanceManyToManyTop1(
        quantization_distance, prev_residuals, codebook, pool);
    for (DatapointIndex dp_idx : Seq(dataset_size)) {
      MutableSpan<uint8_t> mutable_codes = codes_list->GetMutableCodes(dp_idx);
      DCHECK_EQ(num_codebooks, mutable_codes.size());
      const DatapointIndex code = neighbors[dp_idx].first;
      mutable_codes[codebook_idx] = code;
      CopyToDatapoint(prev_residuals[dp_idx], &tmp);
      InplaceUpdateDenseDatapoint(std::minus<double>(), codebook[code], &tmp);
      updated_residuals.AppendOrDie(tmp.ToPtr(), "");
    }
    std::swap(prev_residuals, updated_residuals);
    updated_residuals.clear();
  }
  *residual = std::move(prev_residuals);
  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, StackedQuantizers);

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow
