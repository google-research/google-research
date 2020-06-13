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

#include "scann/hashes/asymmetric_hashing2/indexing.h"

#include "absl/base/optimization.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l1_distance.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/hashes/internal/stacked_quantizers.h"
#include "scann/oss_wrappers/scann_serialize.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
Indexer<T>::Indexer(shared_ptr<const ChunkingProjection<T>> projector,
                    shared_ptr<const DistanceMeasure> quantization_distance,
                    shared_ptr<const Model<T>> model)
    : projector_(std::move(projector)),
      quantization_distance_(std::move(quantization_distance)),
      model_(std::move(model)) {
  auto quantization_scheme = model_->quantization_scheme();
  if (quantization_scheme == AsymmetricHasherConfig::PRODUCT ||
      quantization_scheme == AsymmetricHasherConfig::PRODUCT_AND_BIAS ||
      quantization_scheme == AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    subspace_sizes_.reserve(model_->num_blocks());
    size_t flattend_elements = 0;
    for (const auto& per_block_centers : model_->centers()) {
      uint32_t center_size = per_block_centers.dimensionality();
      uint32_t subspace_size = per_block_centers.n_elements();
      subspace_sizes_.emplace_back(subspace_size, center_size);
      flattend_elements += subspace_size;
    }
    flattend_model_.resize(flattend_elements);
    FloatT* flattend_ptr = flattend_model_.data();
    for (const auto& per_block_centers : model_->centers()) {
      size_t elements = per_block_centers.n_elements();
      memcpy(flattend_ptr, per_block_centers.data().data(),
             elements * sizeof(FloatT));
      flattend_ptr += elements;
    }
  }
}

template <typename T>
Status Indexer<T>::Hash(const DatapointPtr<T>& input,
                        MutableSpan<uint8_t> hashed) const {
  if (model_->quantization_scheme() == AsymmetricHasherConfig::PRODUCT) {
    DCHECK_EQ(hashed.size(), hash_space_dimension());
    return asymmetric_hashing_internal::IndexDatapoint<T>(
        input, *projector_, *quantization_distance_, model_->centers(), hashed);
  } else if (model_->quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    DCHECK_EQ(hashed.size(), hash_space_dimension());

    SCANN_RETURN_IF_ERROR(asymmetric_hashing_internal::IndexDatapoint<T>(
        MakeDatapointPtr(input.values(), input.dimensionality() - 1),
        *projector_, *quantization_distance_, model_->centers(), hashed));
    std::string s =
        strings::FloatToKey(static_cast<float>(input.values_slice().back()));
    DCHECK_EQ(sizeof(float), s.size());
    const auto dim = hash_space_dimension() - sizeof(float);
    for (int i = 0; i < sizeof(float); i++) {
      hashed[dim + i] = static_cast<uint8_t>(s[i]);
    }
    return OkStatus();
  } else if (model_->quantization_scheme() == AsymmetricHasherConfig::STACKED) {
    DCHECK_EQ(hashed.size(), hash_space_dimension());
    return asymmetric_hashing_internal::StackedQuantizers<T>::Hash(
        input, *projector_, *quantization_distance_, model_->centers(), hashed);
  } else if (model_->quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    DCHECK_EQ(hashed.size(), hash_space_dimension());
    std::vector<uint8_t> unpacked(model_->centers().size());
    SCANN_RETURN_IF_ERROR(asymmetric_hashing_internal::IndexDatapoint<T>(
        input, *projector_, *quantization_distance_, model_->centers(),
        MakeMutableSpan(unpacked)));
    PackNibblesDatapoint(unpacked, hashed);
    return OkStatus();
  } else {
    return UnimplementedError(
        "The model's quantization scheme is not supproted.");
  }
}

template <typename T>
Status Indexer<T>::Hash(ConstSpan<T> input, MutableSpan<uint8_t> hashed) const {
  return Hash(MakeDatapointPtr(input), hashed);
}

template <typename T>
Status Indexer<T>::Hash(const DatapointPtr<T>& input,
                        Datapoint<uint8_t>* hashed) const {
  hashed->clear();
  if (model_->quantization_scheme() ==
      AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    hashed->set_dimensionality(model_->centers().size());
  }
  hashed->mutable_values()->resize(hash_space_dimension());
  return Hash(input, hashed->mutable_values_slice());
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(const DatapointPtr<T>& input,
                                        Datapoint<uint8_t>* hashed,
                                        double threshold) const {
  return HashWithNoiseShaping(input, input, hashed, threshold);
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(const DatapointPtr<T>& input,
                                        MutableSpan<uint8_t> hashed,
                                        double threshold) const {
  return HashWithNoiseShaping(input, input, hashed, threshold);
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(ConstSpan<T> input,
                                        MutableSpan<uint8_t> hashed,
                                        double threshold) const {
  return HashWithNoiseShaping(input, input, hashed, threshold);
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(const DatapointPtr<T>& maybe_residual,
                                        const DatapointPtr<T>& original,
                                        Datapoint<uint8_t>* hashed,
                                        double threshold) const {
  hashed->mutable_values()->resize(hash_space_dimension());
  return HashWithNoiseShaping(maybe_residual, original,
                              hashed->mutable_values_slice(), threshold);
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(const DatapointPtr<T>& maybe_residual,
                                        const DatapointPtr<T>& original,
                                        MutableSpan<uint8_t> hashed,
                                        double threshold) const {
  if (quantization_distance_->specially_optimized_distance_tag() !=
      DistanceMeasure::SQUARED_L2) {
    return FailedPreconditionError(
        "Cannot perform noise-shaped hashing with a non-Squared L2 distance "
        "measure.");
  }
  if (!original.IsDense() || !maybe_residual.IsDense()) {
    return UnimplementedError(
        "Noised-shaped hashing only works with dense inputs for now.");
  }
  if (model_->quantization_scheme() != AsymmetricHasherConfig::PRODUCT) {
    return UnimplementedError(
        "Noise-shaped hashing only works with product quantization for now.");
  }
  return asymmetric_hashing_internal::IndexDatapointNoiseShaped(
      maybe_residual, original, *projector_, model_->centers(), threshold,
      hashed);
}

template <typename T>
Status Indexer<T>::HashWithNoiseShaping(ConstSpan<T> maybe_residual,
                                        ConstSpan<T> original,
                                        MutableSpan<uint8_t> hashed,
                                        double threshold) const {
  return HashWithNoiseShaping(MakeDatapointPtr(maybe_residual),
                              MakeDatapointPtr(original), hashed, threshold);
}

template <typename T>
Status Indexer<T>::Hash(const DatapointPtr<T>& input,
                        std::string* hashed) const {
  hashed->resize(hash_space_dimension());
  auto mutable_span = MakeMutableSpan(
      reinterpret_cast<uint8_t*>(const_cast<char*>(hashed->data())),
      hash_space_dimension());
  SCANN_RETURN_IF_ERROR(Hash(input, mutable_span));
  return OkStatus();
}

namespace {

template <typename FloatT>
SCANN_INLINE void ReconstructProductQuantized(
    const std::vector<FloatT>& flattend_model,
    const std::vector<std::pair<uint32_t, uint32_t>>& subspace_sizes,
    ConstSpan<uint8_t> input, MutableSpan<FloatT> reconstructed) {
  DCHECK_LE(subspace_sizes.size(), input.size());

  FloatT* __restrict result_ptr = reconstructed.data();
  const FloatT* __restrict src_ptr = flattend_model.data();
  const uint8_t* input_ptr = input.data();

  uint32_t subspace_size, center_size;
  for (const auto& subspace_info : subspace_sizes) {
    std::tie(subspace_size, center_size) = subspace_info;
    uint32_t center_idx = (*input_ptr) * center_size;
    memcpy(result_ptr, src_ptr + center_idx, sizeof(FloatT) * center_size);
    result_ptr += center_size;
    src_ptr += subspace_size;
    input_ptr++;
  }
}

template <typename FloatT, typename Reduce>
SCANN_INLINE FloatT ComputeDistance(
    ConstSpan<FloatT>& original, ConstSpan<uint8_t> hashed,
    const std::vector<FloatT>& flattend_model,
    const std::vector<std::pair<uint32_t, uint32_t>>& subspace_sizes,
    Reduce reduce) {
  const FloatT* codebook_ptr = flattend_model.data();
  const FloatT* original_ptr = original.data();
  const uint8_t* code_ptr = hashed.data();
  FloatT acc0 = 0, acc1 = 0, acc_odd = 0;

  uint32_t subspace_size, center_size;
  for (const auto& subspace_info : subspace_sizes) {
    std::tie(subspace_size, center_size) = subspace_info;
    uint32_t center_idx = (*code_ptr) * center_size;
    const FloatT* center_ptr = codebook_ptr + center_idx;
    const FloatT* original_it = original_ptr;
    const FloatT* center_ptr_bound = center_ptr + center_size;

    if (center_size & 1) {
      reduce(&acc_odd, original_ptr[center_size - 1],
             center_ptr[center_size - 1]);
    }

    for (; center_ptr + 1 < center_ptr_bound;
         center_ptr += 2, original_it += 2) {
      reduce(&acc0, original_it[0], center_ptr[0]);
      reduce(&acc1, original_it[1], center_ptr[1]);
    }

    codebook_ptr += subspace_size;

    original_ptr += center_size;

    code_ptr++;
  }
  return acc0 + acc1 + acc_odd;
}

}  // namespace

template <typename T>
StatusOr<DenseDataset<uint8_t>> Indexer<T>::HashDataset(
    const TypedDataset<T>& dataset) const {
  DenseDataset<uint8_t> hashed_dataset;
  Datapoint<uint8_t> hashed_dp;
  for (DatapointPtr<T> dp : dataset) {
    SCANN_RETURN_IF_ERROR(Hash(dp, &hashed_dp));
    hashed_dataset.AppendOrDie(hashed_dp.ToPtr(), "");
  }
  return {std::move(hashed_dataset)};
}

template <typename T>
StatusOr<FloatingTypeFor<T>> Indexer<T>::DistanceBetweenOriginalAndHashed(
    ConstSpan<FloatT> original, ConstSpan<uint8_t> hashed,
    shared_ptr<const DistanceMeasure> distance_override) const {
  shared_ptr<const DistanceMeasure> distance =
      distance_override == nullptr ? quantization_distance_ : distance_override;

  if (ABSL_PREDICT_FALSE(model_->quantization_scheme() !=
                         AsymmetricHasherConfig::PRODUCT)) {
    goto fallback;
  }
  switch (distance->specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
      return -ComputeDistance(original, hashed, flattend_model_,
                              subspace_sizes_, DotProductReduce());
    case DistanceMeasure::COSINE:
      return 1.0 - ComputeDistance(original, hashed, flattend_model_,
                                   subspace_sizes_, DotProductReduce());
    case DistanceMeasure::SQUARED_L2:
      return ComputeDistance(original, hashed, flattend_model_, subspace_sizes_,
                             SquaredL2ReduceTwo());
    case DistanceMeasure::L2:
      return std::sqrt(ComputeDistance(original, hashed, flattend_model_,
                                       subspace_sizes_, SquaredL2ReduceTwo()));
    case tensorflow::scann_ops::DistanceMeasure::L1:
      return ComputeDistance(original, hashed, flattend_model_, subspace_sizes_,
                             L1ReduceTwo());

    default:
      goto fallback;
  }

fallback:
  Datapoint<FloatT> reconstructed;
  SCANN_RETURN_IF_ERROR(Reconstruct(MakeDatapointPtr(hashed), &reconstructed));
  return distance->GetDistance(MakeDatapointPtr(original),
                               reconstructed.ToPtr());
}

template <typename T>
Status Indexer<T>::Reconstruct(ConstSpan<uint8_t> input,
                               MutableSpan<FloatT> reconstructed) const {
  if (model_->quantization_scheme() == AsymmetricHasherConfig::PRODUCT) {
    DCHECK_EQ(input.size(), model_->centers().size());
    ReconstructProductQuantized(flattend_model_, subspace_sizes_, input,
                                reconstructed);
  } else if (model_->quantization_scheme() == AsymmetricHasherConfig::STACKED) {
    asymmetric_hashing_internal::StackedQuantizers<T>::Reconstruct(
        input, model_->centers(), reconstructed);
  } else if (model_->quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    DCHECK_EQ(input.size() - sizeof(float), model_->centers().size());

    ReconstructProductQuantized(flattend_model_, subspace_sizes_, input,
                                reconstructed);

    const float bias = strings::KeyToFloat(string_view(
        reinterpret_cast<const char*>(&input[input.size() - sizeof(float)]),
        sizeof(float)));
    reconstructed[original_space_dimension() - 1] = static_cast<T>(bias);
  } else if (model_->quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    DCHECK_EQ(input.size(), (model_->centers().size() + 1) / 2);
    DimensionIndex unpacked_size = model_->centers().size();
    std::vector<uint8_t> unpacked(unpacked_size);
    UnpackNibblesDatapoint(input, MakeMutableSpan(unpacked), unpacked_size);
    ReconstructProductQuantized(flattend_model_, subspace_sizes_, unpacked,
                                reconstructed);
  } else {
    return UnimplementedError(
        "The model's quantization scheme is not supproted.");
  }
  return OkStatus();
}

template <typename T>
Status Indexer<T>::Reconstruct(const DatapointPtr<uint8_t>& input,
                               Datapoint<FloatT>* reconstructed) const {
  reconstructed->mutable_values()->clear();
  reconstructed->mutable_values()->resize(original_space_dimension());
  return Reconstruct(input.values_slice(),
                     reconstructed->mutable_values_slice());
}

template <typename T>
Status Indexer<T>::Reconstruct(absl::string_view input,
                               Datapoint<FloatT>* reconstructed) const {
  DimensionIndex dimensionality =
      model_->quantization_scheme() == AsymmetricHasherConfig::PRODUCT_AND_PACK
          ? model_->centers().size()
          : input.size();
  DatapointPtr<uint8_t> datapoint = MakeDenseBinaryDatapointPtr(
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(input.data()),
                          input.size()),
      dimensionality);
  return Reconstruct(datapoint, reconstructed);
}

template <typename T>
DimensionIndex Indexer<T>::hash_space_dimension() const {
  DCHECK_EQ(model_->centers().size(), projector_->num_blocks());
  switch (model_->quantization_scheme()) {
    case AsymmetricHasherConfig::PRODUCT:
      return model_->centers().size();
    case AsymmetricHasherConfig::PRODUCT_AND_BIAS:
      return model_->centers().size() + sizeof(float);
    case AsymmetricHasherConfig::STACKED:
      return model_->centers().size();
    case AsymmetricHasherConfig::PRODUCT_AND_PACK:
      return (model_->centers().size() + 1) / 2;
  }
}

template <typename T>
DimensionIndex Indexer<T>::original_space_dimension() const {
  switch (model_->quantization_scheme()) {
    case AsymmetricHasherConfig::PRODUCT:
      return projector_->input_dim();
    case AsymmetricHasherConfig::PRODUCT_AND_BIAS:
      return projector_->input_dim() + 1;
    case AsymmetricHasherConfig::STACKED:
      return model_->centers()[0].dimensionality();
    case AsymmetricHasherConfig::PRODUCT_AND_PACK:
      return projector_->input_dim();
  }
}

template <typename T>
Status Indexer<T>::ComputeResidual(const DatapointPtr<T>& original,
                                   const DatapointPtr<uint8_t>& hashed,
                                   Datapoint<FloatT>* result) const {
  SCANN_RETURN_IF_ERROR(Reconstruct(hashed, result));

  for (DimensionIndex i = 0; i < original.dimensionality(); ++i) {
    (*result->mutable_values())[i] =
        original.GetElement(i) - result->values()[i];
  }
  if (original.dimensionality() < result->dimensionality()) {
    result->mutable_values()->resize(original.dimensionality());
  }
  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, Indexer);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow
