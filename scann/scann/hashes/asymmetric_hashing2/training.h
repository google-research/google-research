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



#ifndef SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_H_
#define SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_H_

#include <utility>

#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/hashes/internal/stacked_quantizers.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
StatusOr<unique_ptr<Model<T>>> TrainSingleMachine(
    const TypedDataset<T>& dataset, const TrainingOptions<T>& params,
    shared_ptr<thread::ThreadPool> pool = nullptr) {
  if (params.config().quantization_scheme() ==
      AsymmetricHasherConfig::STACKED) {
    if (!dataset.IsDense())
      return InvalidArgumentError(
          "Stacked quantizers can only process dense datasets.");
    const auto& dense = down_cast<const DenseDataset<T>&>(dataset);
    TF_ASSIGN_OR_RETURN(
        auto centers,
        ::tensorflow::scann_ops::asymmetric_hashing_internal::StackedQuantizers<
            T>::Train(dense, params, pool));
    return Model<T>::FromCenters(std::move(centers),
                                 params.config().quantization_scheme());
  }
  if (params.config().quantization_scheme() ==
      AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    const auto& dense = down_cast<const DenseDataset<T>&>(dataset);
    DenseDataset<T> dataset_no_bias;
    dataset_no_bias.set_dimensionality(dense.dimensionality() - 1);
    dataset_no_bias.Reserve(dense.size());
    for (const auto& dp : dense) {
      SCANN_RETURN_IF_ERROR(dataset_no_bias.Append(
          MakeDatapointPtr(dp.values(), dp.dimensionality() - 1)));
    }

    TF_ASSIGN_OR_RETURN(
        auto centers,
        ::tensorflow::scann_ops::asymmetric_hashing_internal::
            TrainAsymmetricHashing(dataset_no_bias, params, pool));
    auto converted = asymmetric_hashing_internal::ConvertCentersIfNecessary<T>(
        std::move(centers));
    return Model<T>::FromCenters(std::move(converted),
                                 params.config().quantization_scheme());
  } else {
    TF_ASSIGN_OR_RETURN(auto centers,
                        ::tensorflow::scann_ops::asymmetric_hashing_internal::
                            TrainAsymmetricHashing(dataset, params, pool));
    auto converted = asymmetric_hashing_internal::ConvertCentersIfNecessary<T>(
        std::move(centers));
    return Model<T>::FromCenters(std::move(converted),
                                 params.config().quantization_scheme());
  }
}

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
