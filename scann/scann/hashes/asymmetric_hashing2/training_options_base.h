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



#ifndef SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_BASE_H_
#define SCANN__HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_BASE_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

class TrainingOptionsBase {
 public:
  explicit TrainingOptionsBase(
      shared_ptr<const DistanceMeasure> quantization_distance)
      : quantization_distance_(std::move(quantization_distance)) {}

  TrainingOptionsBase(const AsymmetricHasherConfig& config,
                      shared_ptr<const DistanceMeasure> quantization_distance)
      : conf_(config),
        quantization_distance_(std::move(quantization_distance)) {}

  const shared_ptr<const DistanceMeasure>& quantization_distance() const {
    return quantization_distance_;
  }

  const AsymmetricHasherConfig& config() const { return conf_; }

  AsymmetricHasherConfig* mutable_config() { return &conf_; }

 protected:
  AsymmetricHasherConfig conf_;
  shared_ptr<const DistanceMeasure> quantization_distance_;
};

template <typename T>
class TrainingOptionsTyped : public TrainingOptionsBase {
 public:
  TrainingOptionsTyped(const AsymmetricHasherConfig& config,
                       shared_ptr<const DistanceMeasure> quantization_distance)
      : TrainingOptionsBase(config, std::move(quantization_distance)) {}

  TrainingOptionsTyped(shared_ptr<const ChunkingProjection<T>> projector,
                       shared_ptr<const DistanceMeasure> quantization_distance)
      : TrainingOptionsBase(std::move(quantization_distance)),
        projector_(std::move(projector)) {}

  const shared_ptr<const ChunkingProjection<T>>& projector() const {
    return projector_;
  }

  using PreprocessingFunction =
      std::function<StatusOr<Datapoint<T>>(const DatapointPtr<T>&)>;
  void set_preprocessing_function(PreprocessingFunction fn) {
    preprocessing_function_ = std::move(fn);
  }
  const PreprocessingFunction& preprocessing_function() const {
    return preprocessing_function_;
  }

 protected:
  shared_ptr<const ChunkingProjection<T>> projector_;

  PreprocessingFunction preprocessing_function_;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TrainingOptionsTyped);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
