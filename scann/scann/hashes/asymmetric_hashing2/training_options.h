// Copyright 2022 The Google Research Authors.
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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_options_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

template <typename T>
class TrainingOptions : public TrainingOptionsTyped<T> {
 public:
  TrainingOptions(shared_ptr<const ChunkingProjection<T>> projector,
                  shared_ptr<const DistanceMeasure> quantization_distance)
      : TrainingOptionsTyped<T>(projector, std::move(quantization_distance)) {}

  TrainingOptions(const AsymmetricHasherConfig& config,
                  shared_ptr<const DistanceMeasure> quantization_distance,
                  const TypedDataset<T>& dataset);

  Status Validate() const;

 private:
  Status constructor_error_ = OkStatus();
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TrainingOptions);

}  // namespace asymmetric_hashing2
}  // namespace research_scann

#endif
