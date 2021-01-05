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

#include "scann/hashes/asymmetric_hashing2/training_options.h"

#include "scann/projection/projection_factory.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

template <typename T>
TrainingOptions<T>::TrainingOptions(
    const AsymmetricHasherConfig& config,
    shared_ptr<const DistanceMeasure> quantization_distance,
    const TypedDataset<T>& dataset)
    : TrainingOptionsTyped<T>(config, std::move(quantization_distance)) {
  auto statusor = ChunkingProjectionFactory<T>(config.projection(), &dataset);
  if (statusor.ok()) {
    this->projector_ = ValueOrDie(std::move(statusor));
  } else {
    this->constructor_error_ = statusor.status();
  }
}

template <typename T>
Status TrainingOptions<T>::Validate() const {
  SCANN_RETURN_IF_ERROR(constructor_error_);
  if (this->config().num_clusters_per_block() < 1 ||
      this->config().num_clusters_per_block() > 256) {
    return InvalidArgumentError(
        absl::StrCat("num_clusters_per_block must be between 1 and 256, not ",
                     this->config().num_clusters_per_block(), "."));
  }

  if (this->config().max_clustering_iterations() < 1) {
    return InvalidArgumentError(absl::StrCat(
        "max_clustering_iterations must be strictly positive, not ",
        this->config().max_clustering_iterations(), "."));
  }

  if (!(this->config().clustering_convergence_tolerance() > 0)) {
    return InvalidArgumentError(absl::StrCat(
        "clustering_convergence_tolerance must be strictly positive, not ",
        this->config().max_clustering_iterations(), "."));
  }

  if (this->config().min_cluster_size() < 1) {
    return InvalidArgumentError(
        absl::StrCat("min_cluster_size must be strictly positive, not ",
                     this->config().min_cluster_size(), "."));
  }

  if (this->config().sampling_fraction() <= 0.0f ||
      this->config().sampling_fraction() > 1.0f) {
    return InvalidArgumentError(absl::StrCat(
        "sampling_fraction must be strictly positive and <= 1.0, not ",
        this->config().sampling_fraction(), "."));
  }

  if (this->config().max_sample_size() < 1) {
    return InvalidArgumentError(
        absl::StrCat("max_sample_size must be strictly positive, not ",
                     this->config().max_sample_size(), "."));
  }

  if (this->config().has_stacked_quantizers_config()) {
    auto& sq = this->config().stacked_quantizers_config();

    if (sq.min_num_iterations() < 1) {
      return InvalidArgumentError(
          "min_num_iterations for stacked quantizers must be >=1.");
    }

    if (sq.min_num_iterations() > sq.max_num_iterations()) {
      return InvalidArgumentError(
          "min_num_iterations must be <= max_num_iterations for stacked "
          "quantizers.");
    }

    if (sq.relative_improvement_threshold() <= 0.0 ||
        sq.relative_improvement_threshold() >= 1.0) {
      return InvalidArgumentError(
          "relative_improvement_threashold for stacked quantizers must be "
          "within "
          "(0.0, 1.0).");
    }
  }

  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, TrainingOptions);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow
