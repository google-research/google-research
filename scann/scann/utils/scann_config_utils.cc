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

#include "scann/utils/scann_config_utils.h"

#include <string>

#include "scann/data_format/features.pb.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/compressed_reordering.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/evaluation.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/input_output.pb.h"
#include "scann/proto/metadata.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/projection.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/errors.h"

using absl::StartsWith;

namespace tensorflow {
namespace scann_ops {

namespace {

Status CanonicalizeDeprecatedFields(ScannConfig* config) {
  if (config->has_partitioning() &&
      config->partitioning().use_float_centers_for_query_tokenization()) {
    config->mutable_partitioning()->set_query_tokenization_type(
        PartitioningConfig::FLOAT);
    config->mutable_partitioning()
        ->clear_use_float_centers_for_query_tokenization();
  }
  if (config->hash().asymmetric_hash().has_min_number_machines() &&
      !config->hash().asymmetric_hash().has_num_machines()) {
    config->mutable_hash()->mutable_asymmetric_hash()->set_num_machines(
        config->hash().asymmetric_hash().min_number_machines());
  }
  if (config->has_partitioning() && config->partitioning().use_flume_kmeans()) {
    LOG(WARNING) << "use_flume_kmeans to be deprecated, use trainner_type to "
                    "specify FLUME_KMEANS_TRAINER instead.";
    config->mutable_partitioning()->set_trainer_type(
        PartitioningConfig::FLUME_KMEANS_TRAINER);
    config->mutable_partitioning()->clear_use_flume_kmeans();
  }

  if (config->has_brute_force() && config->brute_force().scalar_quantized()) {
    auto* bf = config->mutable_brute_force();
    bf->clear_scalar_quantized();
    bf->mutable_fixed_point()->set_enabled(true);
    if (bf->has_scalar_quantization_multiplier_quantile()) {
      bf->mutable_fixed_point()->set_fixed_point_multiplier_quantile(
          bf->scalar_quantization_multiplier_quantile());
    }
  }
  return OkStatus();
}

Status CanonicalizeRecallCurves(ScannConfig* config) { return OkStatus(); }

Status CanonicalizeScannConfigImpl(ScannConfig* config) {
  SCANN_RETURN_IF_ERROR(CanonicalizeDeprecatedFields(config));
  SCANN_RETURN_IF_ERROR(EnsureCorrectNormalizationForDistanceMeasure(config));
  auto io = config->mutable_input_output();

  if (io->preprocessed_artifacts_dir().empty()) {
    return CanonicalizeRecallCurves(config);
  }
  const std::string& artifacts_dir = io->preprocessed_artifacts_dir();

  if (config->has_partitioning()) {
    const auto partitioning_config = config->partitioning();
    if (partitioning_config.has_database_spilling()) {
      const auto database_spilling_type =
          partitioning_config.database_spilling().spilling_type();
      const auto trainer_type = partitioning_config.trainer_type();

      if (trainer_type != PartitioningConfig::DEFAULT_SAMPLING_TRAINER &&
          (database_spilling_type == DatabaseSpillingConfig::ADDITIVE ||
           database_spilling_type == DatabaseSpillingConfig::MULTIPLICATIVE)) {
        return InvalidArgumentError(
            "ADDITIVE and MULTIPLICATIVE database spilling is only supported "
            "by the DEFAULT_SAMPLING_TRAINER");
      }
    }

    if (config->has_hash() && config->hash().has_asymmetric_hash()) {
      const auto& asymmetric_hash = config->hash().asymmetric_hash();
      if (asymmetric_hash.use_normalized_residual_quantization()) {
        if (!asymmetric_hash.use_residual_quantization()) {
          return InvalidArgumentError(
              "use_normalized_residual_quantization can only be used when "
              "use_residual_quantization is also turned on");
        }

        config->mutable_partitioning()->set_compute_residual_stdev(true);
      }
    }
  }

  if (!io->has_database_wildcard()) {
    return CanonicalizeRecallCurves(config);
  }

  return OkStatus();
}

}  // namespace

Status CanonicalizeScannConfigForRetrieval(ScannConfig* config) {
  SCANN_RETURN_IF_ERROR(CanonicalizeScannConfigImpl(config));

  auto io = config->mutable_input_output();
  if (io->preprocessed_artifacts_dir().empty()) return OkStatus();
  return OkStatus();
}

StatusOr<InputOutputConfig::InMemoryTypes> TagFromGFVFeatureType(
    const GenericFeatureVector::FeatureType& feature_type) {
  switch (feature_type) {
    case GenericFeatureVector::INT64:
      return InputOutputConfig::INT64;
    case GenericFeatureVector::FLOAT:
      return InputOutputConfig::FLOAT;
    case GenericFeatureVector::DOUBLE:
      return InputOutputConfig::DOUBLE;
    case GenericFeatureVector::BINARY:
      return InputOutputConfig::UINT8;
    default:
      return InvalidArgumentError("Invalid feature_type");
  }
}

StatusOr<InputOutputConfig::InMemoryTypes> DetectInMemoryTypeFromGfv(
    const GenericFeatureVector& gfv) {
  TF_ASSIGN_OR_RETURN(auto ret, TagFromGFVFeatureType(gfv.feature_type()));
  return ret;
}

StatusOr<InputOutputConfig::InMemoryTypes> DetectInMemoryTypeFromDisk(
    const ScannConfig& config) {
  if (!config.has_input_output()) {
    return InvalidArgumentError("config must have input_output.");
  }

  const auto memory_data_type = config.input_output().in_memory_data_type();

  if (memory_data_type !=
      InputOutputConfig::IN_MEMORY_DATA_TYPE_NOT_SPECIFIED) {
    return memory_data_type;
  }

  if (!config.input_output().has_database_wildcard()) {
    return InvalidArgumentError(
        "config.input_output() must have database_wildcard if "
        "in_memory_data_type is not explicitly specified.");
  }
  return InvalidArgumentError("Input GFV from disk not supported.");
}

StatusOr<Normalization> NormalizationRequired(
    const std::string& distance_measure_name) {
  TF_ASSIGN_OR_RETURN(auto distance, GetDistanceMeasure(distance_measure_name));
  return distance->NormalizationRequired();
}

Status EnsureCorrectNormalizationForDistanceMeasure(ScannConfig* config) {
  std::string main_distance_measure;
  if (config->has_distance_measure()) {
    main_distance_measure = config->distance_measure().distance_measure();
  } else if (config->has_partitioning()) {
    if (config->partitioning().has_partitioning_distance()) {
      main_distance_measure =
          config->partitioning().partitioning_distance().distance_measure();
    } else {
      main_distance_measure = "SquaredL2Distance";
    }
  } else {
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(const Normalization expected_normalization,
                      NormalizationRequired(main_distance_measure));
  const bool normalization_user_specified =
      config->input_output().has_norm_type();

  if (expected_normalization != NONE) {
    if (normalization_user_specified) {
      if (static_cast<Normalization>(config->input_output().norm_type()) !=
          expected_normalization) {
        return InvalidArgumentError(
            "Normalization required by the main distance measure %s (%s) "
            "does not match normalization specified in "
            "input_output.norm_type (%s).",
            main_distance_measure.c_str(),
            NormalizationString(expected_normalization),
            NormalizationString(static_cast<Normalization>(
                config->input_output().norm_type())));
      }
    } else {
      config->mutable_input_output()->set_norm_type(
          static_cast<InputOutputConfig::FeatureNorm>(expected_normalization));
    }

    TF_ASSIGN_OR_RETURN(InputOutputConfig::InMemoryTypes in_memory_type,
                        DetectInMemoryTypeFromDisk(*config));
    if (in_memory_type != InputOutputConfig::FLOAT &&
        in_memory_type != InputOutputConfig::DOUBLE) {
      LOG(WARNING) << "Performing "
                   << NormalizationString(expected_normalization)
                   << " normalization with an integral type.";
    }
  }

  auto verify_consistency = [&](const std::string& secondary_distance_measure,
                                const std::string& param_name) -> Status {
    TF_ASSIGN_OR_RETURN(const Normalization secondary_expected,
                        NormalizationRequired(secondary_distance_measure));
    if (secondary_expected != expected_normalization &&
        !(normalization_user_specified && secondary_expected == NONE)) {
      return InvalidArgumentError(
          "Normalization required by main distance measure (%s) does not "
          "match normalization required by %s distance measure "
          "(%s).",
          NormalizationString(expected_normalization), param_name.c_str(),
          NormalizationString(secondary_expected));
    }

    return OkStatus();
  };

  if (config->partitioning().has_partitioning_distance()) {
    SCANN_RETURN_IF_ERROR(verify_consistency(
        config->partitioning().partitioning_distance().distance_measure(),
        "partitioning"));
  }

  if (config->partitioning().has_database_tokenization_distance_override()) {
    SCANN_RETURN_IF_ERROR(
        verify_consistency(config->partitioning()
                               .database_tokenization_distance_override()
                               .distance_measure(),
                           "database tokenization"));
  }

  if (config->partitioning().has_query_tokenization_distance_override()) {
    SCANN_RETURN_IF_ERROR(
        verify_consistency(config->partitioning()
                               .query_tokenization_distance_override()
                               .distance_measure(),
                           "query tokenization"));
  }

  if (config->exact_reordering().has_approx_distance_measure()) {
    SCANN_RETURN_IF_ERROR(verify_consistency(
        config->exact_reordering().approx_distance_measure().distance_measure(),
        "approximate"));
  }

  if (config->has_metadata() && config->metadata().metadata_type_case() ==
                                    MetadataConfig::kExactDistance) {
    SCANN_RETURN_IF_ERROR(verify_consistency(
        config->metadata().exact_distance().distance_measure(), "metadata"));
  }

  return OkStatus();
}

}  // namespace scann_ops
}  // namespace tensorflow
