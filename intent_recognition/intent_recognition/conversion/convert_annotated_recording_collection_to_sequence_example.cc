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

#include "intent_recognition/conversion/convert_annotated_recording_collection_to_sequence_example.h"

#include <string>
#include <string_view>
#include <utility>

#include "google/protobuf/duration.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"


namespace ambient_sensing {
namespace {
using ::tensorflow::SequenceExample;

constexpr absl::string_view kDelimiter = "/";

// Copies float_value into feature_list. It puts exactly 'dimensionality' values
// into a feature, so if float_value is of length 10 and dimensionality is 2,
// then 5 features will be added.
template <typename T>
void CopyDataToFeatureContainer(T datapoints, bool flatten,
                                tensorflow::FeatureList* feature_list) {
  tensorflow::Feature* current_feature;
  if (flatten) {
    current_feature = feature_list->add_feature();
    for (const auto& datapoint : datapoints) {
      for (auto value : datapoint.double_value().value()) {
        current_feature->mutable_float_list()->add_value(value);
      }
    }
  } else {
    for (const auto& datapoint : datapoints) {
      current_feature = feature_list->add_feature();
      for (auto value : datapoint.double_value().value()) {
        current_feature->mutable_float_list()->add_value(value);
      }
    }
  }
}

absl::optional<absl::Duration> UpdateFirstArcMeasurement(
    const google::protobuf::Duration& first_sequence_timestamp_proto,
    absl::optional<absl::Duration> first_arc_measurement) {
  absl::Duration first_sequence_timestamp =
      ConvertProtoToDuration(first_sequence_timestamp_proto);
  if (!first_arc_measurement.has_value() ||
      first_arc_measurement.value() > first_sequence_timestamp) {
    first_arc_measurement = first_sequence_timestamp;
  }
  return first_arc_measurement;
}

// Returns true if data has been added to the feature list.
absl::StatusOr<bool> AddSequence(
    bool flatten,
    const AnnotatedRecordingCollection& annotated_recording_collection,
    google::protobuf::Map<std::string, tensorflow::Feature>* feature_map,
    google::protobuf::Map<std::string, tensorflow::FeatureList>* feature_list
) {
  bool added_data = false;
  absl::optional<absl::Duration> first_arc_measurement;
  for (const auto& sequence :
       annotated_recording_collection.recording_collection().sequence()) {
    std::string feature_name = absl::StrJoin(
        {sequence.metadata().type(), sequence.metadata().subtype()},
        kDelimiter);

    if (sequence.repeated_datapoint().datapoint().empty() &&
        sequence.repeated_window().window().empty()) {
      continue;
    }

    // We add std::string around strings because the open source build fails
    // without it.
    (*feature_map)[absl::StrJoin({feature_name, std::string("dimensionality")},
                                 kDelimiter)]
        .mutable_int64_list()
        ->add_value(sequence.metadata().measurement_dimensionality());

    // Copy the feature data over.
    tensorflow::FeatureList& feature_container = (*feature_list)[absl::StrJoin(
        {feature_name, std::string("floats")}, kDelimiter)];

    switch (sequence.datapoints_or_windows_case()) {
      case Sequence::kRepeatedDatapoint: {
        const RepeatedDatapoint& datapoints = sequence.repeated_datapoint();
        const auto& first_datapoint = datapoints.datapoint(0);
        if (first_datapoint.has_offset()) {
          first_arc_measurement = UpdateFirstArcMeasurement(
              first_datapoint.offset(), first_arc_measurement);
        }
        CopyDataToFeatureContainer(datapoints.datapoint(), flatten,
                                   &feature_container);
        break;
      }
      case Sequence::kRepeatedWindow: {
        // Add window size feature.
        (*feature_map)[absl::StrJoin({feature_name, std::string("window_size")},
                                     kDelimiter)]
            .mutable_int64_list()
            ->add_value(sequence.metadata().window_size());

        const RepeatedWindow& windows = sequence.repeated_window();
        const auto& first_window = windows.window(0);
        if (first_window.has_offset()) {
          first_arc_measurement = UpdateFirstArcMeasurement(
              first_window.offset(), first_arc_measurement);
        }
        for (const auto& window : windows.window()) {
          CopyDataToFeatureContainer(window.datapoint(), flatten,
                                     &feature_container);
        }
        break;
      }
      case Sequence::DATAPOINTS_OR_WINDOWS_NOT_SET:
        return absl::InvalidArgumentError(absl::Substitute(
            "Malformed Sequence data for key $0.",
            annotated_recording_collection.recording_collection()
                .metadata()
                .session_id()));
    }
    added_data = true;
  }
  if (first_arc_measurement.has_value()) {
    (*feature_map)["first_measurement/timestamp/seconds"]
        .mutable_float_list()
        ->add_value(absl::ToDoubleSeconds(first_arc_measurement.value()));
  }
  return added_data;
}

}  // namespace

absl::StatusOr<SequenceExample>
ConvertAnnotatedRecordingCollectionToSequenceExample(
    bool flatten, const std::string& resource_name,
    const AnnotatedRecordingCollection& input
) {
  SequenceExample sensor_features_base;
  // Set the ID from the sensor data.
  using TfFeatureMap = google::protobuf::Map<std::string, tensorflow::Feature>;
  TfFeatureMap* feat_map_base =
      sensor_features_base.mutable_context()->mutable_feature();
  (*feat_map_base)["session_id"].mutable_bytes_list()->add_value(resource_name);

  // Add groundtruth labels.
  const std::string& activity_name = input.recording_collection()
                                         .metadata()
                                         .mobile_collection_metadata()
                                         .session_activity();

  for (const auto& annotation_group : input.annotation_group()) {
    if (IsGroundTruthGroup(annotation_group)) {
      for (const auto& annotation_sequence :
           annotation_group.annotation_sequence()) {
        if (IsClassLabelSequence(annotation_sequence)) {
          const std::string& mapping_name = annotation_sequence.metadata()
                                                .source_details()
                                                .identifier()
                                                .name();
          (*feat_map_base)[absl::StrJoin({mapping_name, std::string("label")},
                                         kDelimiter)]
              .mutable_int64_list()
              ->add_value(annotation_sequence.annotation(0).label(0).id());
          (*feat_map_base)[absl::StrJoin(
                               {mapping_name, std::string("label/name")},
                               kDelimiter)]
              .mutable_bytes_list()
              ->add_value(annotation_sequence.annotation(0).label(0).name());
        } else if (annotation_sequence.metadata().annotation_type() ==
                   AnnotationMetadata::DERIVED) {
          const std::string& sequence_name = annotation_sequence.metadata()
                                                 .source_details()
                                                 .identifier()
                                                 .name();
          for (const Annotation& annotation :
               annotation_sequence.annotation()) {
            for (const AnnotationLabel& label : annotation.label()) {
              // Since for some derived annotation labels it might be better to
              // access the mappings by name, e.g., the home or away team in
              // ShotTracker data and for others the id might be more useful,
              // e.g. to find the jersey number to a given player ID in
              // ShotTracker data, we provide both directions of label mappings.
              (*feat_map_base)[absl::StrJoin({sequence_name, label.name()},
                                             kDelimiter)]
                  .mutable_int64_list()
                  ->add_value(label.id());
              (*feat_map_base)[absl::StrJoin(
                                   {sequence_name, std::to_string(label.id())},
                                   kDelimiter)]
                  .mutable_bytes_list()
                  ->add_value(label.name());
            }
          }
        }
      }
    }
  }

  (*feat_map_base)["activity/label"].mutable_bytes_list()->add_value(
      activity_name);
  SequenceExample sensor_features(sensor_features_base);
  TfFeatureMap* feat_map = sensor_features.mutable_context()->mutable_feature();

  absl::StatusOr<bool> added_data = AddSequence(
      flatten, input, feat_map,
      sensor_features.mutable_feature_lists()->mutable_feature_list()
  );
  if (!added_data.ok()) return added_data.status();
  if (!(*added_data)) {
    return absl::UnavailableError("Unabled to add sequence data");
  }
  return sensor_features;
}

}  // namespace ambient_sensing
