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

#include "intent_recognition/annotated_recording_collection_utils.h"

#include <algorithm>
#include <cstdint>

#include "google/protobuf/duration.pb.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "mediapipe/framework/tool/template_expander.h"
#include "mediapipe/framework/tool/template_parser.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "intent_recognition/annotated_recording_collection.pb.h"

namespace ambient_sensing {

namespace {

bool HasAnnotationGroupType(const AnnotationGroup& group,
                            AnnotationGroupMetadata::AnnotationGroupType type) {
  return group.metadata().group_type() == type;
}

bool HasAnnotationType(const AnnotationSequence& sequence,
                       AnnotationMetadata::AnnotationType type) {
  return sequence.metadata().annotation_type() == type;
}
}  // namespace

absl::Duration ConvertProtoToDuration(
    const google::protobuf::Duration& duration_proto) {
  return absl::Seconds(duration_proto.seconds()) +
         absl::Nanoseconds(duration_proto.nanos());
}

google::protobuf::Duration ConvertDurationToProto(
    const absl::Duration duration) {
  constexpr int64_t kNanosInSeconds = 1000 * 1000 * 1000;
  int64_t total_time_nanos = absl::ToInt64Nanoseconds(duration);
  google::protobuf::Duration duration_as_proto;
  duration_as_proto.set_seconds(total_time_nanos / kNanosInSeconds);
  duration_as_proto.set_nanos(total_time_nanos % kNanosInSeconds);
  return duration_as_proto;
}

bool IsGroundTruthGroup(const AnnotationGroup& group) {
  return HasAnnotationGroupType(group, AnnotationGroupMetadata::GROUND_TRUTH);
}

bool IsStringLabelMappingGroup(const AnnotationGroup& group) {
  return group.metadata().group_descriptor() ==
         kStringLabelMappingGroupDescriptor;
}

bool IsModelGroup(const AnnotationGroup& group) {
  return HasAnnotationGroupType(group, AnnotationGroupMetadata::MODEL);
}

bool IsTaskDurationSequence(const AnnotationSequence& sequence) {
  return HasAnnotationType(sequence, AnnotationMetadata::TASK_DURATION);
}

bool IsTagSequence(const AnnotationSequence& sequence) {
  return HasAnnotationType(sequence, AnnotationMetadata::TAG);
}

bool IsClassLabelSequence(const AnnotationSequence& sequence) {
  return HasAnnotationType(sequence, AnnotationMetadata::CLASS_LABEL);
}

bool IsStagedModelSequence(const AnnotationSequence& sequence) {
  return HasAnnotationType(sequence,
                           AnnotationMetadata::STAGED_MODEL_GENERATED);
}

bool IsHumanLabelSequence(const AnnotationSequence& sequence) {
  return HasAnnotationType(sequence, AnnotationMetadata::HUMAN_LABEL);
}

void AddStringArg(const std::string& key, const std::string& value,
                  mediapipe::TemplateDict* arguments) {
  auto arg = arguments->add_arg();
  arg->set_key(key);
  arg->mutable_value()->set_str(value);
}

void AddNumArg(const std::string& key, double value,
               mediapipe::TemplateDict* arguments) {
  auto arg = arguments->add_arg();
  arg->set_key(key);

  arg->mutable_value()->set_num(value);
}

template <typename T>
void AddStringElements(const std::string& key, T values,
                       mediapipe::TemplateDict* arguments) {
  auto arg = arguments->add_arg();
  arg->set_key(key);
  auto value = arg->mutable_value();
  for (const auto& v : values) {
    mediapipe::TemplateArgument* value_arg = value->add_element();
    value_arg->set_str(v);
  }
}

void PopulateProcessingOptions(const ProcessingOptions& processing_options,
                               mediapipe::TemplateDict* arguments) {
  AddStringElements("label_mapping_files",
                    processing_options.label_mapping_files(), arguments);
  AddStringElements("exclude_user_ids", processing_options.exclude_user_id(),
                    arguments);
  AddStringElements("exclude_session_activities",
                    processing_options.exclude_session_activity(), arguments);
  AddStringElements("include_session_activities",
                    processing_options.include_session_activity(), arguments);
  AddStringElements("at_least_one_annotation_with_substring",
                    processing_options.at_least_one_annotation_with_substring(),
                    arguments);
  AddStringElements("no_annotations_with_substring",
                    processing_options.no_annotations_with_substring(),
                    arguments);
  AddStringArg("filter_label_mapping_name",
               processing_options.filter_label_mapping_name(), arguments);
  AddStringElements("exclude_class_names",
                    processing_options.exclude_class_name(), arguments);
  AddStringElements("include_class_names",
                    processing_options.include_class_name(), arguments);
  AddNumArg("sampling_rate", processing_options.sampling_rate(), arguments);
  AddNumArg("window_size", processing_options.window_size(), arguments);
  AddNumArg("window_stride", processing_options.window_stride(), arguments);
  AddNumArg("window_padding_strategy",
            static_cast<int>(processing_options.padding_strategy()), arguments);
  AddNumArg("minimum_windows", processing_options.minimum_windows(), arguments);
}

mediapipe::CalculatorGraphConfig BuildDrishtiGraphWithProcessingOptions(
    const std::string& config_template,
    const ProcessingOptions& processing_options) {
  mediapipe::CalculatorGraphTemplate graph_template;
  mediapipe::tool::TemplateParser::Parser parser;
  CHECK(parser.ParseFromString(config_template, &graph_template));

  mediapipe::TemplateDict arguments;
  PopulateProcessingOptions(processing_options, &arguments);

  mediapipe::tool::TemplateExpander expander;
  mediapipe::CalculatorGraphConfig graph_config;
  CHECK_OK(expander.ExpandTemplates(arguments, graph_template, &graph_config));

  return graph_config;
}

}  // namespace ambient_sensing
