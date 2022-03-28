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

#ifndef INTENT_RECOGNITION_ANNOTATED_RECORDING_COLLECTION_UTILS_H_
#define INTENT_RECOGNITION_ANNOTATED_RECORDING_COLLECTION_UTILS_H_

#include <string>

#include "google/protobuf/duration.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/processing/processing_options.pb.h"

namespace ambient_sensing {

inline constexpr absl::string_view kStringLabelMappingGroupDescriptor =
    "string_label_mapping";

// Converts a google::protobuf::Duration into absl::Duration and checks
// whether the conversion was successful.
absl::Duration ConvertProtoToDuration(
    const google::protobuf::Duration& duration_proto);

// Converts a absl::Duration into google::protobuf::Duration and checks
// whether the conversion was successful.
google::protobuf::Duration ConvertDurationToProto(
    const absl::Duration duration);

// Returns true if AnnotationGroup type is GROUND_TRUTH.
bool IsGroundTruthGroup(const AnnotationGroup& group);

// Returns true if AnnotationGroup description is
// kStringLabelMappingGroupDescriptor.
bool IsStringLabelMappingGroup(const AnnotationGroup& group);

// Returns true if AnnotationGroup type is MODEL.
bool IsModelGroup(const AnnotationGroup& group);

// Returns true if AnnotationSequence type is TASK_DURATION.
bool IsTaskDurationSequence(const AnnotationSequence& sequence);

// Returns true if AnnotationSequence type is TAG.
bool IsTagSequence(const AnnotationSequence& sequence);

// Returns true if AnnotationSequence type is CLASS_LABEL.
bool IsClassLabelSequence(const AnnotationSequence& sequence);

// Returns true if AnnotationSequence type is STAGED_MODEL_GENERATED.
bool IsStagedModelSequence(const AnnotationSequence& sequence);

// Returns true if AnnotationSequence type is HUMAN_LABEL.
bool IsHumanLabelSequence(const AnnotationSequence& sequence);


// Given a Drishti graph template, substitutes the provided parameters and
// returns a valid CalculatorGraphConfig. If any error occurs (e.g. the config
// couldn't be parsed), crashes the binary.
mediapipe::CalculatorGraphConfig BuildDrishtiGraphWithProcessingOptions(
    const std::string& config_template,
    const ProcessingOptions& processing_options);

}  // namespace ambient_sensing

#endif  // INTENT_RECOGNITION_ANNOTATED_RECORDING_COLLECTION_UTILS_H_
