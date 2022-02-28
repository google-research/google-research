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

}  // namespace ambient_sensing
