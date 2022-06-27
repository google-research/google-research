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

// Calculator that consumes a stream of AnnotatedRecordinCollection messages and
// only outputs those that pass the filtering criteria (see options for which
// criteria are supported). For those messages that pass the filter, the output
// timestamp is the same as the input timestamp.
// Alternatively, it can consume input as an input side packet, in that case
// the output timestamp is PreStream().

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "glog/logging.h"
#include "google/protobuf/repeated_field.h"
#include "mediapipe/framework/calculator_framework.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/processing/filter_annotated_recording_collection_calculator.pb.h"

namespace ambient_sensing {
namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kInputAnnotatedRecordingCollectionTag[] =
    "INPUT_ANNOTATED_RECORDING_COLLECTION";
constexpr char kFilteredAnnotatedRecordingCollectionTag[] =
    "FILTERED_ANNOTATED_RECORDING_COLLECTION";

}  // namespace

class FilterAnnotatedRecordingCollectionCalculator
    : public mediapipe::CalculatorBase {
 public:
  FilterAnnotatedRecordingCollectionCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    bool inputs_has_tag =
        cc->Inputs().HasTag(kInputAnnotatedRecordingCollectionTag);
    bool input_side_packets_has_tag =
        cc->InputSidePackets().HasTag(kInputAnnotatedRecordingCollectionTag);
    if (inputs_has_tag && input_side_packets_has_tag) {
      return absl::InvalidArgumentError(
          "Input stream and input side packet can't be used simultaneously.");
    }
    if (!inputs_has_tag && !input_side_packets_has_tag) {
      return absl::InvalidArgumentError(
          "Input stream or input side packet must be specified.");
    }
    if (inputs_has_tag) {
      cc->Inputs()
          .Tag(kInputAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    if (input_side_packets_has_tag) {
      cc->InputSidePackets()
          .Tag(kInputAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    cc->Outputs()
        .Tag(kFilteredAnnotatedRecordingCollectionTag)
        .Set<AnnotatedRecordingCollection>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options().GetExtension(
        FilterAnnotatedRecordingCollectionCalculatorOptions::ext);
    if (!options_.exclude_session_activity().empty() &&
        !options_.include_session_activity().empty()) {
      return absl::InvalidArgumentError(
          "Only one of black_session_activity and "
          "include_session_activity may be specified.");
    }
    exclude_user_ids_ = absl::flat_hash_set<std::string>(
        options_.exclude_user_id().begin(), options_.exclude_user_id().end());
    for (const std::string& subtype :
         options_.filter_if_sensor_not_present_or_empty()) {
      filter_if_sensor_not_present_or_empty_.insert(subtype);
    }

    if (!options_.filter_label_mapping_name().empty()) {
      // Filter using new label mappings.
      evaluate_class_name_ = true;
      if (!options_.exclude_class_name().empty() &&
          !options_.include_class_name().empty()) {
        return absl::InvalidArgumentError(
            "Only one of exclude_class and include_class may be specified.");
      }
    }
    trace_length_less_or_equal_ =
        absl::Milliseconds(options_.trace_length_less_or_equal_millis());
    trace_length_greater_or_equal_ =
        absl::Milliseconds(options_.trace_length_greater_or_equal_millis());

    if (cc->InputSidePackets().HasTag(kInputAnnotatedRecordingCollectionTag)) {
      auto input = cc->InputSidePackets()
                       .Tag(kInputAnnotatedRecordingCollectionTag)
                       .Get<AnnotatedRecordingCollection>();
      return ProcessAnnotatedRecordingCollection(
          cc, input, mediapipe::Timestamp::PreStream());
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (!cc->Inputs().HasTag(kInputAnnotatedRecordingCollectionTag))
      return mediapipe::tool::StatusStop();
    const auto& input = cc->Inputs()
                            .Tag(kInputAnnotatedRecordingCollectionTag)
                            .Get<AnnotatedRecordingCollection>();
    return ProcessAnnotatedRecordingCollection(cc, input, cc->InputTimestamp());
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  bool evaluate_class_name_ = false;
  absl::Duration trace_length_less_or_equal_;
  absl::Duration trace_length_greater_or_equal_;
  FilterAnnotatedRecordingCollectionCalculatorOptions options_;
  absl::flat_hash_set<std::string> exclude_user_ids_;
  absl::flat_hash_set<std::string> filter_if_sensor_not_present_or_empty_;

  bool HasAnySubstring(const AnnotationLabel& label,
                       const google::protobuf::RepeatedPtrField<std::string>& substrings,
                       CalculatorContext* cc) {
    for (const auto& substring : substrings) {
      if (absl::StrContains(label.name(), substring)) {
        cc->GetCounter(
              absl::StrCat("has-label-with-substring-[", substring, "]"))
            ->Increment();
        return true;
      }
    }
    return false;
  }

  bool HasAnnotationWithAnySubstring(
      const AnnotationSequence& sequence,
      const google::protobuf::RepeatedPtrField<std::string>& substrings,
      CalculatorContext* cc) {
    for (const auto& annotation : sequence.annotation()) {
      for (const auto& label : annotation.label()) {
        if (HasAnySubstring(label, substrings, cc)) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasSequenceWithAnySubstring(
      const AnnotationGroup& group,
      const google::protobuf::RepeatedPtrField<std::string>& substrings,
      CalculatorContext* cc) {
    for (const auto& sequence : group.annotation_sequence()) {
      if (!IsTaskDurationSequence(sequence)) {
        continue;
      }
      if (HasAnnotationWithAnySubstring(sequence, substrings, cc)) {
        return true;
      }
    }
    return false;
  }

  bool EvaluateAtLeastOneAnnotationWithSubstringCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    if (options_.at_least_one_annotation_with_substring().empty()) {
      return true;
    }
    for (const auto& group :
         annotated_recording_collection.annotation_group()) {
      if (!IsGroundTruthGroup(group)) {
        continue;
      }
      if (HasSequenceWithAnySubstring(
              group, options_.at_least_one_annotation_with_substring(), cc)) {
        return true;
      }
    }
    return false;
  }

  absl::Duration GetTraceDuration(
      const AnnotatedRecordingCollection& annotated_recording_collection) {
    absl::optional<absl::Duration> min_timestamp, max_timestamp;
    for (const auto& sequence :
         annotated_recording_collection.recording_collection().sequence()) {
      for (const auto& window : sequence.repeated_window().window()) {
        absl::Duration offset = ConvertProtoToDuration(window.offset());
        min_timestamp = std::min(min_timestamp.value_or(offset), offset);
        max_timestamp = std::max(max_timestamp.value_or(offset), offset);
      }
      for (const auto& d : sequence.repeated_datapoint().datapoint()) {
        absl::Duration offset = ConvertProtoToDuration(d.offset());
        min_timestamp = std::min(min_timestamp.value_or(offset), offset);
        max_timestamp = std::max(max_timestamp.value_or(offset), offset);
      }
    }
    absl::Duration result = max_timestamp.value_or(absl::ZeroDuration()) -
                            min_timestamp.value_or(absl::ZeroDuration());

    DLOG(INFO) << absl::StrCat("trace-duration-", absl::FormatDuration(result));
    return result;
  }

  bool TraceLengthLowerBoundUndefined() {
    return trace_length_greater_or_equal_ <= absl::ZeroDuration();
  }

  bool TraceLengthUpperBoundUndefined() {
    return trace_length_less_or_equal_ <= absl::ZeroDuration();
  }

  bool EvaluateTraceLengthLessOrEqualCondition(absl::Duration duration) {
    if (TraceLengthUpperBoundUndefined()) {
      return true;
    }
    return duration <= trace_length_less_or_equal_;
  }

  bool EvaluateTraceLengthGreaterOrEqualCondition(absl::Duration duration) {
    return duration >= trace_length_greater_or_equal_;
  }

  bool EvaluateTraceLength(
      const AnnotatedRecordingCollection& annotated_recording_collection) {
    if (TraceLengthLowerBoundUndefined() && TraceLengthUpperBoundUndefined()) {
      return true;
    }
    absl::Duration duration = GetTraceDuration(annotated_recording_collection);
    return EvaluateTraceLengthLessOrEqualCondition(duration) &&
           EvaluateTraceLengthGreaterOrEqualCondition(duration);
  }

  bool EvaluateNoAnnotationsWithSubstringCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    for (const auto& group :
         annotated_recording_collection.annotation_group()) {
      if (!IsGroundTruthGroup(group)) {
        continue;
      }
      if (HasSequenceWithAnySubstring(
              group, options_.no_annotations_with_substring(), cc)) {
        return false;
      }
    }
    return true;
  }

  bool EvaluateExcludeUserIdCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    const auto& user_id = annotated_recording_collection.recording_collection()
                              .metadata()
                              .user_id();
    bool excluded = exclude_user_ids_.contains(user_id);
    if (!excluded) {
      cc->GetCounter("user-id-not-excluded")->Increment();
    } else {
      cc->GetCounter("user-id-excluded")->Increment();
      cc->GetCounter(absl::StrCat("user-id-excluded-", user_id))->Increment();
    }
    return !excluded;
  }

  bool EvaluateExcludeSessionActivityCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    const auto& session_activity =
        annotated_recording_collection.recording_collection()
            .metadata()
            .mobile_collection_metadata()
            .session_activity();
    bool excluded =
        absl::c_find(options_.exclude_session_activity(), session_activity) !=
        options_.exclude_session_activity().end();
    if (!excluded) {
      cc->GetCounter("session-activity-not-excluded")->Increment();
    } else {
      cc->GetCounter("session-activity-excluded")->Increment();
      cc->GetCounter(
            absl::StrCat("session-activity-excluded-", session_activity))
          ->Increment();
    }
    return !excluded;
  }

  bool EvaluateIncludeSessionActivityCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    if (options_.include_session_activity().empty()) {
      return true;
    }
    const auto& session_activity =
        annotated_recording_collection.recording_collection()
            .metadata()
            .mobile_collection_metadata()
            .session_activity();
    bool included =
        absl::c_find(options_.include_session_activity(), session_activity) !=
        options_.include_session_activity().end();
    std::string counter_name = included ? "session-activity-included"
                                        : "session-activity-not-included";
    cc->GetCounter(counter_name)->Increment();
    cc->GetCounter(absl::StrCat(counter_name, "-", session_activity))
        ->Increment();
    return included;
  }

  bool EvaluateHasNonEmptySequenceCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    absl::flat_hash_set<std::string> present_subtypes;
    for (const auto& sequence :
         annotated_recording_collection.recording_collection().sequence()) {
      if ((sequence.metadata().type() == "SENSOR") &&
          (!sequence.repeated_datapoint().datapoint().empty() ||
           !sequence.repeated_window().window().empty())) {
        present_subtypes.insert(sequence.metadata().subtype());
      }
    }
    bool result = true;
    for (const std::string& expected_subtype :
         filter_if_sensor_not_present_or_empty_) {
      if (!present_subtypes.contains(expected_subtype)) {
        cc->GetCounter(
              absl::StrCat("non-empty-sequence-not-present-", expected_subtype))
            ->Increment();
        result = false;
      }
    }
    if (!result) {
      cc->GetCounter("non-empty-sequence-not-present")->Increment();
    }
    return result;
  }

  bool LabelInClassList(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      const absl::flat_hash_map<std::string, int>& class_name_to_id_mapping,
      const google::protobuf::RepeatedField<int32_t>& class_list, CalculatorContext* cc) {
    absl::string_view label =
        annotated_recording_collection.recording_collection()
            .metadata()
            .mobile_collection_metadata()
            .session_activity();
    CHECK(class_name_to_id_mapping.contains(label))
        << "Class label " << label << " is missing in label map.";
    const int id = class_name_to_id_mapping.at(label);
    bool label_in_list = absl::c_find(class_list, id) != class_list.end();
    if (label_in_list) {
      cc->GetCounter(absl::Substitute("class-id-$0-found", id))->Increment();
      return true;
    }
    cc->GetCounter(absl::Substitute("class-id-$0-not-found", id))->Increment();
    return false;
  }

  // use google::protobuf::RepeatedPtrField or proto_ns::RepeatedPtrField)
  bool LabelInClassNameList(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      const google::protobuf::RepeatedPtrField<std::string>& class_list,
      CalculatorContext* cc) {
    for (const auto& annotation_group :
         annotated_recording_collection.annotation_group()) {
      if (IsGroundTruthGroup(annotation_group)) {
        for (const auto& annotation_sequence :
             annotation_group.annotation_sequence()) {
          if (IsClassLabelSequence(annotation_sequence)) {
            std::string mapping_name = annotation_sequence.metadata()
                                           .source_details()
                                           .identifier()
                                           .name();
            if (mapping_name == options_.filter_label_mapping_name()) {
              std::string class_name =
                  annotation_sequence.annotation(0).label(0).name();
              if (absl::c_find(class_list, class_name) != class_list.end()) {
                cc->GetCounter(
                      absl::Substitute("class-name-$0-found", class_name))
                    ->Increment();
                return true;
              } else {
                cc->GetCounter(
                      absl::Substitute("class-name-$0-not-found", class_name))
                    ->Increment();
                return false;
              }
            }
          }
        }
      }
    }
    return false;
  }

  bool EvaluateExcludeClassNameCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    if (!evaluate_class_name_ || options_.exclude_class_name().empty()) {
      return true;
    }
    return !LabelInClassNameList(annotated_recording_collection,
                                 options_.exclude_class_name(), cc);
  }

  bool EvaluateIncludeClassNameCondition(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    if (!evaluate_class_name_ || options_.include_class_name().empty()) {
      return true;
    }
    return LabelInClassNameList(annotated_recording_collection,
                                options_.include_class_name(), cc);
  }

  absl::Status ProcessAnnotatedRecordingCollection(
      CalculatorContext* cc, const AnnotatedRecordingCollection& input,
      const mediapipe::Timestamp& ts) {
    // Evaluate all conditions (don't rely on lazy evaluation) to make
    // counters more useful.
    bool c1 = EvaluateAtLeastOneAnnotationWithSubstringCondition(input, cc),
         c2 = EvaluateTraceLength(input),
         c3 = EvaluateNoAnnotationsWithSubstringCondition(input, cc),
         c4 = EvaluateExcludeSessionActivityCondition(input, cc),
         c5 = EvaluateIncludeSessionActivityCondition(input, cc),
         c6 = EvaluateHasNonEmptySequenceCondition(input, cc),
         c7 = EvaluateExcludeClassNameCondition(input, cc),
         c8 = EvaluateIncludeClassNameCondition(input, cc),
         c9 = EvaluateExcludeUserIdCondition(input, cc);

    if (c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {
      auto result = absl::make_unique<AnnotatedRecordingCollection>(input);
      cc->Outputs()
          .Tag(kFilteredAnnotatedRecordingCollectionTag)
          .Add(result.release(), ts);
    } else {
      cc->GetCounter("RemovedArc")->Increment();
    }
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(FilterAnnotatedRecordingCollectionCalculator);

}  // namespace ambient_sensing
