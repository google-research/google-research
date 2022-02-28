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

// Calculator that takes 2 AnnotatedRecordingCollection messages: <original> and
// <processed>, and augments <processed> with same details from <original>.
// The motivation to have this calculator is that some metadata and various bits
// of information are lost in the Drishti graph.
// In particular, these details are restored:
// - From recording_collection.metadata():
//     session_id(),
//     mobile_collection_metadata(),
//     base_timestamp()
// - annotation_group()
//
// It assumes that the data that is being copied over from <original> to
// <processed> is not already present in <processed>.
//
// Note that the <original> message can be provided as stream or side packet,
// while <processed> message is received from an input stream.
//
// TODO(b/149385281): this can be easily extended to handle a stream of
// processed messages. Right now, it checks that exactly 1 processed message is
// given as an input.

#include <cstdint>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_sensor_options.pb.h"
#include "intent_recognition/processing/set_annotated_recording_collection_metadata_calculator.pb.h"

namespace ambient_sensing {
namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kOriginalAnnotatedRecordingCollectionTag[] =
    "ORIGINAL_ANNOTATED_RECORDING_COLLECTION";
constexpr char kProcessedAnnotatedRecordingCollectionTag[] =
    "PROCESSED_ANNOTATED_RECORDING_COLLECTION";
constexpr char kOutputDataStreamTag[] = "OUTPUT_ANNOTATED_RECORDING_COLLECTION";
constexpr char kUnknown[] = "UNKNOWN";
constexpr int64_t kNumMicrosPerMilli = 1000LL;

}  // namespace

class SetAnnotatedRecordingCollectionMetadataCalculator
    : public mediapipe::CalculatorBase {
 public:
  SetAnnotatedRecordingCollectionMetadataCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    bool inputs_has_tag =
        cc->Inputs().HasTag(kOriginalAnnotatedRecordingCollectionTag);
    bool input_side_packets_has_tag =
        cc->InputSidePackets().HasTag(kOriginalAnnotatedRecordingCollectionTag);
    if (inputs_has_tag && input_side_packets_has_tag) {
      return absl::InvalidArgumentError(absl::StrCat(
          kOriginalAnnotatedRecordingCollectionTag,
          " can't simultaneously be input stream and input side packet."));
    }
    if (inputs_has_tag) {
      cc->Inputs()
          .Tag(kOriginalAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    if (input_side_packets_has_tag) {
      cc->InputSidePackets()
          .Tag(kOriginalAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    cc->Inputs()
        .Tag(kProcessedAnnotatedRecordingCollectionTag)
        .Set<AnnotatedRecordingCollection>();
    cc->Outputs().Tag(kOutputDataStreamTag).Set<AnnotatedRecordingCollection>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options().GetExtension(
        SetAnnotatedRecordingCollectionMetadataCalculatorOptions::ext);
    for (const auto& p : options_.original_to_synthesized_stream_mapping()) {
      const auto& original_stream = p.original_stream();
      const auto& synthesized_stream = p.synthesized_stream();

      decltype(original_stream_by_synthesized_stream_)::value_type pair = {
          {synthesized_stream.type(), synthesized_stream.subtype()},
          {original_stream.type(), original_stream.subtype()}};
      original_stream_by_synthesized_stream_.insert(
          // Add a mapping synthesized_stream_sensor --> original_stream_sensor.
          pair);
      // Also insert this pair into set for suppressing remapping conflicts.
      if (p.suppress_conflict_counters()) {
        suppress_remapping_conflict_counters_.insert({pair.second, pair.first});
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (process_called_) {
      return absl::InvalidArgumentError(
          "SetAnnotatedRecordingCollectionCalculator expects exactly 1 input "
          "element in the stream.");
    }

    AnnotatedRecordingCollection original_annotated_recording_collection;
    if (cc->Inputs().Tag(kProcessedAnnotatedRecordingCollectionTag).IsEmpty()) {
      cc->GetCounter("No processed data")->Increment();
      return absl::OkStatus();
    }
    if (cc->Inputs().HasTag(kOriginalAnnotatedRecordingCollectionTag)) {
      original_annotated_recording_collection =
          cc->Inputs()
              .Tag(kOriginalAnnotatedRecordingCollectionTag)
              .Get<AnnotatedRecordingCollection>();
    } else {
      original_annotated_recording_collection =
          cc->InputSidePackets()
              .Tag(kOriginalAnnotatedRecordingCollectionTag)
              .Get<AnnotatedRecordingCollection>();
    }
    auto result = absl::make_unique<AnnotatedRecordingCollection>(
        cc->Inputs()
            .Tag(kProcessedAnnotatedRecordingCollectionTag)
            .Get<AnnotatedRecordingCollection>());

    process_called_ = true;

    absl::Status status =
        BuildSequencesSensorSet(original_annotated_recording_collection);
    if (status != absl::OkStatus()) return status;

    // Copy the metadata.
    if (result->recording_collection().has_metadata() ||
        !result->annotation_group().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Processed message already has metadata or annotation: ",
          result->recording_collection().metadata().session_id(),
          cc->InputTimestamp().Microseconds() / kNumMicrosPerMilli));
    }

    if (original_annotated_recording_collection.recording_collection()
            .metadata()
            .has_mobile_collection_metadata()) {
      *result->mutable_recording_collection()
           ->mutable_metadata()
           ->mutable_mobile_collection_metadata() =
          original_annotated_recording_collection.recording_collection()
              .metadata()
              .mobile_collection_metadata();
    }

    const auto& metadata =
        original_annotated_recording_collection.recording_collection()
            .metadata();
    auto* result_metadata =
        result->mutable_recording_collection()->mutable_metadata();
    if (!metadata.session_id().empty()) {
      result_metadata->set_session_id(metadata.session_id());
    }
    if (metadata.has_base_timestamp()) {
      *result_metadata->mutable_base_timestamp() = metadata.base_timestamp();
    }

    *result->mutable_annotation_group() =
        original_annotated_recording_collection.annotation_group();

    if (options_.recover_recording_sequences_not_present()) {
      using SequenceType = std::pair<std::string, std::string>;
      absl::flat_hash_set<SequenceType> sequences_in_processed;
      for (const auto& sequence : result->recording_collection().sequence()) {
        sequences_in_processed.insert(
            {sequence.metadata().type(), sequence.metadata().subtype()});
      }

      for (const auto& sequence :
           original_annotated_recording_collection.recording_collection()
               .sequence()) {
        SequenceType sequence_type = {sequence.metadata().type(),
                                      sequence.metadata().subtype()};
        if (!sequences_in_processed.contains(sequence_type)) {
          *result->mutable_recording_collection()->add_sequence() = sequence;
        }
      }
    }

    cc->Outputs()
        .Tag(kOutputDataStreamTag)
        .Add(result.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  absl::Status BuildSequencesSensorSet(
      const AnnotatedRecordingCollection& annotated_recording_collection) {
    for (const auto& sequence :
         annotated_recording_collection.recording_collection().sequence()) {
      // Sensors that have not been converted properly or are not relevant.
      if (sequence.metadata().subtype() == kUnknown) {
        continue;
      }
      auto insert = sequence_sensor_types_set_.insert(
          {sequence.metadata().type(), sequence.metadata().subtype()});
      if (!insert.second) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Duplicate sequence of subtype %s (session id: %s)",
            sequence.metadata().subtype(),
            annotated_recording_collection.recording_collection()
                .metadata()
                .session_id()));
      }
    }
    return absl::OkStatus();
  }

  SetAnnotatedRecordingCollectionMetadataCalculatorOptions options_;

  // Key: sensor type and subtype.
  absl::flat_hash_set<std::pair<std::string, std::string>>
      sequence_sensor_types_set_;
  absl::flat_hash_map<std::pair<std::string, std::string>,
                      std::pair<std::string, std::string>>
      original_stream_by_synthesized_stream_;

  // Mapping from <sensor type, subtype> to <sensor type, subtype>.
  // If a pair is present in this set, then the counter for conflicting
  // timestamps remapping is not incremented.
  absl::flat_hash_set<
      decltype(original_stream_by_synthesized_stream_)::value_type>
      suppress_remapping_conflict_counters_;

  bool process_called_ = false;
};

REGISTER_CALCULATOR(SetAnnotatedRecordingCollectionMetadataCalculator);

}  // namespace ambient_sensing
