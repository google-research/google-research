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

#include "mediapipe/framework/calculator_framework.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/processing/class_mappings_provider.h"

namespace ambient_sensing {

namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kInputAnnotatedRecordingCollectionTag[] =
    "INPUT_ANNOTATED_RECORDING_COLLECTION";
constexpr char kClassMappingsProvider[] = "CLASS_MAPPINGS_PROVIDER";
constexpr char kOutputAnnotatedRecordingCollectionTag[] =
    "OUTPUT_ANNOTATED_RECORDING_COLLECTION";
}  // namespace

// Calculator that adds label mappings to AnnotatedRecordingCollection samples.
// Relies on the class mapping packet factory for receiving a class mapping
// provider.
class AddClassMappingsCalculator : public mediapipe::CalculatorBase {
 public:
  AddClassMappingsCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs()
        .Tag(kInputAnnotatedRecordingCollectionTag)
        .Set<AnnotatedRecordingCollection>();
    cc->Outputs()
        .Tag(kOutputAnnotatedRecordingCollectionTag)
        .Set<AnnotatedRecordingCollection>();
    cc->InputSidePackets()
        .Tag(kClassMappingsProvider)
        .Set<std::unique_ptr<ClassMappingsProvider>>();
    if (cc->Inputs().NumEntries() != cc->Outputs().NumEntries()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of input streams must match number of output streams: ",
          cc->Inputs().NumEntries(), " != ", cc->Outputs().NumEntries()));
    }
    if (cc->InputSidePackets().NumEntries() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Must receive exactly one input side packet: ",
                       cc->InputSidePackets().NumEntries(), " != 1"));
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    if (cc->InputSidePackets().HasTag(kClassMappingsProvider)) {
      label_mapping_provider_ =
          cc->InputSidePackets()
              .Tag(kClassMappingsProvider)
              .Get<std::unique_ptr<ClassMappingsProvider>>()
              .get();
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto result_arc = absl::make_unique<AnnotatedRecordingCollection>(
        cc->Inputs()
            .Tag(kInputAnnotatedRecordingCollectionTag)
            .Get<AnnotatedRecordingCollection>());

    // Add prediction class + ID to sample for all mappings.
    absl::Status status =
        label_mapping_provider_->AddClassMappings(result_arc.get());
    if (status != absl::OkStatus()) return status;

    for (const auto& annotation_group : result_arc->annotation_group()) {
      if (annotation_group.metadata().group_type() ==
          AnnotationGroupMetadata::GROUND_TRUTH) {
        for (const auto& annotation_sequence :
             annotation_group.annotation_sequence()) {
          if (annotation_sequence.metadata().annotation_type() ==
              AnnotationMetadata::CLASS_LABEL) {
            cc
                ->GetCounter(absl::Substitute(
                    "class-mapping-$0-with-label-$1",
                    annotation_sequence.metadata()
                        .source_details()
                        .identifier()
                        .name(),
                    annotation_sequence.annotation(0).label(0).name()))
                ->Increment();
          }
        }
      }
    }

    cc->Outputs()
        .Tag(kOutputAnnotatedRecordingCollectionTag)
        .Add(result_arc.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  const ClassMappingsProvider* label_mapping_provider_;
};
REGISTER_CALCULATOR(AddClassMappingsCalculator);

}  // namespace ambient_sensing
