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

#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "intent_recognition/processing/class_mappings_provider.h"
#include "intent_recognition/processing/class_mappings_side_packet_calculator.pb.h"

namespace ambient_sensing {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kClassMappingsProvider[] = "CLASS_MAPPINGS_PROVIDER";

// Calculator that based on the provided options creates a ClassMappingsProvider
// that can be used to add mappings to AnnotatedRecordingCollections.
class ClassMappingsSidePacketCalculator : public mediapipe::CalculatorBase {
 public:
  ClassMappingsSidePacketCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets()
        .Tag(kClassMappingsProvider)
        .Set<std::unique_ptr<ClassMappingsProvider>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const ClassMappingsSidePacketCalculatorOptions& options =
        cc->Options().GetExtension(
            ClassMappingsSidePacketCalculatorOptions::ext);
    std::unique_ptr<ClassMappingsProvider> mapping_provider;

    switch (options.mapping_type()) {
      case ClassMappingsSidePacketCalculatorOptions::STRING_CLASS: {
        ASSIGN_OR_RETURN(mapping_provider,
                         ClassMappingsProvider::NewStringLabelProvider(
                             {options.label_mapping_files().begin(),
                              options.label_mapping_files().end()}));
      } break;
      default:
        return absl::InvalidArgumentError("Unsupported label type provided.");
    }

    cc->OutputSidePackets()
        .Tag(kClassMappingsProvider)
        .Set(mediapipe::AdoptAsUniquePtr<ClassMappingsProvider>(
            mapping_provider.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(ClassMappingsSidePacketCalculator);

}  // namespace ambient_sensing
