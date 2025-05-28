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

// Calculator that checks whether the data in the reference stream is above or
// equal to a specified threshold. The size of INPUT_REFERENCE_VALUE has to
// match that of options.threshold, such that each value has their own reference
// value. How the assessments of multiple values are aggregated can be
// determined via options.comparator. Setting invert_threshold=true will drop
// values above or equal to threshold. Only data where the threshold requirement
// is met will be passed. Data without a reference value will be dropped.

#include <cstdint>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "intent_recognition/processing/drop_on_threshold_value_calculator.pb.h"

namespace ambient_sensing {

namespace {
constexpr char kInputDataTag[] = "INPUT_DATA";
constexpr char kReferenceValueTag[] = "INPUT_REFERENCE_VALUE";
constexpr char kOutputDataTag[] = "OUTPUT_DATA";

using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;
}  // namespace

class DropOnThresholdValueCalculator : public mediapipe::CalculatorBase {
 public:
  DropOnThresholdValueCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->Inputs().NumEntries(kReferenceValueTag) != 1) {
      return absl::InvalidArgumentError(
          "Bad configuration: Exactly one reference value stream required.");
    }
    cc->Inputs().Tag(kReferenceValueTag).Set<std::vector<float>>();
    int32_t num_input_streams = cc->Inputs().NumEntries(kInputDataTag);
    if (num_input_streams < 1) {
      return absl::InvalidArgumentError(
          "Bad configuration: At least one data stream required.");
    }
    int32_t num_output_streams = cc->Outputs().NumEntries(kOutputDataTag);
    if (num_output_streams != num_input_streams) {
      return absl::InvalidArgumentError(
          absl::Substitute("Bad configuration: The number of inputs for $0 "
                           "must match the number of outputs for $1.",
                           kInputDataTag, kOutputDataTag));
    }
    for (int i = 0; i < num_input_streams; i++) {
      cc->Inputs().Get(kInputDataTag, i).Set<std::vector<float>>();
      cc->Outputs().Get(kOutputDataTag, i).Set<std::vector<float>>();
    }

    if (!cc->Options().HasExtension(
            DropOnThresholdValueCalculatorOptions::ext)) {
      return absl::InvalidArgumentError(
          "DropOnThresholdValueCalculatorOptions must be specified.");
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const DropOnThresholdValueCalculatorOptions& options =
        cc->Options().GetExtension(DropOnThresholdValueCalculatorOptions::ext);
    if (options.threshold().empty()) {
      return absl::InvalidArgumentError("Threshold has to be specified.");
    }
    options_ = options;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const mediapipe::InputStream& reference_stream =
        cc->Inputs().Tag(kReferenceValueTag);
    if (reference_stream.IsEmpty()) {
      cc->GetCounter("No reference value available. Data dropped.")
          ->Increment();
      return absl::OkStatus();
    }

    const auto& reference = reference_stream.Get<std::vector<float>>();
    if (reference.size() != options_.threshold().size()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Reference size $0 does not match threshold size $1.",
          reference.size(), options_.threshold().size()));
    }
    bool below_threshold = false;
    switch (options_.drop_below_threshold()) {
      case DropOnThresholdValueCalculatorOptions::UNDEFINED:
        return absl::InvalidArgumentError("Invalid comparator selected.");
      case DropOnThresholdValueCalculatorOptions::ANY:
        for (int i = 0; i < reference.size(); ++i) {
          if (reference[i] < options_.threshold().at(i)) {
            below_threshold = true;
            break;
          }
        }
        break;
      case DropOnThresholdValueCalculatorOptions::ALL:
        bool has_value_above_threshold = false;
        for (int i = 0; i < reference.size(); ++i) {
          if (reference[i] >= options_.threshold().at(i)) {
            has_value_above_threshold = true;
            break;
          }
        }
        if (!has_value_above_threshold) {
          below_threshold = true;
        }
        break;
    }
    if (below_threshold) {
      if (!options_.invert_threshold()) {
        cc->GetCounter("Reference below threshold. Data dropped.")->Increment();
        return absl::OkStatus();
      }
      cc->GetCounter(
            "Reference below threshold and inversion selected. Keeping Data.")
          ->Increment();
    } else {
      if (options_.invert_threshold()) {
        cc->GetCounter(
              "Reference not below threshold and inversion selected. Data "
              "dropped.")
            ->Increment();
        return absl::OkStatus();
      }
      cc->GetCounter("Reference not below threshold. Keeping Data.")
          ->Increment();
    }

    for (int i = 0; i < cc->Inputs().NumEntries(kInputDataTag); i++) {
      const mediapipe::Packet& packet =
          cc->Inputs().Get(kInputDataTag, i).Value();
      cc->Outputs().Get(kOutputDataTag, i).AddPacket(packet);
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  DropOnThresholdValueCalculatorOptions options_;
};
REGISTER_CALCULATOR(DropOnThresholdValueCalculator);

}  // namespace ambient_sensing
