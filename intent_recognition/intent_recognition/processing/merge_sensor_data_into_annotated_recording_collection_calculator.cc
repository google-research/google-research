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

// Calculator that takes streams of sensor data and packs those values into an
// AnnotatedRecordingCollection proto. It produces only 1 message after
// consuming all data from streams. Input streams all have tag INPUT_DATA_STREAM
// and the number of input streams equals number of sensor_options specified in
// calculator options. Each input stream has type std::vector<float> and
// checking is performed after consuming each input value to validate the size
// of passed vector. For each input sensor stream there will be a sequence
// added to the produced output message, even if there is no data in the stream.
//
// Note that in this approach a sensor that produces scalars is instead treated
// as a sensor that produces 1-dimensional vector.
//
// Note that currently this calculator only fills in the timeserieses, and
// doesn't set any metadata.

#include <cstdint>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_sensor_options.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/processing/merge_sensor_data_into_annotated_recording_collection_calculator.pb.h"

namespace ambient_sensing {
namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kInputSensorDataStreamTag[] = "INPUT_DATA_STREAM";
constexpr char kOutputAnnotatedRecordingCollectionTag[] =
    "OUTPUT_ANNOTATED_RECORDING_COLLECTION";

using ::ambient_sensing::Sequence;
}  // namespace

// TODO(rachelhornung) accept windows of data.
class MergeSensorDataIntoAnnotatedRecordingCollectionCalculator
    : public mediapipe::CalculatorBase {
 public:
  MergeSensorDataIntoAnnotatedRecordingCollectionCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs()
        .Tag(kOutputAnnotatedRecordingCollectionTag)
        .Set<AnnotatedRecordingCollection>();

    if (!cc->Options().HasExtension(
            MergeSensorDataIntoAnnotatedRecordingCollectionCalculatorOptions::
                ext)) {
      return absl::InvalidArgumentError(
          "MergeSensorDataIntoAnnotatedRecordingCollectionCalculatorOptions "
          "not set. This will result in calculator processing 0 streams and "
          "is likely not what you intended to do.");
    }

    auto options = cc->Options().GetExtension(
        MergeSensorDataIntoAnnotatedRecordingCollectionCalculatorOptions::ext);

    if (options.sensor_options_size() !=
        cc->Inputs().NumEntries(kInputSensorDataStreamTag)) {
      return absl::InvalidArgumentError(
          "Bad configuration: number of input data streams does not match "
          "number of sensor options.");
    }

    for (int i = 0; i < options.sensor_options_size(); i++) {
      cc->Inputs().Get(kInputSensorDataStreamTag, i).Set<std::vector<float>>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options().GetExtension(
        MergeSensorDataIntoAnnotatedRecordingCollectionCalculatorOptions::ext);
    int current_stream_index = 0;
    for (const auto& sensor_options : options_.sensor_options()) {
      sensor_type_by_input_stream_index_[current_stream_index] = {
          sensor_options.type(), sensor_options.subtype()};

      auto sequence = absl::make_unique<Sequence>();
      sequence->mutable_metadata()->set_type(sensor_options.type());
      sequence->mutable_metadata()->set_subtype(sensor_options.subtype());
      sequence_by_input_stream_index_.push_back(std::move(sequence));
      sequence_sample_size_.push_back(absl::optional<int32_t>());
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    for (int i = 0; i < cc->Inputs().NumEntries(kInputSensorDataStreamTag);
         i++) {
      auto& input_stream = cc->Inputs().Get(kInputSensorDataStreamTag, i);
      auto& sensor_options = options_.sensor_options(i);
      auto* sequence = sequence_by_input_stream_index_.at(i).get();
      if (!input_stream.IsEmpty()) {
        auto data = input_stream.Get<std::vector<float>>();
        absl::Status status = ValidateDataDimensions(data, sensor_options, cc);
        if (status != absl::OkStatus()) return status;
        status = VerifyDataSizeMatch(data.size(), &sequence_sample_size_.at(i));
        if (status != absl::OkStatus()) return status;
        if (sensor_options.window_dims() < 1) {
          sequence->mutable_metadata()->set_measurement_dimensionality(
              data.size());
          sequence->mutable_metadata()->set_windowed(false);
          auto* datapoint =
              sequence->mutable_repeated_datapoint()->add_datapoint();
          *datapoint->mutable_offset() = ConvertDurationToProto(
              absl::Microseconds(cc->InputTimestamp().Microseconds()));
          for (int i = 0; i < data.size(); i++) {
            datapoint->mutable_double_value()->add_value(data[i]);
          }
        } else {
          sequence->mutable_metadata()->set_window_size(
              sensor_options.window_dims());
          sequence->mutable_metadata()->set_windowed(true);
          auto* window = sequence->mutable_repeated_window()->add_window();
          *window->mutable_offset() = ConvertDurationToProto(
              absl::Microseconds(cc->InputTimestamp().Microseconds()));
          ambient_sensing::Datapoint* datapoint;
          int32_t sample_size = data.size() / sensor_options.window_dims();
          sequence->mutable_metadata()->set_measurement_dimensionality(
              sample_size);
          for (int i = 0; i < data.size(); i++) {
            // Create a new datapoint every window_dims inputs.
            if (i % sample_size == 0) {
              datapoint = window->add_datapoint();
            }
            datapoint->mutable_double_value()->add_value(data[i]);
          }
        }
      }
    }
    merged_data_ = true;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    // Only release ARC if at least one input timestamp has been processed.
    if (merged_data_) {
      // Copy all sequences into the result message.
      for (auto& sequence : sequence_by_input_stream_index_) {
        output_.mutable_recording_collection()
            ->mutable_sequence()
            ->AddAllocated(sequence.release());
      }
      cc->Outputs()
          .Tag(kOutputAnnotatedRecordingCollectionTag)
          .Add(absl::make_unique<AnnotatedRecordingCollection>(output_)
                   .release(),
               mediapipe::Timestamp::PreStream());
    }
    return absl::OkStatus();
  }

 private:
  // TODO(b/149202418): factor out?
  // TODO(rachelhornung) consider removing required_dims, since we have
  // measurement)dimensionality and window_size for each sequence.
  absl::Status ValidateDataDimensions(
      const std::vector<float>& data,
      const RecordingCollectionSensorOptions& sensor_options,
      CalculatorContext* cc) {
    int32_t required_datapoint_dims = sensor_options.required_dims();
    if (sensor_options.window_dims() > 0) {
      int32_t required_window_dims =
          required_datapoint_dims * sensor_options.window_dims();
      if (required_datapoint_dims > 0 && data.size() != required_window_dims) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid window data size at timestamp %d ms. "
                            "Required dimensions: %d, actual dimensions: %d",
                            cc->InputTimestamp().Microseconds(),
                            required_window_dims, data.size()));
      }
    } else {
      if (required_datapoint_dims > 0 &&
          data.size() != required_datapoint_dims) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid datapoint data size at timestamp %d ms. "
                            "Required dimensions: %d, actual dimensions: %d",
                            cc->InputTimestamp().Microseconds(),
                            required_datapoint_dims, data.size()));
      }
    }

    return absl::OkStatus();
  }

  absl::Status VerifyDataSizeMatch(
      int32_t current_sample_size,
      absl::optional<int32_t>* previous_sample_sizes) {
    if (previous_sample_sizes->has_value() &&
        previous_sample_sizes->value() != current_sample_size) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Sample size mismatch: previously $0, now $1.",
          previous_sample_sizes->value(), current_sample_size));
    }
    *previous_sample_sizes = current_sample_size;
    return absl::OkStatus();
  }

  bool merged_data_ = false;
  AnnotatedRecordingCollection output_;
  absl::flat_hash_map<int32_t, std::pair<std::string, std::string>>
      sensor_type_by_input_stream_index_;
  std::vector<std::unique_ptr<Sequence>> sequence_by_input_stream_index_;
  std::vector<absl::optional<int32_t>> sequence_sample_size_;
  MergeSensorDataIntoAnnotatedRecordingCollectionCalculatorOptions options_;
};
REGISTER_CALCULATOR(MergeSensorDataIntoAnnotatedRecordingCollectionCalculator);

}  // namespace ambient_sensing
