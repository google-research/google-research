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

// Calculator that takes a non-windowed AnnotatedRecordingCollection proto and
// produces a stream of sensor data. There will be 1 output stream per each type
// of sensor specified in the config with tag DATA_STREAM and index starting
// from 0, e.g. DATA_STREAM:0, DATA_STREAM:1, etc. The order of data streams
// will match that of the config, e.g. if the first specified sensor type is <2,
// 3> (SENSOR, ACCELEROMETER), then DATA_STREAM:0 will contain accelerometer
// data. For sensors that have SensorOptions.default_value set, the
// default_value will be prepended to the timeseries. This is used to have a
// value for packet_cloner at the first timestamp of the tick signal. Raises an
// error if default dimensions do not match the requested dimensions.
//
// The input can be passed as a side input packet or as an input stream with
// exactly one packet. If the input is passed through via input stream, its
// timestamp is not used for producing the output: the output timestamps are
// set based on the data within the annotated timeseries message.

#include <cstdint>
#include <vector>

#include "net/google::protobuf/util/legacy_debug_string/legacy_unredacted_debug_string.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_sensor_options.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/processing/extract_sensor_data_from_annotated_recording_collection_calculator.pb.h"

namespace ambient_sensing {
namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;
using ::google::protobuf::util::LegacyUnredactedDebugString;

constexpr char kOutputDataStreamTag[] = "OUTPUT_DATA_STREAM";
constexpr char kInputAnnotatedRecordingCollectionTag[] =
    "INPUT_ANNOTATED_RECORDING_COLLECTION";

using SensorType = std::pair<std::string, std::string>;
}  // namespace

class ExtractSensorDataFromAnnotatedRecordingCollectionCalculator
    : public mediapipe::CalculatorBase {
 public:
  ExtractSensorDataFromAnnotatedRecordingCollectionCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    bool input_side_packets_has_tag =
        cc->InputSidePackets().HasTag(kInputAnnotatedRecordingCollectionTag);
    bool inputs_has_tag =
        cc->Inputs().HasTag(kInputAnnotatedRecordingCollectionTag);
    // TODO(b/149202382): factor out (copied from
    // FilterAnnotatedTimeseriesCalculator).
    if (inputs_has_tag && input_side_packets_has_tag) {
      return absl::InvalidArgumentError(
          "Input stream and input side packet can't be used simultaneously.");
    }
    if (!inputs_has_tag && !input_side_packets_has_tag) {
      return absl::InvalidArgumentError(
          "Input stream or input side packet must be specified.");
    }

    if (input_side_packets_has_tag) {
      cc->InputSidePackets()
          .Tag(kInputAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    if (inputs_has_tag) {
      cc->Inputs()
          .Tag(kInputAnnotatedRecordingCollectionTag)
          .Set<AnnotatedRecordingCollection>();
    }
    for (int i = 0; i < cc->Outputs().NumEntries(kOutputDataStreamTag); i++) {
      cc->Outputs().Get(kOutputDataStreamTag, i).Set<std::vector<float>>();
    }
    for (
        const auto& sensor_options :
        cc->Options()
            .GetExtension(
                ExtractSensorDataFromAnnotatedRecordingCollectionCalculatorOptions:: // NOLINT
                    ext)
            .sensor_options()) {
      bool required_dims_set = sensor_options.required_dims() > 0;
      bool default_value_provided = sensor_options.default_value_size() != 0;
      bool dims_mismatch =
          sensor_options.default_value_size() != sensor_options.required_dims();
      if (required_dims_set && default_value_provided && dims_mismatch) {
        return absl::InvalidArgumentError(
            absl::StrCat("Bad argument for default value: given ",
                         sensor_options.default_value_size(), ", but required ",
                         sensor_options.required_dims()));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    auto options = cc->Options().GetExtension(
        ExtractSensorDataFromAnnotatedRecordingCollectionCalculatorOptions::
            ext);
    return BuildSensorDataStreamsMap(options, cc);
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (!cc->Inputs().HasTag(kInputAnnotatedRecordingCollectionTag))
      return mediapipe::tool::StatusStop();
    if (processed_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input stream must have exactly 1 value. Found a "
                       "second value @ ",
                       cc->InputTimestamp().Microseconds()));
    }
    processed_ = true;
    const auto& input = cc->Inputs()
                            .Tag(kInputAnnotatedRecordingCollectionTag)
                            .Get<AnnotatedRecordingCollection>();
    return ExtractSensorData(input, cc);
  }

  absl::Status Close(CalculatorContext* cc) override {
    if (cc->InputSidePackets().HasTag(kInputAnnotatedRecordingCollectionTag)) {
      const auto& annotated_recording_collection =
          cc->InputSidePackets()
              .Tag(kInputAnnotatedRecordingCollectionTag)
              .Get<AnnotatedRecordingCollection>();
      return ExtractSensorData(annotated_recording_collection, cc);
    }
    return absl::OkStatus();
  }

 private:
  absl::Status BuildSensorDataStreamsMap(
      const ExtractSensorDataFromAnnotatedRecordingCollectionCalculatorOptions&
          options,
      CalculatorContext* cc) {
    int current_index = 0;
    for (const auto& sensor_options : options.sensor_options()) {
      if (sensor_options.duplicate_timestamps_handling_strategy() ==
          RecordingCollectionSensorOptions::UNKNOWN) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Bad configuration: duplicate_timestamps_handling_strategy must be "
            "set. ",
            LegacyUnredactedDebugString(sensor_options), " ",
            LegacyUnredactedDebugString(options)));
      }
      SensorType sensor_type = {sensor_options.type(),
                                sensor_options.subtype()};
      auto emplace1 =
          sensor_data_streams_map_.emplace(sensor_type, current_index++);
      auto emplace2 = sensor_options_map_.emplace(sensor_type, sensor_options);

      if (!emplace1.second || !emplace2.second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid options: repeated type/subtype combination ",
                         LegacyUnredactedDebugString(sensor_options), " ",
                         LegacyUnredactedDebugString(options)));
      }
    }

    if (sensor_data_streams_map_.size() != cc->Outputs().NumEntries()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Bad configuration: number of output data streams does not match "
          "number of sensor data streams map: ",
          cc->Outputs().NumEntries(), " ", sensor_data_streams_map_.size()));
    }
    return absl::OkStatus();
  }

  // Adds the default value to the output stream at Timestamp::Min().
  absl::Status OutputDefault(
      const RecordingCollectionSensorOptions& sensor_options, int output_index,
      CalculatorContext* cc) {
    auto sensor_data = absl::make_unique<std::vector<float>>(
        sensor_options.default_value().cbegin(),
        sensor_options.default_value().cend());
    cc->Outputs()
        .Get(kOutputDataStreamTag, output_index)
        .Add(sensor_data.release(), mediapipe::Timestamp::Min());
    return absl::OkStatus();
  }

  // Adds a value to the output stream.
  // Raises an error if actual dimensions don't match the required dimensions.
  absl::Status OutputData(
      const RecordingCollectionSensorOptions& sensor_options, int output_index,
      const Datapoint& datapoint, CalculatorContext* cc) {
    absl::Duration datapoint_offset =
        ConvertProtoToDuration(datapoint.offset());
    if (sensor_options.required_dims() >= 0 &&
        datapoint.double_value().value_size() !=
            sensor_options.required_dims()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Bad argument at offset ", absl::FormatDuration(datapoint_offset),
          " - size ", datapoint.double_value().value_size(), ", but required ",
          sensor_options.required_dims()));
    }
    auto sensor_data = absl::make_unique<std::vector<float>>(
        datapoint.double_value().value().cbegin(),
        datapoint.double_value().value().cend());
    cc->Outputs()
        .Get(kOutputDataStreamTag, output_index)
        .Add(sensor_data.release(),
             mediapipe::Timestamp(absl::ToInt64Microseconds(datapoint_offset)));
    return absl::OkStatus();
  }

  absl::Status ExtractSensorData(
      const AnnotatedRecordingCollection& annotated_recording_collection,
      CalculatorContext* cc) {
    // Provide default value for sensors for which SensorOptions.default_value
    // is set at Timestamp::Min(), if requested in sensor_options.
    for (const auto& requested_sensor : sensor_options_map_) {
      const RecordingCollectionSensorOptions& sensor_options =
          requested_sensor.second;
      if (!sensor_options.default_value().empty()) {
        auto sensor_data_streams_map_it =
            sensor_data_streams_map_.find(requested_sensor.first);
        CHECK(sensor_data_streams_map_it != sensor_data_streams_map_.end());
        int32_t output_index = sensor_data_streams_map_it->second;
        absl::Status status = OutputDefault(sensor_options, output_index, cc);
        if (status != absl::OkStatus()) return status;
      }
    }

    // Pass measurements.
    for (const auto& sequence :
         annotated_recording_collection.recording_collection().sequence()) {
      SensorType sensor_type = {sequence.metadata().type(),
                                sequence.metadata().subtype()};
      auto sensor_data_streams_map_it =
          sensor_data_streams_map_.find(sensor_type);
      auto sensor_options_map_it = sensor_options_map_.find(sensor_type);
      if (sensor_data_streams_map_it == sensor_data_streams_map_.end() ||
          sensor_options_map_it == sensor_options_map_.end()) {
        cc->GetCounter(absl::StrCat("sensor-not-found-in-map-",
                                    sensor_type.first, " ", sensor_type.second))
            ->Increment();
        continue;
      }

      int32_t output_index = sensor_data_streams_map_it->second;
      const RecordingCollectionSensorOptions& sensor_options =
          sensor_options_map_it->second;

      for (int i = 0; i < sequence.repeated_datapoint().datapoint_size(); i++) {
        // Either last value in the stream, or the next value has different
        // timestamp.
        bool last_value =
            (i + 1 == sequence.repeated_datapoint().datapoint_size());
        absl::Duration datapoint_offset = ConvertProtoToDuration(
            sequence.repeated_datapoint().datapoint(i).offset());
        bool last_value_in_cluster;
        absl::Duration next_datapoint_offset;
        if (!last_value) {
          next_datapoint_offset = ConvertProtoToDuration(
              sequence.repeated_datapoint().datapoint(i + 1).offset());
          last_value_in_cluster =
              !last_value && (datapoint_offset != next_datapoint_offset);
        }
        if (last_value || last_value_in_cluster) {
          absl::Status status =
              OutputData(sensor_options, output_index,
                         sequence.repeated_datapoint().datapoint(i), cc);
          if (status != absl::OkStatus()) return status;
        } else {
          switch (sensor_options.duplicate_timestamps_handling_strategy()) {
            case RecordingCollectionSensorOptions::KEEP_LAST_VALUE:
              // This case already handled above; do nothing (i.e. drop the
              // current value since the one ahead of it has the same
              // timestamp).
              break;
            // Configuration is checked at startup, so this should not be
            // possible.
            case RecordingCollectionSensorOptions::UNKNOWN:
              return absl::InternalError("Internal error");
            case RecordingCollectionSensorOptions::RAISE_ERROR:
              return absl::InvalidArgumentError(absl::StrCat(
                  "Duplicate timestamp for stream ",
                  LegacyUnredactedDebugString(sensor_options), " at offset ",
                  absl::FormatDuration(datapoint_offset)));
          }
        }
      }
    }
    return absl::OkStatus();
  }

  // Key: SensorType.
  // Value: output stream index.
  absl::flat_hash_map<SensorType, int32_t> sensor_data_streams_map_;
  // Key: SensorType.
  // Value: sensor options.
  absl::flat_hash_map<SensorType, RecordingCollectionSensorOptions>
      sensor_options_map_;
  AnnotatedRecordingCollection input_;
  bool processed_ = false;
};
REGISTER_CALCULATOR(
    ExtractSensorDataFromAnnotatedRecordingCollectionCalculator);

}  // namespace ambient_sensing
