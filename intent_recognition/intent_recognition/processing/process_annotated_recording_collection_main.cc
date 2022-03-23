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

// Process AnnotatedRecordingCollections.

#include <fcntl.h>

#include <fstream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "mediapipe/calculators/core/packet_resampler_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_sensor_options.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/processing/class_mappings_side_packet_calculator.pb.h"
#include "intent_recognition/processing/filter_annotated_recording_collection_calculator.pb.h"
#include "intent_recognition/processing/merge_sensor_data_into_annotated_recording_collection_calculator.pb.h"
#include "intent_recognition/processing/window_calculator.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"

ABSL_FLAG(std::string, record_input_filename, "",
          "Path to the riegeli file containing the unprocessed "
          "AnnotatedRecordingCollections.");

ABSL_FLAG(std::string, record_output_filename, "",
          "Path to write the riegeli file containing the resulting "
          "processed AnnotatedRecordingCollections.");

ABSL_FLAG(std::string, processing_options_filename, "",
          "Path to processing options file.");

namespace ambient_sensing {
namespace {

// Process a single AnnotatedRecordingCollection.
absl::StatusOr<std::vector<AnnotatedRecordingCollection>> Process(
    const AnnotatedRecordingCollection& arc,
    const mediapipe::CalculatorGraphConfig& graph_config) {
  mediapipe::CalculatorGraph graph;
  absl::Status status;

  status = graph.Initialize(graph_config);
  if (!status.ok()) return status;

  absl::StatusOr<mediapipe::OutputStreamPoller> poller =
      graph.AddOutputStreamPoller("output");
  if (!poller.ok()) return poller.status();

  status = graph.StartRun(
      {{"input", mediapipe::MakePacket<AnnotatedRecordingCollection>(arc)}});
  if (!status.ok()) return status;

  mediapipe::Packet packet;
  std::vector<AnnotatedRecordingCollection> result;
  while (poller->Next(&packet)) {
    result.push_back(packet.Get<AnnotatedRecordingCollection>());
  }

  status = graph.CloseAllPacketSources();
  if (!status.ok()) return status;
  status = graph.WaitUntilDone();
  if (!status.ok()) return status;

  return result;
}

absl::Status ProcessAnnotatedRecordingCollection(
    absl::string_view input_record_filename,
    absl::string_view output_record_filename,
    const ProcessingOptions& processing_options) {
  int input_fd = open(input_record_filename.data(), O_RDONLY);
  if (input_fd < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to open input file: ", strerror(errno)));
  }

  int output_fd = open(output_record_filename.data(), O_RDWR | O_CREAT, 666);
  if (output_fd < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to open output file: ", strerror(errno)));
  }

  riegeli::RecordReader input_reader((riegeli::FdReader(input_fd)));
  riegeli::RecordWriter output_writer((riegeli::FdWriter(output_fd)));

  std::ifstream input_stream(processing_options.processing_graph_file());
  if (!input_stream.is_open()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find file ", processing_options.processing_graph_file()));
  }
  std::string proto_string =
      std::string((std::istreambuf_iterator<char>(input_stream)),
                  (std::istreambuf_iterator<char>()));

  // Create the graph config based on the processing options.
  mediapipe::CalculatorGraphConfig graph_config =
      BuildDrishtiGraphWithProcessingOptions(proto_string, processing_options);
  LOG(INFO) << "Graph config: " << graph_config.DebugString();

  // Read input records, process them, and store the result.
  AnnotatedRecordingCollection input_arc;
  int num_records = 0;
  int num_records_written = 0;
  while (input_reader.ReadRecord(input_arc)) {
    ++num_records;
    absl::StatusOr<std::vector<AnnotatedRecordingCollection>> processed_result =
        Process(input_arc, graph_config);
    if (!processed_result.ok()) return processed_result.status();

    for (const auto& result : *processed_result) {
      output_writer.WriteRecord(result);
      ++num_records_written;
    }
  }
  LOG(INFO) << "Num of records read: " << num_records;
  LOG(INFO) << "Num of records written: " << num_records_written;

  input_reader.Close();
  if (!output_writer.Close()) return output_writer.status();

  close(input_fd);
  close(output_fd);

  return absl::OkStatus();
}

}  // namespace
}  // namespace ambient_sensing

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Get the processing options.
  std::ifstream input_stream(absl::GetFlag(FLAGS_processing_options_filename));
  if (!input_stream.is_open()) {
    LOG(ERROR) << "Could not find file: "
               << absl::GetFlag(FLAGS_processing_options_filename);
    return 1;
  }
  std::string proto_string =
      std::string((std::istreambuf_iterator<char>(input_stream)),
                  (std::istreambuf_iterator<char>()));
  ambient_sensing::ProcessingOptions processing_options;
  if (!google::protobuf::TextFormat::ParseFromString(proto_string, &processing_options)) {
    LOG(ERROR) << "Failed to parse processing options textproto from string";
    return 1;
  }

  // Process the AnnotatedRecordingCollections.
  absl::Status status = ambient_sensing::ProcessAnnotatedRecordingCollection(
      absl::GetFlag(FLAGS_record_input_filename),
      absl::GetFlag(FLAGS_record_output_filename), processing_options);
  if (!status.ok()) {
    LOG(ERROR) << "Processing failed: " << status;
    return 1;
  }
  LOG(INFO) << "Processing successful";

  return 0;
}
