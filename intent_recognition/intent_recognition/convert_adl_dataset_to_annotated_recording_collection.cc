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

// Convert the ADL dataset into AnnotatedRecordingCollections.
// Information about the dataset can be found at
// https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

#include <stdint.h>

#include <filesystem>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/duration.pb.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/lines/line_reading.h"
#include "riegeli/records/record_writer.h"

ABSL_FLAG(std::string, adl_dataset_dir_path, "",
          "Path to the top-level directory that contains the ADL dataset.");
ABSL_FLAG(std::string, record_output_filename, "",
          "Path to write the riegeli file containing the resulting "
          "AnnotatedRecordingCollections.");

constexpr int64_t kSamplingRateHz = 32;
constexpr int64_t kNanosInSecond = 1000 * 1000 * 1000;
constexpr double kGravity = 9.8067;

// Convert the raw acceleration value to m/s^2.
double DecodeAccel(double raw_accel) {
  double accel_gs = -1.5 + (raw_accel / 63.0) * 3.0;
  double accel_mpsps = accel_gs * kGravity;
  return accel_mpsps;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  riegeli::RecordWriter writer(
      riegeli::FdWriter(absl::GetFlag(FLAGS_record_output_filename)));
  if (!writer.ok()) {
    LOG(ERROR) << "Failed to open output file: " << writer.status();
    return 1;
  }

  // Session ID will start at 0 and increment for each entry.
  int session_id = 0;

  // Create an AnnotatedRecordingCollection for each file in the ADL dataset.
  for (const auto& path : std::filesystem::recursive_directory_iterator(
           absl::GetFlag(FLAGS_adl_dataset_dir_path))) {
    if (std::filesystem::is_directory(path)) continue;
    std::vector<std::string> filename_split =
        absl::StrSplit(path.path().filename().generic_string(), '-');
    if (filename_split[0] != "Accelerometer") continue;
    ambient_sensing::AnnotatedRecordingCollection arc;
    ambient_sensing::RecordingCollection* recording_collection =
        arc.mutable_recording_collection();
    ambient_sensing::RecordingCollectionMetadata* recording_metadata =
        recording_collection->mutable_metadata();
    recording_metadata->set_session_id(std::to_string(session_id));
    recording_metadata->set_user_id(
        std::vector<std::string>(absl::StrSplit(filename_split[8], '.'))[0]);
    recording_metadata->mutable_mobile_collection_metadata()
        ->set_session_activity(filename_split[7]);
    ambient_sensing::Sequence* recording_sequence =
        recording_collection->add_sequence();
    ambient_sensing::SequenceMetadata* sequence_metadata =
        recording_sequence->mutable_metadata();
    sequence_metadata->set_type("SENSOR");
    sequence_metadata->set_subtype("ACCELEROMETER");
    sequence_metadata->set_measurement_dimensionality(3);
    ambient_sensing::RepeatedDatapoint* repeated_datapoint =
        recording_sequence->mutable_repeated_datapoint();

    ++session_id;

    // Add accelorometer reading found in the file
    riegeli::FdReader input_file(path.path().generic_string());
    absl::string_view line;
    int64_t entry = 0;
    while (riegeli::ReadLine(input_file, line)) {
      ambient_sensing::Datapoint* datapoint =
          repeated_datapoint->add_datapoint();

      // Add time offset.
      int64_t time_offset_nanos = (entry * kNanosInSecond) / kSamplingRateHz;
      datapoint->mutable_offset()->set_seconds(time_offset_nanos /
                                               kNanosInSecond);
      datapoint->mutable_offset()->set_nanos(time_offset_nanos %
                                             kNanosInSecond);

      // Add accelerometer reading.
      std::vector<std::string> accel_string = absl::StrSplit(line, ' ');
      ambient_sensing::DoubleDatapoint* datapoint_double =
          datapoint->mutable_double_value();
      datapoint_double->add_value(DecodeAccel(std::stod(accel_string[0])));
      datapoint_double->add_value(DecodeAccel(std::stod(accel_string[1])));
      datapoint_double->add_value(DecodeAccel(std::stod(accel_string[2])));

      // Increment entry;
      ++entry;
    }
    if (!input_file.Close()) {
      LOG(ERROR) << "Failed to read input file: " << input_file.status();
    }

    writer.WriteRecord(arc);
  }

  if (!writer.Close()) {
    LOG(ERROR) << "Failed to close record writer: " << writer.status();
    return 1;
  }

  LOG(INFO) << "Successfully converted ADL dataset into "
               "AnnotatedRecordingCollections";

  return 0;
}
