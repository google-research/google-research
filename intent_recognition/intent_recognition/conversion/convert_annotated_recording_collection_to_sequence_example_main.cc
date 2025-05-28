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

// Convert AnnotatedRecordingCollections to SequenceExamples.

#include <fcntl.h>

#include <memory>

#include "glog/logging.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/conversion/convert_annotated_recording_collection_to_sequence_example.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/records/record_reader.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"

ABSL_FLAG(std::string, riegeli_record_input_filename, "",
          "Path to the riegeli file containing processed "
          "AnnotatedRecordingCollections.");

ABSL_FLAG(std::string, tf_record_output_filename, "",
          "Path to write the tfrecord file containing the resulting "
          "SequenceExamples");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  riegeli::RecordReader input_reader(
      riegeli::FdReader(absl::GetFlag(FLAGS_riegeli_record_input_filename)));
  if (!input_reader.status().ok()) {
    LOG(ERROR) << "Failed to read input file: " << input_reader.status();
    return 1;
  }

  std::unique_ptr<tensorflow::WritableFile> output_file;
  absl::Status tf_status = tensorflow::Env::Default()->NewWritableFile(
      absl::GetFlag(FLAGS_tf_record_output_filename), &output_file);
  tensorflow::io::RecordWriter output_writer(output_file.get());

  ambient_sensing::AnnotatedRecordingCollection processed_arc;
  int num_records_read = 0;
  int num_records_written = 0;
  while (input_reader.ReadRecord(processed_arc)) {
    ++num_records_read;
    absl::StatusOr<tensorflow::SequenceExample> example =
        ConvertAnnotatedRecordingCollectionToSequenceExample(
            /*flatten=*/false,
            processed_arc.recording_collection().metadata().session_id(),
            processed_arc);
    if (!example.ok()) {
      LOG(WARNING) << "Failed to convert AnnotatedRecordingCollection to "
                      "SequenceExample: "
                   << example.status();

      continue;
    }

    tf_status = output_writer.WriteRecord(example->SerializeAsString());
    if (!tf_status.ok()) {
      LOG(ERROR) << "Failed to write to tf record: " << tf_status;
      return 1;
    }

    ++num_records_written;
  }
  LOG(INFO) << "Num of records read: " << num_records_read;
  LOG(INFO) << "Num of records written: " << num_records_written;

  tf_status = output_writer.Close();
  if (!tf_status.ok()) {
    LOG(WARNING) << "Failed to close record writer";
  }
  if (!input_reader.Close()) {
    LOG(WARNING) << "Failed to close record reader: " << input_reader.status();
  }

  LOG(INFO) << "Conversion to SequenceExample complete";

  return 0;
}
