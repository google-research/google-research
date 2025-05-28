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

#include "scann/utils/io_oss_wrapper.h"

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "scann/utils/common.h"

namespace research_scann {

OpenSourceableFileWriter::OpenSourceableFileWriter(absl::string_view filename)
    : fout_(std::string(filename), std::ofstream::binary) {}

Status OpenSourceableFileWriter::Write(ConstSpan<char> bytes) {
  if (!fout_.write(bytes.data(), bytes.size())) {
    return InternalError("I/O error");
  }
  return OkStatus();
}

OpenSourceableFileReader::OpenSourceableFileReader(absl::string_view filename)
    : fin_(std::string(filename), std::ifstream::binary) {}

Status OpenSourceableFileReader::ReadLine(std::string& dest) {
  if (!std::getline(fin_, dest)) {
    return fin_.bad() ? InternalError("I/O error")
                      : OutOfRangeError("File too short");
  }
  return OkStatus();
}

Status OpenSourceableFileReader::Read(size_t bytes, char* buffer) {
  if (!fin_.read(buffer, bytes)) {
    return fin_.bad() ? InternalError("I/O error")
                      : OutOfRangeError("File too short");
  }
  return OkStatus();
}

Status WriteProtobufToFile(absl::string_view filename,
                           const google::protobuf::Message& message) {
  std::ofstream fout(std::string(filename), std::ofstream::binary);
  if (!fout) {
    return InternalError("Failed to open file " + std::string(filename));
  }
  if (!message.SerializeToOstream(&fout)) {
    return InternalError("Failed to write proto to " + std::string(filename));
  }
  return OkStatus();
}

Status ReadProtobufFromFile(absl::string_view filename,
                            google::protobuf::Message* message) {
  std::ifstream fin(std::string(filename), std::ifstream::binary);
  if (!fin) {
    return InternalError("Failed to open file " + std::string(filename));
  }
  if (!message->ParseFromIstream(&fin)) {
    return InternalError("Failed to parse proto from " + std::string(filename));
  }
  return OkStatus();
}

absl::StatusOr<std::string> GetContents(absl::string_view filename) {
  std::ifstream input_stream{std::string(filename)};
  if (!input_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Input file %s not opened successfully.", filename));
  }
  std::stringstream content;
  content << input_stream.rdbuf();
  return content.str();
}

}  // namespace research_scann
