// Copyright 2024 The Google Research Authors.
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

#ifndef SCANN_UTILS_IO_OSS_WRAPPER_H_
#define SCANN_UTILS_IO_OSS_WRAPPER_H_

#include <fstream>
#include <memory>
#include <string>

#include "google/protobuf/message.h"
#include "scann/utils/common.h"

namespace research_scann {

class OpenSourceableFileWriter {
 public:
  explicit OpenSourceableFileWriter(absl::string_view filename);
  Status Write(ConstSpan<char> bytes);

 private:
  std::ofstream fout_;
};

class OpenSourceableFileReader {
 public:
  explicit OpenSourceableFileReader(absl::string_view filename);
  void ReadLine(std::string& dest);
  void Read(size_t bytes, char* buffer);

 private:
  std::ifstream fin_;
};

Status WriteProtobufToFile(absl::string_view filename,
                           google::protobuf::Message* message);
Status ReadProtobufFromFile(absl::string_view filename,
                            google::protobuf::Message* message);
absl::StatusOr<std::string> GetContents(absl::string_view filename);

}  // namespace research_scann

#endif
