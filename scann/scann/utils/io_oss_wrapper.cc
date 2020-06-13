// Copyright 2020 The Google Research Authors.
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

namespace tensorflow {
namespace scann_ops {

OpenSourceableFileWriter::OpenSourceableFileWriter(absl::string_view filename)
    : fout_(std::string(filename), std::ofstream::binary) {}

Status OpenSourceableFileWriter::Write(ConstSpan<char> bytes) {
  fout_.write(bytes.data(), bytes.size());
  return OkStatus();
}

Status WriteProtobufToFile(absl::string_view filename,
                           google::protobuf::Message* message) {
  std::ofstream fout(std::string(filename).c_str(), std::ofstream::binary);
  if (!fout)
    return InternalError("Failed to open file " + std::string(filename));
  if (!message->SerializeToOstream(&fout))
    return InternalError("Failed to write proto to " + std::string(filename));
  return OkStatus();
}

Status ReadProtobufFromFile(absl::string_view filename,
                            google::protobuf::Message* message) {
  std::ifstream fin(std::string(filename).c_str(), std::ifstream::binary);
  if (!fin)
    return InternalError("Failed to open file " + std::string(filename));
  if (!message->ParseFromIstream(&fin))
    return InternalError("Failed to parse proto from " + std::string(filename));
  return OkStatus();
}

}  // namespace scann_ops
}  // namespace tensorflow
