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

// Utils for reading either from GCS or from local files.
// Filenames can take one of the following forms:
// - "gcs://bucketname/filepath" will read from GCS.
// - Anything else will open the file normally, using fstream.

#ifndef FILE_FILE_H_
#define FILE_FILE_H_

#include <string>

#include "absl/status/statusor.h"
#include "google/cloud/storage/client.h"

namespace file {

// Class for repeatedly opening and reading files.
// Reading from GCS is more efficient if we reuse the client, which is
// encapsulated in this class.
class FileReader {
 public:
  FileReader() {}
  ~FileReader() {}

  absl::StatusOr<std::string> GetFileContents(const std::string& filename);

 private:
  ::google::cloud::storage::Client gcs_client_;
};

// Note: If reading multiple files from GCS, this is less efficient than
// FileReader::GetFileContents.
inline absl::StatusOr<std::string> GetSingleFileContents(
    const std::string& filename) {
  return FileReader().GetFileContents(filename);
}

}  // namespace file

#endif  // FILE_FILE_H_
