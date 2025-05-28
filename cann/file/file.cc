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

#include "file/file.h"

#include <fstream>
#include <string>
#include <tuple>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

using std::string;

namespace file {

namespace {

constexpr const char* kGcsPrefix = "gs://";

absl::StatusOr<std::tuple<string, string>> GetBucketAndFilepath(
    const std::string& filename) {
  if (!absl::StartsWith(filename, kGcsPrefix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Not a GCS filename: ", filename));
  }
  size_t bucket_end_pos = filename.find('/', 5);
  return std::tuple<string, string>{
      filename.substr(5, bucket_end_pos - 5),  // bucket
      filename.substr(bucket_end_pos + 1)      // filepath
  };
}

}  // namespace

absl::StatusOr<std::string> FileReader::GetFileContents(
    const std::string& filename) {
  if (absl::StartsWith(filename, kGcsPrefix)) {
    auto [bucket, filepath] = GetBucketAndFilepath(filename).value();
    auto reader = gcs_client_.ReadObject(bucket, filepath);
    if (!reader) {
      return absl::NotFoundError(
          absl::StrCat("Could not open file:", filename));
    }
    return string{std::istreambuf_iterator<char>{reader}, {}};
  } else {
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs) {
      return absl::NotFoundError(
          absl::StrCat("Could not open file:", filename));
    }
    return string{std::istreambuf_iterator<char>{ifs}, {}};
  }
}

}  // namespace file
