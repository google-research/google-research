// Copyright 2021 The Google Research Authors.
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

#include "io.h"

#include <fstream>
#include <iostream>
#include <string>

#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"

namespace sliding_window {

std::vector<double> parse_point_from_string(const std::string& file_line) {
  std::vector<double> point;
  vector<string> fields = absl::StrSplit(file_line, '\t');
  point.reserve(fields.size());
  for (int i = 0; i < fields.size(); i++) {
    point.push_back(0);
    bool ok = absl::SimpleAtod(fields[i], &point.back());
    CHECK(ok);
    ok = true;  // Needed to avoid unused variable warnings.
  }
  return point;
}

std::vector<std::vector<TimePointPair>> get_windows_samples(
    const string& stream_file_path, int32_t window_size, int32_t num_samples,
    int32_t prefix_to_skip, int32_t* stream_size) {
  vector<vector<TimePointPair>> samples;
  samples.reserve(num_samples);
  int64_t lines_read = 0;

  std::string line;
  std::ifstream input_file(stream_file_path, std::ios::in);
  CHECK(input_file.is_open());
  vector<TimePointPair> current_sample;

  while (std::getline(input_file, line)) {
    const auto& point = parse_point_from_string(line);
    lines_read++;

    if (prefix_to_skip == 0) {
      // Add point to the last sample.
      current_sample.push_back(std::make_pair(current_sample.size(), point));
    } else {
      // Skip the point.
      prefix_to_skip--;
    }
    // Sample is full
    if (current_sample.size() == window_size) {
      samples.push_back(current_sample);
      current_sample.clear();
      // Read all samples needed.
      if (samples.size() == num_samples) {
        *stream_size = lines_read;
        return samples;
      }
    }
  }
  input_file.close();

  *stream_size = lines_read;
  return samples;
}

}  // namespace sliding_window
