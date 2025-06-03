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

#include "tsv_utils.h"

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace utils_tsv {

TsvRow TsvSpec::CreateRow() const { return TsvRow(this); }

std::string TsvSpec::GetHeaderLine() const {
  return absl::StrJoin(field_names_.begin(), field_names_.end(),
                       std::string(1, delim_));
}

std::string TsvRow::ToString() const {
  std::vector<std::string> values;
  for (const std::string& field_name : spec_->field_names()) {
    auto it = field_values_.find(field_name);
    if (it != field_values_.end())
      values.push_back(it->second);
    else
      values.push_back("");
  }
  return absl::StrJoin(values, std::string(1, spec_->delim()));
}

std::vector<std::string> TsvReader::GetValuesFrom(const std::string& line) {
  return absl::StrSplit(line, std::string(1, delim_));
}

TsvReader::TsvReader(const std::string& file_path, char delim)
    : delim_(delim), file_(file_path) {
  std::string str;
  std::getline(file_, str);
  field_names_ = GetValuesFrom(str);
}

absl::flat_hash_map<std::string, std::string> TsvReader::ReadRow() {
  std::string str;
  std::getline(file_, str);
  std::vector<std::string> unmapped_values = GetValuesFrom(str);
  absl::flat_hash_map<std::string, std::string> mapped_values;
  for (int i = 0; i < field_names_.size(); ++i) {
    mapped_values[field_names_[i]] = unmapped_values[i];
  }
  return mapped_values;
}

TsvWriter::TsvWriter(const std::string& file_path, const TsvSpec* spec)
    : spec_(spec) {
  file_.open(file_path, std::ios::trunc);
  file_ << absl::StrCat(spec_->GetHeaderLine(), "\n");
}

void TsvWriter::WriteRow(const TsvRow& tsv_row) {
  CHECK_EQ(spec_, tsv_row.spec());
  file_ << absl::StrCat(tsv_row.ToString(), "\n");
}

TsvWriter::~TsvWriter() { file_.close(); }

}  // namespace utils_tsv
