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

#ifndef TSV_UTILS_H_
#define TSV_UTILS_H_

#include <fstream>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

namespace utils_tsv {

class TsvRow;

class TsvSpec {
 public:
  explicit TsvSpec(const std::vector<std::string>& field_names,
                   char delim = '\t')
      : field_names_(field_names), delim_(delim) {}

  // Creates a TsvRow using this TsvSpec. Note that the TsvSpec must outlive the
  // created row.
  TsvRow CreateRow() const;

  std::string GetHeaderLine() const;

  bool HasField(const std::string& field) const {
    for (const auto& f : field_names_) {
      if (f == field) return true;
    }
    return false;
  }

  const std::vector<std::string>& field_names() const { return field_names_; }
  char delim() const { return delim_; }

 private:
  std::vector<std::string> field_names_;
  char delim_;
};

class TsvRow {
 public:
  explicit TsvRow(const TsvSpec* spec) : spec_(spec) {}
  TsvRow(const TsvSpec* spec,
         const absl::flat_hash_map<std::string, std::string>& field_values)
      : spec_(spec), field_values_(field_values) {}

  // Add a value to the row. CHECK fails if the column is not in the spec.
  template <typename T>
  void Add(const std::string& field, const T& value) {
    CHECK(spec_->HasField(field))
        << "Field " << field << " is not in TSV spec!";
    field_values_[field] = absl::StrCat(value);
  }

  // Same as above, but skips checking whether the column is in the spec.
  template <typename T>
  void AddUnvalidated(const std::string& field, const T& value) {
    field_values_[field] = absl::StrCat(value);
  }

  std::string ToString() const;
  const TsvSpec* spec() const { return spec_; }

 private:
  const TsvSpec* spec_;
  absl::flat_hash_map<std::string, std::string> field_values_;
};

class TsvReader {
 public:
  explicit TsvReader(const std::string& file_path, char delim = '\t');

  bool AtEnd() { return file_.peek() == EOF; }

  absl::flat_hash_map<std::string, std::string> ReadRow();

  const std::vector<std::string>& field_names() const { return field_names_; }

 private:
  std::vector<std::string> GetValuesFrom(const std::string& line);

  const char delim_;
  std::vector<std::string> field_names_;
  std::ifstream file_;
};

class TsvWriter {
 public:
  TsvWriter(const std::string& file_path, const TsvSpec* spec);

  void WriteRow(const TsvRow& tsv_row);

  ~TsvWriter();

 private:
  const TsvSpec* spec_;
  std::ofstream file_;
};

}  // namespace utils_tsv

#endif  // TSV_UTILS_H_
