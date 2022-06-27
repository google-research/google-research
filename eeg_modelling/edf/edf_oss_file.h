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

#ifndef EEG_MODELLING_EDF_EDF_OSS_FILE_H_
#define EEG_MODELLING_EDF_EDF_OSS_FILE_H_

#include <string>

#include "absl/strings/string_view.h"
#include "edf/base/status.h"
#include "edf/edf_file.h"

using std::string;

namespace eeg_modelling {

class EdfOssFile : public EdfFile {
 public:
  EdfOssFile(FILE* fp);

  EdfOssFile(const char* filename, const char* mode);

  ~EdfOssFile();

  size_t Tell() const;

  Status SeekFromBegin(size_t position);

  size_t Read(void* ptr, size_t n) const;

  size_t ReadToString(string* str, size_t n) const;

  size_t Write(const void* ptr, size_t n);

 private:
  FILE* fp_;
};

}  // namespace eeg_modelling

#endif  // EEG_MODELLING_EDF_EDF_OSS_FILE_H_
