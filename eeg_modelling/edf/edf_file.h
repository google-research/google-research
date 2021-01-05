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

#ifndef EEG_MODELLING_EDF_EDF_FILE_H_
#define EEG_MODELLING_EDF_EDF_FILE_H_

#include <string>

#include "edf/base/status.h"

namespace eeg_modelling {

// Interface to bridge open source and google internal file APIs.
class EdfFile {
 public:
  EdfFile();
  virtual ~EdfFile();

  // Tells the current position of the file, -1 in case
  virtual size_t Tell() const = 0;
  // Seek from the beginning of the file.
  virtual Status SeekFromBegin(size_t position) = 0;
  virtual size_t Read(void* ptr, size_t n) const = 0;
  virtual size_t ReadToString(std::string* str, size_t n) const = 0;
  virtual size_t Write(const void* ptr, size_t n) = 0;
};

}  // namespace eeg_modelling

#endif  // EEG_MODELLING_EDF_EDF_FILE_H_
