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

#include "edf/edf_oss_file.h"

#include <memory>

#include "absl/base/internal/raw_logging.h"
#include "edf/base/status.h"

namespace eeg_modelling {

EdfOssFile::EdfOssFile(FILE* fp) : fp_(fp) {}

EdfOssFile::~EdfOssFile() { fclose(fp_); }

EdfOssFile::EdfOssFile(const char* filename, const char* mode) {
  fp_ = fopen(filename, mode);
  if (fp_ == nullptr) {
    ABSL_RAW_LOG(FATAL, "File open failed %s", filename);
  }
}

size_t EdfOssFile::Tell() const { return std::ftell(fp_); }

Status EdfOssFile::SeekFromBegin(size_t position) {
  auto ret = std::fseek(fp_, position, SEEK_SET);
  if (ret != 0) {
    return Status(StatusCode::kUnknown, "Seek failed");
  }
  return OkStatus();
}

size_t EdfOssFile::Read(void* ptr, size_t n) const {
  return std::fread(ptr, 1, n, fp_);
}

size_t EdfOssFile::ReadToString(string* str, size_t n) const {
  std::unique_ptr<char[]> buf(new char[n]);
  auto bytes_read = std::fread(buf.get(), 1, n, fp_);
  if (bytes_read <= 0) {
    return bytes_read;
  }
  str->assign(buf.get(), bytes_read);
  return bytes_read;
}

size_t EdfOssFile::Write(const void* ptr, size_t n) {
  return std::fwrite(ptr, 1, n, fp_);
}

}  // namespace eeg_modelling
