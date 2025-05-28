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



#ifndef SCANN_DATA_FORMAT_INTERNAL_STRING_VIEW32_H_
#define SCANN_DATA_FORMAT_INTERNAL_STRING_VIEW32_H_

#include "scann/utils/types.h"

namespace research_scann {
namespace data_format_internal {

class string_view32 {
 public:
  string_view32(const char* ptr, uint32_t length)
      : ptr_(ptr), length_(length) {}
  explicit string_view32(string_view s) : string_view32(s.data(), s.length()) {
    CHECK_LE(s.length(), numeric_limits<uint32_t>::max());
  }

  bool empty() const { return length_ == 0; }

  explicit operator string_view() const {
    return string_view(static_cast<const char*>(ptr_), length_);
  }
  bool operator==(string_view32 rhs) const {
    return static_cast<string_view>(*this) == static_cast<string_view>(rhs);
  }

  struct Hash {
    size_t operator()(string_view32 s) const {
      return absl::Hash<string_view>()(static_cast<string_view>(s));
    }
  };

 private:
  const char* ptr_ = nullptr;

  uint32_t length_ = 0;
};

}  // namespace data_format_internal
}  // namespace research_scann

#endif
