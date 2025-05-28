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

#include "scann/oss_wrappers/scann_status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "scann/oss_wrappers/scann_status_builder.h"

namespace research_scann {

absl::Status AnnotateStatus(const absl::Status& s, absl::string_view msg) {
  if (s.ok() || msg.empty()) return s;

  absl::string_view new_msg = msg;
  std::string annotated;
  if (!s.message().empty()) {
    absl::StrAppend(&annotated, s.message(), "; ", msg);
    new_msg = annotated;
  }
  return absl::Status(s.code(), new_msg);
}

StatusBuilder RetCheckFail(absl::string_view msg) {
  return InternalErrorBuilder() << msg;
}

}  // namespace research_scann
