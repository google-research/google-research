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

#include "scann/oss_wrappers/scann_status.h"

#include "absl/strings/str_cat.h"

namespace research_scann {

Status AnnotateStatus(const Status& s, absl::string_view msg) {
  if (s.ok() || msg.empty()) return s;

  absl::string_view new_msg = msg;
  std::string annotated;
  if (!s.error_message().empty()) {
    absl::StrAppend(&annotated, s.error_message(), "; ", msg);
    new_msg = annotated;
  }
  return Status(s.code(), new_msg);
}

StatusBuilder RetCheckFail(absl::string_view msg) {
  return InternalErrorBuilder() << msg;
}

}  // namespace research_scann
