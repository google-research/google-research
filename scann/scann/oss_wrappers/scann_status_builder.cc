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

#include "scann/oss_wrappers/scann_status_builder.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace research_scann {

StatusBuilder::StatusBuilder(const absl::Status& status) : status_(status) {}

StatusBuilder::StatusBuilder(absl::Status&& status) : status_(status) {}

StatusBuilder::StatusBuilder(absl::StatusCode code) : status_(code, "") {}

StatusBuilder::StatusBuilder(const StatusBuilder& sb) : status_(sb.status_) {
  if (sb.streamptr_ != nullptr) {
    streamptr_ = std::make_unique<std::ostringstream>(sb.streamptr_->str());
  }
}

absl::Status StatusBuilder::CreateStatus() && {
  auto result = [&] {
    if (streamptr_->str().empty()) return status_;
    std::string new_msg =
        absl::StrCat(status_.message(), "; ", streamptr_->str());
    return absl::Status(status_.code(), new_msg);
  }();
  status_ = absl::UnknownError("");
  streamptr_ = nullptr;
  return result;
}

StatusBuilder& StatusBuilder::LogError() & { return *this; }
StatusBuilder&& StatusBuilder::LogError() && { return std::move(LogError()); }

StatusBuilder::operator absl::Status() const& {
  if (streamptr_ == nullptr) return status_;
  return StatusBuilder(*this).CreateStatus();
}

StatusBuilder::operator absl::Status() && {
  if (streamptr_ == nullptr) return status_;
  return std::move(*this).CreateStatus();
}

StatusBuilder AbortedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kAborted);
}
StatusBuilder AlreadyExistsErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kAlreadyExists);
}
StatusBuilder CancelledErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kCancelled);
}
StatusBuilder FailedPreconditionErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kFailedPrecondition);
}
StatusBuilder InternalErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kInternal);
}
StatusBuilder InvalidArgumentErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kInvalidArgument);
}
StatusBuilder NotFoundErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kNotFound);
}
StatusBuilder OutOfRangeErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kOutOfRange);
}
StatusBuilder UnauthenticatedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnauthenticated);
}
StatusBuilder UnavailableErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnavailable);
}
StatusBuilder UnimplementedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnimplemented);
}
StatusBuilder UnknownErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnknown);
}

}  // namespace research_scann
