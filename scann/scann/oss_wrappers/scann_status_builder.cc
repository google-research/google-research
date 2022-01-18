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

#include "scann/oss_wrappers/scann_status_builder.h"

namespace research_scann {

StatusBuilder::StatusBuilder(const Status& status) : status_(status) {}

StatusBuilder::StatusBuilder(Status&& status) : status_(status) {}

StatusBuilder::StatusBuilder(tensorflow::error::Code code)
    : status_(code, "") {}

StatusBuilder::StatusBuilder(const StatusBuilder& sb) : status_(sb.status_) {
  if (sb.streamptr_ != nullptr) {
    streamptr_ = absl::make_unique<std::ostringstream>(sb.streamptr_->str());
  }
}

Status StatusBuilder::CreateStatus() && {
  auto result = [&] {
    if (streamptr_->str().empty()) return status_;
    std::string new_msg =
        absl::StrCat(status_.error_message(), "; ", streamptr_->str());
    return Status(status_.code(), new_msg);
  }();
  status_ = errors::Unknown("");
  streamptr_ = nullptr;
  return result;
}

StatusBuilder& StatusBuilder::LogError() & { return *this; }
StatusBuilder&& StatusBuilder::LogError() && { return std::move(LogError()); }

StatusBuilder::operator Status() const& {
  if (streamptr_ == nullptr) return status_;
  return StatusBuilder(*this).CreateStatus();
}

StatusBuilder::operator Status() && {
  if (streamptr_ == nullptr) return status_;
  return std::move(*this).CreateStatus();
}

StatusBuilder AbortedErrorBuilder() { return StatusBuilder(error::ABORTED); }
StatusBuilder AlreadyExistsErrorBuilder() {
  return StatusBuilder(error::ALREADY_EXISTS);
}
StatusBuilder CancelledErrorBuilder() {
  return StatusBuilder(error::CANCELLED);
}
StatusBuilder FailedPreconditionErrorBuilder() {
  return StatusBuilder(error::FAILED_PRECONDITION);
}
StatusBuilder InternalErrorBuilder() { return StatusBuilder(error::INTERNAL); }
StatusBuilder InvalidArgumentErrorBuilder() {
  return StatusBuilder(error::INVALID_ARGUMENT);
}
StatusBuilder NotFoundErrorBuilder() { return StatusBuilder(error::NOT_FOUND); }
StatusBuilder OutOfRangeErrorBuilder() {
  return StatusBuilder(error::OUT_OF_RANGE);
}
StatusBuilder UnauthenticatedErrorBuilder() {
  return StatusBuilder(error::UNAUTHENTICATED);
}
StatusBuilder UnavailableErrorBuilder() {
  return StatusBuilder(error::UNAVAILABLE);
}
StatusBuilder UnimplementedErrorBuilder() {
  return StatusBuilder(error::UNIMPLEMENTED);
}
StatusBuilder UnknownErrorBuilder() { return StatusBuilder(error::UNKNOWN); }

}  // namespace research_scann
