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

#include "edf/base/status.h"

#include <atomic>
#include <ostream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace eeg_modelling {

absl::string_view StatusCodeToString(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kCancelled:
      return "CANCELLED";
    case StatusCode::kUnknown:
      return "UNKNOWN";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case StatusCode::kDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kAlreadyExists:
      return "ALREADY_EXISTS";
    case StatusCode::kPermissionDenied:
      return "PERMISSION_DENIED";
    case StatusCode::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::kFailedPrecondition:
      return "FAILED_PRECONDITION";
    case StatusCode::kAborted:
      return "ABORTED";
    case StatusCode::kOutOfRange:
      return "OUT_OF_RANGE";
    case StatusCode::kUnimplemented:
      return "UNIMPLEMENTED";
    case StatusCode::kInternal:
      return "INTERNAL";
    case StatusCode::kUnavailable:
      return "UNAVAILABLE";
    case StatusCode::kDataLoss:
      return "DATA_LOSS";
    case StatusCode::kUnauthenticated:
      return "UNAUTHENTICATED";
    default:
      return "";
  }
}

std::ostream& operator<<(std::ostream& out, StatusCode code) {
  return out << StatusCodeToString(code);
}

inline Status::Rep::Rep(StatusCode code, absl::string_view message)
    : code(code), message(message) {}

void Status::Rep::Unref() {
  if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
}

Status::Status(StatusCode code, absl::string_view message) {
  if (code != StatusCode::kOk) rep_ = new Rep(code, message);
}

bool Status::EqualsSlow(const Status& a, const Status& b) {
  if (a.rep_ == nullptr || b.rep_ == nullptr) return false;
  return a.rep_->code == b.rep_->code && a.rep_->message == b.rep_->message;
}

std::string Status::ToStringSlow() const {
  return absl::StrCat(StatusCodeToString(rep_->code), ": ", rep_->message);
}

Status Annotate(const Status& status, absl::string_view message) {
  if (status.ok() || message.empty()) return status;
  if (status.message().empty()) return Status(status.code(), message);
  return Status(status.code(), absl::StrCat(status.message(), "; ", message));
}

std::ostream& operator<<(std::ostream& out, const Status& status) {
  return out << StatusToString(status);
}

}  // namespace eeg_modelling
