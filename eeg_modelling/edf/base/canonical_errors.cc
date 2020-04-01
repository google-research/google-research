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

#include "edf/base/canonical_errors.h"

#include "absl/strings/string_view.h"
#include "edf/base/status.h"

namespace eeg_modelling {

Status AbortedError(absl::string_view message) {
  return Status(StatusCode::kAborted, message);
}

Status AlreadyExistsError(absl::string_view message) {
  return Status(StatusCode::kAlreadyExists, message);
}

Status CancelledError(absl::string_view message) {
  return Status(StatusCode::kCancelled, message);
}

Status DataLossError(absl::string_view message) {
  return Status(StatusCode::kDataLoss, message);
}

Status DeadlineExceededError(absl::string_view message) {
  return Status(StatusCode::kDeadlineExceeded, message);
}

Status FailedPreconditionError(absl::string_view message) {
  return Status(StatusCode::kFailedPrecondition, message);
}

Status InternalError(absl::string_view message) {
  return Status(StatusCode::kInternal, message);
}

Status InvalidArgumentError(absl::string_view message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status NotFoundError(absl::string_view message) {
  return Status(StatusCode::kNotFound, message);
}

Status OutOfRangeError(absl::string_view message) {
  return Status(StatusCode::kOutOfRange, message);
}

Status PermissionDeniedError(absl::string_view message) {
  return Status(StatusCode::kPermissionDenied, message);
}

Status ResourceExhaustedError(absl::string_view message) {
  return Status(StatusCode::kResourceExhausted, message);
}

Status UnauthenticatedError(absl::string_view message) {
  return Status(StatusCode::kUnauthenticated, message);
}

Status UnavailableError(absl::string_view message) {
  return Status(StatusCode::kUnavailable, message);
}

Status UnimplementedError(absl::string_view message) {
  return Status(StatusCode::kUnimplemented, message);
}

Status UnknownError(absl::string_view message) {
  return Status(StatusCode::kUnknown, message);
}

bool IsAborted(const Status& status) {
  return status.code() == StatusCode::kAborted;
}

bool IsAlreadyExists(const Status& status) {
  return status.code() == StatusCode::kAlreadyExists;
}

bool IsCancelled(const Status& status) {
  return status.code() == StatusCode::kCancelled;
}

bool IsDataLoss(const Status& status) {
  return status.code() == StatusCode::kDataLoss;
}

bool IsDeadlineExceeded(const Status& status) {
  return status.code() == StatusCode::kDeadlineExceeded;
}

bool IsFailedPrecondition(const Status& status) {
  return status.code() == StatusCode::kFailedPrecondition;
}

bool IsInternal(const Status& status) {
  return status.code() == StatusCode::kInternal;
}

bool IsInvalidArgument(const Status& status) {
  return status.code() == StatusCode::kInvalidArgument;
}

bool IsNotFound(const Status& status) {
  return status.code() == StatusCode::kNotFound;
}

bool IsOutOfRange(const Status& status) {
  return status.code() == StatusCode::kOutOfRange;
}

bool IsPermissionDenied(const Status& status) {
  return status.code() == StatusCode::kPermissionDenied;
}

bool IsResourceExhausted(const Status& status) {
  return status.code() == StatusCode::kResourceExhausted;
}

bool IsUnauthenticated(const Status& status) {
  return status.code() == StatusCode::kUnauthenticated;
}

bool IsUnavailable(const Status& status) {
  return status.code() == StatusCode::kUnavailable;
}

bool IsUnimplemented(const Status& status) {
  return status.code() == StatusCode::kUnimplemented;
}

bool IsUnknown(const Status& status) {
  return status.code() == StatusCode::kUnknown;
}

}  // namespace eeg_modelling
