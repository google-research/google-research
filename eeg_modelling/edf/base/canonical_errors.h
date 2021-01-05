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

#ifndef EEG_MODELLING_BASE_CANONICAL_ERRORS_H_
#define EEG_MODELLING_BASE_CANONICAL_ERRORS_H_

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "edf/base/status.h"

namespace eeg_modelling {

// Each of the functions below creates a canonical error with the given
// message. The error code of the returned status object matches the name of
// the function.
Status AbortedError(absl::string_view message);
Status AlreadyExistsError(absl::string_view message);
Status CancelledError(absl::string_view message);
Status DataLossError(absl::string_view message);
Status DeadlineExceededError(absl::string_view message);
Status FailedPreconditionError(absl::string_view message);
Status InternalError(absl::string_view message);
Status InvalidArgumentError(absl::string_view message);
Status NotFoundError(absl::string_view message);
Status OutOfRangeError(absl::string_view message);
Status PermissionDeniedError(absl::string_view message);
Status ResourceExhaustedError(absl::string_view message);
Status UnauthenticatedError(absl::string_view message);
Status UnavailableError(absl::string_view message);
Status UnimplementedError(absl::string_view message);
Status UnknownError(absl::string_view message);

// Creates a canonical error with the CANCELLED error code and an empty message.
inline Status CancelledError() { return CancelledError(""); }

// Each of the functions below returns true if the given status matches the
// canonical error code implied by the function's name. If necessary, the
// status will be converted to the canonical error space to perform the
// comparison.
ABSL_MUST_USE_RESULT bool IsAborted(const Status& status);
ABSL_MUST_USE_RESULT bool IsAlreadyExists(const Status& status);
ABSL_MUST_USE_RESULT bool IsCancelled(const Status& status);
ABSL_MUST_USE_RESULT bool IsDataLoss(const Status& status);
ABSL_MUST_USE_RESULT bool IsDeadlineExceeded(const Status& status);
ABSL_MUST_USE_RESULT bool IsFailedPrecondition(const Status& status);
ABSL_MUST_USE_RESULT bool IsInternal(const Status& status);
ABSL_MUST_USE_RESULT bool IsInvalidArgument(const Status& status);
ABSL_MUST_USE_RESULT bool IsNotFound(const Status& status);
ABSL_MUST_USE_RESULT bool IsOutOfRange(const Status& status);
ABSL_MUST_USE_RESULT bool IsPermissionDenied(const Status& status);
ABSL_MUST_USE_RESULT bool IsResourceExhausted(const Status& status);
ABSL_MUST_USE_RESULT bool IsUnauthenticated(const Status& status);
ABSL_MUST_USE_RESULT bool IsUnavailable(const Status& status);
ABSL_MUST_USE_RESULT bool IsUnimplemented(const Status& status);
ABSL_MUST_USE_RESULT bool IsUnknown(const Status& status);

}  // namespace eeg_modelling
#endif  // EEG_MODELLING_BASE_CANONICAL_ERRORS_H_
