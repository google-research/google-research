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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__OSS_WRAPPERS_SCANN_STATUS_BUILDER_H_
#define SCANN__OSS_WRAPPERS_SCANN_STATUS_BUILDER_H_

#include <sstream>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace scann_ops {
namespace internal {

using ::stream_executor::port::StatusOr;

}

class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  explicit StatusBuilder(const Status& status);
  explicit StatusBuilder(Status&& status);
  explicit StatusBuilder(tensorflow::error::Code code);
  StatusBuilder(const StatusBuilder& sb);

  template <typename T>
  StatusBuilder& operator<<(const T& value) & {
    if (status_.ok()) return *this;
    if (streamptr_ == nullptr)
      streamptr_ = absl::make_unique<std::ostringstream>();
    *streamptr_ << value;
    return *this;
  }

  template <typename T>
  StatusBuilder&& operator<<(const T& value) && {
    return std::move(operator<<(value));
  }

  StatusBuilder& LogError() &;
  StatusBuilder&& LogError() &&;

  operator Status() const&;
  operator Status() &&;

  template <typename T>
  inline operator internal::StatusOr<T>() const& {
    if (streamptr_ == nullptr) return internal::StatusOr<T>(status_);
    return internal::StatusOr<T>(StatusBuilder(*this).CreateStatus());
  }

  template <typename T>
  inline operator internal::StatusOr<T>() && {
    if (streamptr_ == nullptr) return internal::StatusOr<T>(status_);
    return internal::StatusOr<T>(StatusBuilder(*this).CreateStatus());
  }

  template <typename Enum>
  StatusBuilder& SetErrorCode(Enum code) & {
    status_ = Status(code, status_.error_message());
    return *this;
  }

  template <typename Enum>
  StatusBuilder&& SetErrorCode(Enum code) && {
    return std::move(SetErrorCode(code));
  }

  Status CreateStatus() &&;

 private:
  std::unique_ptr<std::ostringstream> streamptr_;

  Status status_;
};

StatusBuilder AbortedErrorBuilder();
StatusBuilder AlreadyExistsErrorBuilder();
StatusBuilder CancelledErrorBuilder();
StatusBuilder FailedPreconditionErrorBuilder();
StatusBuilder InternalErrorBuilder();
StatusBuilder InvalidArgumentErrorBuilder();
StatusBuilder NotFoundErrorBuilder();
StatusBuilder OutOfRangeErrorBuilder();
StatusBuilder UnauthenticatedErrorBuilder();
StatusBuilder UnavailableErrorBuilder();
StatusBuilder UnimplementedErrorBuilder();
StatusBuilder UnknownErrorBuilder();

}  // namespace scann_ops
}  // namespace tensorflow

#endif
