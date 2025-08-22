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

#ifndef SCANN_OSS_WRAPPERS_SCANN_STATUS_BUILDER_H_
#define SCANN_OSS_WRAPPERS_SCANN_STATUS_BUILDER_H_

#include <memory>
#include <sstream>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace research_scann {

class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  explicit StatusBuilder(const absl::Status& status);
  explicit StatusBuilder(absl::Status&& status);
  explicit StatusBuilder(absl::StatusCode code);
  StatusBuilder(const StatusBuilder& sb);

  template <typename T>
  StatusBuilder& operator<<(const T& value) & {
    if (status_.ok()) return *this;
    if (streamptr_ == nullptr)
      streamptr_ = std::make_unique<std::ostringstream>();
    *streamptr_ << value;
    return *this;
  }

  template <typename T>
  StatusBuilder&& operator<<(const T& value) && {
    return std::move(operator<<(value));
  }

  StatusBuilder& LogError() &;
  StatusBuilder&& LogError() &&;

  operator absl::Status() const&;
  operator absl::Status() &&;

  template <typename T>
  inline operator absl::StatusOr<T>() const& {
    if (streamptr_ == nullptr) return absl::StatusOr<T>(status_);
    return absl::StatusOr<T>(StatusBuilder(*this).CreateStatus());
  }

  template <typename T>
  inline operator absl::StatusOr<T>() && {
    if (streamptr_ == nullptr) return absl::StatusOr<T>(status_);
    return absl::StatusOr<T>(StatusBuilder(*this).CreateStatus());
  }

  inline StatusBuilder& SetCode(absl::StatusCode code) & {
    status_ = absl::Status(code, status_.message());
    return *this;
  }

  inline StatusBuilder&& SetCode(absl::StatusCode code) && {
    return std::move(SetCode(code));
  }

  absl::Status CreateStatus() &&;

 private:
  std::unique_ptr<std::ostringstream> streamptr_;

  absl::Status status_;
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

}  // namespace research_scann

#endif
