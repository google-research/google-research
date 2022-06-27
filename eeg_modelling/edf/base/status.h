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

#ifndef EEG_MODELLING_BASE_STATUS_H_
#define EEG_MODELLING_BASE_STATUS_H_

#include <atomic>
#include <iosfwd>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/utility/utility.h"

namespace eeg_modelling {

enum class StatusCode {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kUnauthenticated = 16,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kDoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

absl::string_view StatusCodeToString(StatusCode code);

std::ostream& operator<<(std::ostream& out, StatusCode code);

class ABSL_MUST_USE_RESULT Status;

class Status final {
 public:
  // Creates an OK status with no message.
  Status() {}

  // Creates a status with the specified code and message.
  //
  // If `code == StatusCode::kOk`, `message` is ignored and an object identical
  // to an OK status is constructed.
  //
  // `message` must be in UTF-8.
  Status(StatusCode code, absl::string_view message);

  Status(const Status& that) noexcept;
  Status& operator=(const Status& that) noexcept;

  Status(Status&& that) noexcept;
  Status& operator=(Status&& that) noexcept;

  ~Status();

  // Returns true if the Status is OK.
  ABSL_MUST_USE_RESULT bool ok() const;

  // Returns the error code.
  StatusCode code() const;

  // Returns the message. Note: prefer StatusToString() for debug logging.
  // This message rarely describes the error code. It is not unusual for the
  // message to be the empty string.
  absl::string_view message() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  friend bool operator==(const Status&, const Status&);
  friend bool operator!=(const Status&, const Status&);

  // Swap the contents of `a` with `b`
  friend void swap(Status& a, Status& b);

 private:
  friend std::string StatusToString(const Status& status);

  // Reference-counted representation.
  struct Rep {
    Rep(StatusCode code, absl::string_view message);

    void Ref();
    void Unref();

    std::atomic<size_t> ref_count{1};
    StatusCode code;
    std::string message;
  };

  static bool EqualsSlow(const Status& a, const Status& b);

  std::string ToStringSlow() const;

  // nullptr: status is OK. Otherwise a pointer to a representation.
  Rep* rep_ = nullptr;
};

// Returns an OK status, equivalent to a default constructed instance.
Status OkStatus();

Status UnknownStatus();

// Returns a Status that is identical to `status` except that the message() has
// been augmented by adding `message` to the end of the original message.
//
// Annotate() adds the appropriate separators, so callers should not include a
// separator in `message`. The exact formatting is subject to change, so you
// should not depend on it in your tests.
//
// OK status values have no message and therefore if `status` is OK, the result
// is unchanged.
Status Annotate(const Status& status, absl::string_view message);

// Returns a human-readable representation of `status`.
std::string StatusToString(const Status& status);

// Prints a human-readable representation of `status` to `out`.
std::ostream& operator<<(std::ostream& out, const Status& status);

// Implementation details follow.

inline void Status::Rep::Ref() {
  ref_count.fetch_add(1, std::memory_order_relaxed);
}

inline Status::Status(const Status& that) noexcept : rep_(that.rep_) {
  if (rep_ != nullptr) rep_->Ref();
}

inline Status& Status::operator=(const Status& that) noexcept {
  Rep* const old_rep = rep_;
  rep_ = that.rep_;
  if (rep_ != nullptr) rep_->Ref();
  if (old_rep != nullptr) old_rep->Unref();
  return *this;
}

inline Status::Status(Status&& that) noexcept
    : rep_(absl::exchange(that.rep_, nullptr)) {}

inline Status& Status::operator=(Status&& that) noexcept {
  Rep* const old_rep = rep_;
  rep_ = absl::exchange(that.rep_, nullptr);
  if (old_rep != nullptr) old_rep->Unref();
  return *this;
}

inline Status::~Status() {
  if (rep_ != nullptr) rep_->Unref();
}

inline bool Status::ok() const { return rep_ == nullptr; }

inline StatusCode Status::code() const {
  if (rep_ == nullptr) return StatusCode::kOk;
  return rep_->code;
}

inline absl::string_view Status::message() const {
  if (rep_ == nullptr) return absl::string_view();
  return rep_->message;
}

inline void Status::IgnoreError() const {}

inline bool operator==(const Status& a, const Status& b) {
  return a.rep_ == b.rep_ || Status::EqualsSlow(a, b);
}

inline bool operator!=(const Status& a, const Status& b) {
  return a.rep_ != b.rep_ && !Status::EqualsSlow(a, b);
}

inline void swap(Status& a, Status& b) {
  using std::swap;
  swap(a.rep_, b.rep_);
}

inline Status OkStatus() { return Status(); }

inline Status UnknownStatus() { return Status(StatusCode::kUnknown, ""); }

inline std::string StatusToString(const Status& status) {
  if (status.ok()) return "OK";
  return status.ToStringSlow();
}

}  // namespace eeg_modelling

#endif  // EEG_MODELLING_BASE_STATUS_H_
