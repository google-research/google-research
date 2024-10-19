// Copyright 2024 The Google Research Authors.
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

#include "edf/base/statusor.h"

#include <errno.h>

namespace eeg_modelling {

::eeg_modelling::Status internal::StatusOrHelper::HandleInvalidStatusCtorArg() {
  ABSL_RAW_LOG(FATAL,
               "Status::OK is not a valid constructor argument to StatusOr<T>");
  // Workaround.
  return OkStatus();
}

::eeg_modelling::Status internal::StatusOrHelper::HandleNullObjectCtorArg() {
  ABSL_RAW_LOG(FATAL,
               "NULL is not a valid constructor argument to StatusOr<T*>");
  // Workaround.
  return OkStatus();
}

void internal::StatusOrHelper::Crash(const Status& status) {
  ABSL_RAW_LOG(FATAL,
               "Attempting to fetch value instead of handling error status");
}

}  // namespace eeg_modelling
