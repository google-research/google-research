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

#ifndef EEG_MODELLING_BASE_STATUS_MACROS_H_
#define EEG_MODELLING_BASE_STATUS_MACROS_H_

#include "absl/base/optimization.h"

// Macros below are a limited adaptation of //util/task/status_macros.h
// until absl::Status is opensourced.
#define RETURN_IF_ERROR(expr)                          \
  do {                                                 \
    const auto _status_to_verify = (expr);             \
    if (ABSL_PREDICT_FALSE(!_status_to_verify.ok())) { \
      return _status_to_verify;                        \
    }                                                  \
  } while (false)

#define ASSIGN_OR_RETURN(lhs, rexpr)                  \
  do {                                                \
    auto _status_or_value = (rexpr);                  \
    if (ABSL_PREDICT_FALSE(!_status_or_value.ok())) { \
      return _status_or_value.status();               \
    }                                                 \
    lhs = std::move(_status_or_value).ValueOrDie();   \
  } while (false)

#endif  // EEG_MODELLING_BASE_STATUS_MACROS_H_
