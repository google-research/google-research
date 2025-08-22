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

#ifndef SCANN_OSS_WRAPPERS_SCANN_STATUS_H_
#define SCANN_OSS_WRAPPERS_SCANN_STATUS_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "scann/oss_wrappers/scann_status_builder.h"

#define SCANN_RET_CHECK(cond)         \
  while (ABSL_PREDICT_FALSE(!(cond))) \
  return RetCheckFail(                \
      "SCANN_RET_CHECK "              \
      "failure ")

#define SCANN_RET_CHECK_EQ(lhs, rhs)         \
  while (ABSL_PREDICT_FALSE((lhs) != (rhs))) \
  return RetCheckFail("SCANN_RET_CHECK_EQ failure ")

#define SCANN_RET_CHECK_NE(lhs, rhs)         \
  while (ABSL_PREDICT_FALSE((lhs) == (rhs))) \
  return RetCheckFail("SCANN_RET_CHECK_NE failure ")

#define SCANN_RET_CHECK_GE(lhs, rhs)            \
  while (ABSL_PREDICT_FALSE(!((lhs) >= (rhs)))) \
  return RetCheckFail("SCANN_RET_CHECK_GE failure ")

#define SCANN_RET_CHECK_LE(lhs, rhs)            \
  while (ABSL_PREDICT_FALSE(!((lhs) <= (rhs)))) \
  return RetCheckFail("SCANN_RET_CHECK_LE failure ")

#define SCANN_RET_CHECK_GT(lhs, rhs)           \
  while (ABSL_PREDICT_FALSE(!((lhs) > (rhs)))) \
  return RetCheckFail("SCANN_RET_CHECK_GT failure ")

#define SCANN_RET_CHECK_LT(lhs, rhs)           \
  while (ABSL_PREDICT_FALSE(!((lhs) < (rhs)))) \
  return RetCheckFail("SCANN_RET_CHECK_LT failure ")

#define SCANN_RETURN_IF_ERROR(expr)                      \
  for (auto __return_if_error_res = (expr);              \
       ABSL_PREDICT_FALSE(!__return_if_error_res.ok());) \
  return StatusBuilder(__return_if_error_res)

#define SCANN_LOG_NOOP(...) \
  while (false) LOG(ERROR)

#define VLOG SCANN_LOG_NOOP
#define DVLOG SCANN_LOG_NOOP

#define SCANN_ASSIGN_OR_RETURN(lhs, rexpr) \
  SCANN_ASSIGN_OR_RETURN_IMPL(             \
      SCANN_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define SCANN_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {               \
    return statusor.status();                             \
  }                                                       \
  lhs = std::move(statusor).value()

#define SCANN_MACROS_CONCAT_NAME(x, y) SCANN_MACROS_CONCAT_IMPL(x, y)
#define SCANN_MACROS_CONCAT_IMPL(x, y) x##y

namespace research_scann {

absl::Status AnnotateStatus(const absl::Status& s, absl::string_view msg);

StatusBuilder RetCheckFail(absl::string_view msg);

}  // namespace research_scann

#endif
