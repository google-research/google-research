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

#ifndef SCANN__OSS_WRAPPERS_SCANN_STATUS_H_
#define SCANN__OSS_WRAPPERS_SCANN_STATUS_H_

#include "scann/oss_wrappers/scann_status_builder.h"

#include "absl/strings/string_view.h"

#define SCANN_RET_CHECK(cond)         \
  while (ABSL_PREDICT_FALSE(!(cond))) \
  return RetCheckFail("SCANN_RET_CHECK failure ")

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

#define SCANN_RETURN_IF_ERROR(...) \
  while (ABSL_PREDICT_FALSE(!(__VA_ARGS__).ok())) return InternalErrorBuilder()

#define SCANN_LOG_NOOP(...) \
  while (false) LOG(ERROR)

namespace tensorflow {
namespace scann_ops {

Status AnnotateStatus(const Status& s, absl::string_view msg);

StatusBuilder RetCheckFail(absl::string_view msg);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
