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

#ifndef UTIL_TIME_PROTOUTIL_H_
#define UTIL_TIME_PROTOUTIL_H_

#include "google/protobuf/timestamp.pb.h"
#include "absl/time/time.h"
#include "edf/base/status.h"
#include "edf/base/statusor.h"

namespace eeg_modelling {

// Encodes an absl::Time as a google::protobuf::Timestamp.
StatusOr<google::protobuf::Timestamp> EncodeGoogleApiProto(absl::Time t);

// Decodes the given protobuf and returns an absl::Time, or returns an error
// status if the argument is invalid according to
// google/protobuf/timestamp.proto.
StatusOr<absl::Time> DecodeGoogleApiProto(
    const google::protobuf::Timestamp& proto);

}  // namespace eeg_modelling

#endif  // UTIL_TIME_PROTOUTIL_H_
