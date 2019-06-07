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
