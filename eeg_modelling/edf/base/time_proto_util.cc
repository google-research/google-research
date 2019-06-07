#include "edf/base/time_proto_util.h"

#include "google/protobuf/timestamp.pb.h"
#include <cstdint>
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "edf/base/status.h"
#include "edf/base/statusor.h"

namespace eeg_modelling {

namespace {
// Validation requirements documented at:
Status Validate(const google::protobuf::Timestamp& t) {
  const auto sec = t.seconds();
  const auto ns = t.nanos();
  // sec must be [0001-01-01T00:00:00Z, 9999-12-31T23:59:59.999999999Z]
  if (sec < -62135596800 || sec > 253402300799) {
    return Status(StatusCode::kInvalidArgument, absl::StrCat("seconds=", sec));
  }
  if (ns < 0 || ns > 999999999) {
    return Status(StatusCode::kInvalidArgument, absl::StrCat("nanos=", ns));
  }
  return OkStatus();
}

Status EncodeGoogleApiProto(absl::Time t, google::protobuf::Timestamp* proto) {
  const int64_t s = absl::ToUnixSeconds(t);
  proto->set_seconds(s);
  proto->set_nanos((t - absl::FromUnixSeconds(s)) / absl::Nanoseconds(1));
  return Validate(*proto);
}
}  // namespace

StatusOr<google::protobuf::Timestamp> EncodeGoogleApiProto(absl::Time t) {
  google::protobuf::Timestamp proto;
  Status status = EncodeGoogleApiProto(t, &proto);
  if (!status.ok()) return status;
  return proto;
}

StatusOr<absl::Time> DecodeGoogleApiProto(
    const google::protobuf::Timestamp& proto) {
  Status status = Validate(proto);
  if (!status.ok()) return status;
  return absl::FromUnixSeconds(proto.seconds()) +
         absl::Nanoseconds(proto.nanos());
}
}  // namespace eeg_modelling
