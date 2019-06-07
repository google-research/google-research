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
