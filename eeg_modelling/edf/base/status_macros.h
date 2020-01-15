#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_BASE_STATUS_MACROS_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_BASE_STATUS_MACROS_H_

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

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_BASE_STATUS_MACROS_H_
