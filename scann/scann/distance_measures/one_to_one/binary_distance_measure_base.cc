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

#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"

#include <cstdint>

namespace research_scann {

#define LOG_FATAL_CRASH_OK LOG(FATAL)

#define SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(T)               \
  double BinaryDistanceMeasureBase::GetDistanceDense(                       \
      const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {           \
    LOG_FATAL_CRASH_OK << "Binary distance measures don't support " #T      \
                          " data.";                                         \
  }                                                                         \
  double BinaryDistanceMeasureBase::GetDistanceDense(                       \
      const DatapointPtr<T>& a, const DatapointPtr<T>& b, double threshold) \
      const {                                                               \
    LOG_FATAL_CRASH_OK << "Binary distance measures don't support " #T      \
                          " data.";                                         \
  }                                                                         \
  double BinaryDistanceMeasureBase::GetDistanceSparse(                      \
      const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {           \
    LOG_FATAL_CRASH_OK << "Binary distance measures don't support " #T      \
                          " data.";                                         \
  }                                                                         \
  double BinaryDistanceMeasureBase::GetDistanceHybrid(                      \
      const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {           \
    LOG_FATAL_CRASH_OK << "Binary distance measures don't support " #T      \
                          " data.";                                         \
  }

SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(int8_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(int16_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(uint16_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(int32_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(uint32_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(int64_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(uint64_t);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(float);
SCANN_DEFINE_BINARY_DISTANCE_METHODS_UNIMPLEMENTED(double);

}  // namespace research_scann
