// Copyright 2022 The Google Research Authors.
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

#ifndef EEG_MODELLING_EDF_EDF_UTIL_H_
#define EEG_MODELLING_EDF_EDF_UTIL_H_

#include <tuple>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "edf/base/status.h"
#include "edf/base/statusor.h"
#include "edf/proto/edf.pb.h"

using std::string;
namespace eeg_modelling {

// Utilities for more easily navigating around the edf -------------------------

// First edf annotation signal indicates start of data record.
StatusOr<int> GetFirstEdfAnnotationSignalIndex(const Edf& edf);
bool HasEdfAnnotationSignal(const Edf& edf);
StatusOr<double> GetNumSecondsPerDataRecord(const EdfHeader& edf_header);
StatusOr<absl::Time> GetStartTime(absl::Time recording_start_time,
                                  const string& offset_start_sec_str);
StatusOr<std::tuple<absl::Time, absl::Time>> GetStartEndTimes(
    absl::Time recording_start_time, const string& offset_start_sec_str,
    const string& duration_sec_str);
StatusOr<absl::Time> ParseEdfStartTime(const EdfHeader& header);
StatusOr<absl::Time> GetEdfDataRecordStartTime(const Edf& edf, int data_record);

// Utilities for converting between different versions of EDF ------------------
Status ConvertFromEdfToEdfPlusC(Edf* edf);
StatusOr<bool> CanConvertToEdfPlusC(const Edf& edf);
Status ConvertFromEdfPlusDToEdfPlusC(Edf* edf);
Status ConvertToEdfPlusC(Edf* edf);

// Utilities to modify EDF files -----------------------------------------------
Status ResetEdfAnnotationSamplesPerDataRecord(Edf* edf);

}  // namespace eeg_modelling

#endif  // EEG_MODELLING_EDF_EDF_UTIL_H_
