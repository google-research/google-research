#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_UTIL_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_UTIL_H_

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

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_UTIL_H_
