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

#include "edf/edf_util.h"

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "edf/base/canonical_errors.h"
#include "edf/base/status_macros.h"
#include "edf/base/statusor.h"
#include "edf/proto/edf.pb.h"

namespace eeg_modelling {

namespace {
// See date insanity here: www.edfplus.info/specs/edfplus.html#additionalspecs
string FixEDFY2KDateString(const string& edf_date_str) {
  string fixed_date_str = edf_date_str;
  // EDF spec: For DD-MM-YY, if YY >= 85, then it is in 19YY, else 20YY.
  int year = 0;
  if (!absl::SimpleAtoi(fixed_date_str.substr(6, 2), &year)) {
    return "";
  }
  if (year >= 85) {
    return fixed_date_str.insert(6, "19");
  }
  return fixed_date_str.insert(6, "20");
}
}  // namespace

StatusOr<int> GetFirstEdfAnnotationSignalIndex(const Edf& edf) {
  for (int i = 0; i < edf.header().signal_headers_size(); ++i) {
    if (edf.header().signal_headers(i).label() == "EDF Annotations") {
      return i;
    }
  }
  return InvalidArgumentError(
      "Edf has no annotation signals specified in header.");
}

bool HasEdfAnnotationSignal(const Edf& edf) {
  return !GetFirstEdfAnnotationSignalIndex(edf).ok();
}

StatusOr<double> GetNumSecondsPerDataRecord(const EdfHeader& edf_header) {
  double num_seconds_per_data_record = 0.0;
  if (!absl::SimpleAtod(edf_header.num_seconds_per_data_record(),
                        &num_seconds_per_data_record)) {
    return InvalidArgumentError("Unable to parse num seconds per data record.");
  }
  if (num_seconds_per_data_record <= 0.0) {
    return InvalidArgumentError(
        absl::StrCat("Invalid num_seconds_per_data_record specified: ",
                     edf_header.num_seconds_per_data_record()));
  }
  return num_seconds_per_data_record;
}

StatusOr<double> GetDataRecordStartSecOffset(const DataRecord& data_record,
                                             int annotation_signal_index) {
  if (data_record.signals_size() <= annotation_signal_index) {
    return InvalidArgumentError(
        "Data record did not have signal at given index.");
  }
  auto annotation_signal = data_record.signals(annotation_signal_index);
  if (!annotation_signal.has_annotations()) {
    return InvalidArgumentError(
        "Data record signal at given index is not for annotations.");
  }
  if (annotation_signal.annotations().tals_size() == 0) {
    return InvalidArgumentError("Data record missing start timestamp.");
  }
  auto tal = annotation_signal.annotations().tals(0);
  if (tal.annotations_size() == 0) {
    return InvalidArgumentError("Data record TAL missing start annotation.");
  }
  if (!tal.annotations(0).empty()) {
    return InvalidArgumentError(
        "Data record TAL has non-blank start annotation.");
  }
  double seconds = 0.0;
  if (!absl::SimpleAtod(tal.start_offset_seconds(), &seconds)) {
    return InvalidArgumentError("Unable to start time of data record.");
  }
  return seconds;
}

StatusOr<absl::Time> GetStartTime(absl::Time recording_start_time,
                                  const string& offset_start_sec_str) {
  // Parse start into a time.
  if (offset_start_sec_str.empty()) {
    return InvalidArgumentError("Empty time start");
  }
  float start = 0.0f;
  if (!absl::SimpleAtof(offset_start_sec_str, &start)) {
    return InvalidArgumentError(absl::StrCat(
        "Could not parse time start to double: ", offset_start_sec_str));
  }
  return recording_start_time + absl::Seconds(start);
}

StatusOr<std::tuple<absl::Time, absl::Time>> GetStartEndTimes(
    absl::Time recording_start_time, const string& offset_start_sec_str,
    const string& duration_sec_str) {
  absl::Time start_time;
  ASSIGN_OR_RETURN(start_time,
                   GetStartTime(recording_start_time, offset_start_sec_str));
  float duration = 0.0f;
  // If it is not empty, and we can't parse it, error out.
  if (!duration_sec_str.empty() &&
      (!absl::SimpleAtof(duration_sec_str, &duration))) {
    return InvalidArgumentError(
        absl::StrCat("Could not parse duration: ", duration_sec_str));
  }
  if (duration < 0.0f) {
    // EDF+ spec disallows negative durations.
    return InvalidArgumentError("Unexpected negative duration.");
  }

  // TODO(jjtswan): Decide if we should output event_end_time if zero?
  return std::make_tuple<absl::Time, absl::Time>(
      absl::Time(start_time), start_time + absl::Seconds(duration));
}

string SimpleDtoa(double value) {
  std::ostringstream ss;
  ss << value;
  std::string s(ss.str());
  return s;
}

Status ConvertFromEdfToEdfPlusC(Edf* edf) {
  if (HasEdfAnnotationSignal(*edf)) {
    return OkStatus();
  }
  EdfHeader::SignalHeader* annotation_signal_header =
      edf->mutable_header()->add_signal_headers();
  annotation_signal_header->set_label("EDF Annotations");
  // EDF+ Spec requires these specific values.
  annotation_signal_header->set_digital_min("-32768");
  annotation_signal_header->set_digital_max("32767");
  // Spec requires different values.
  annotation_signal_header->set_physical_min("-32768");
  annotation_signal_header->set_physical_max("32767");

  edf->mutable_header()->set_num_signals(edf->header().signal_headers_size());
  edf->mutable_header()->set_num_header_bytes(
      edf->header().num_header_bytes() + 256 /* bytes per signal header */);

  if (edf->header().type_from_reserved() == EdfHeader::UNSPECIFIED) {
    // Convert from EDF to EDF+C which supports annotations.
    edf->mutable_header()->set_type_from_reserved(EdfHeader::EDF_PLUS_C);
  }

  double num_seconds_per_data_record;
  ASSIGN_OR_RETURN(num_seconds_per_data_record,
                   GetNumSecondsPerDataRecord(edf->header()));

  // EDF+ spec requires that the first TAL in an data record should have one
  // empty annotation, and should be time-stamped to the beginning of the data
  // record, relative to start of the file.
  for (int i = 0; i < edf->data_records_size(); ++i) {
    TimeStampedAnnotationList* tal = edf->mutable_data_records(i)
                                         ->add_signals()
                                         ->mutable_annotations()
                                         ->add_tals();
    tal->set_start_offset_seconds(
        absl::StrCat("+", SimpleDtoa(num_seconds_per_data_record * i)));
    *tal->add_annotations() = "";
  }

  return OkStatus();
}

StatusOr<bool> CanConvertToEdfPlusC(const Edf& edf) {
  double num_seconds_per_data_record;
  ASSIGN_OR_RETURN(num_seconds_per_data_record,
                   GetNumSecondsPerDataRecord(edf.header()));

  // EDF+ spec requires that the first TAL in an data record should have one
  // empty annotation, and should be time-stamped to the beginning of the data
  // record, relative to start of the file.
  int annotation_signal_index;
  ASSIGN_OR_RETURN(annotation_signal_index,
                   GetFirstEdfAnnotationSignalIndex(edf));
  for (int i = 0; i < edf.data_records_size(); ++i) {
    double record_start;
    ASSIGN_OR_RETURN(record_start,
                     GetDataRecordStartSecOffset(edf.data_records(i),
                                                 annotation_signal_index));
    if (record_start != num_seconds_per_data_record * i) {
      // Discontinouous, so we cannot convert.
      return false;
    }
  }
  return true;
}

Status ConvertFromEdfPlusDToEdfPlusC(Edf* edf) {
  bool can_convert;
  ASSIGN_OR_RETURN(can_convert, CanConvertToEdfPlusC(*edf));
  if (!can_convert) {
    return InvalidArgumentError(
        "Cannot convert EDF+D to EDF+C without splitting to multiple files.");
  }
  // If all the records are continuous, then this is the only difference between
  // EDF+D and EDF+C.
  edf->mutable_header()->set_type_from_reserved(EdfHeader::EDF_PLUS_C);
  return OkStatus();
}

Status ConvertToEdfPlusC(Edf* edf) {
  if (edf->header().type_from_reserved() == EdfHeader::EDF_PLUS_C) {
    return OkStatus();
  }
  if (edf->header().type_from_reserved() == EdfHeader::UNSPECIFIED) {
    // We assume this is just normal EDF.
    return ConvertFromEdfToEdfPlusC(edf);
  }
  if (edf->header().type_from_reserved() == EdfHeader::EDF_PLUS_D) {
    return ConvertFromEdfPlusDToEdfPlusC(edf);
  }
  return InvalidArgumentError("Invalid edf type for conversion.");
}

Status ResetEdfAnnotationSamplesPerDataRecord(Edf* edf) {
  int annotation_signal_index;
  ASSIGN_OR_RETURN(annotation_signal_index,
                   GetFirstEdfAnnotationSignalIndex(*edf));

  // TODO(jjtswan): Right now we hard code.  Consider actually calculating this
  // based on max bytes of annotations stored in an individual data record.
  edf->mutable_header()
      ->mutable_signal_headers(annotation_signal_index)
      ->set_num_samples_per_data_record(500);
  return OkStatus();
}

// Parsing of date depends on three fields, and whether or not we are dealing
// with EDF or EDF+: http://www.edfplus.info/specs/edfplus.html#additionalspecs
StatusOr<absl::Time> ParseEdfStartTime(const EdfHeader& header) {
  const string start_date_str = header.recording_start_date();
  const string start_time_str = header.recording_start_time();
  const string recording_id_str =
      header.local_recording_information().full_text();

  absl::Time start_time;
  string error;
  bool parsed = false;
  string timestamp;
  if (absl::StartsWith(recording_id_str, "Startdate ") &&
      !absl::StartsWith(recording_id_str, "Startdate X")) {
    timestamp = absl::StrCat(
        recording_id_str.substr(0, strlen("Startdate 03-JAN-2005")), " ",
        start_time_str);
    parsed = absl::ParseTime("Startdate %d-%b-%Y %H.%M.%S", timestamp,
                             &start_time, &error);
  } else {
    // Input is DD-MM-YY.  Convert to DD-MM-YYYY based on EDF specifications.
    const string fixed_date_str = FixEDFY2KDateString(start_date_str);
    timestamp = absl::StrCat(fixed_date_str, " ", start_time_str);
    parsed =
        absl::ParseTime("%d.%m.%Y %H.%M.%S", timestamp, &start_time, &error);
  }
  if (!parsed) {
    return InvalidArgumentError(
        absl::StrCat("Error parsing segment start from: ", timestamp));
  }
  return start_time;
}

StatusOr<absl::Time> GetEdfDataRecordStartTime(const Edf& edf,
                                               int data_record) {
  if (data_record < 0 || data_record >= edf.data_records_size()) {
    return InvalidArgumentError(
        absl::StrCat("Invalid data record index of ", data_record));
  }
  absl::Time recording_start_time;
  ASSIGN_OR_RETURN(recording_start_time, ParseEdfStartTime(edf.header()));
  int annotation_index;
  ASSIGN_OR_RETURN(annotation_index, GetFirstEdfAnnotationSignalIndex(edf));
  const AnnotationSignal& annotation_signal =
      edf.data_records(data_record).signals(annotation_index).annotations();
  if (annotation_signal.tals_size() == 0) {
    return InvalidArgumentError(
        "Edf Annotation does not have a TAL indicating start of data record");
  }
  if (!annotation_signal.tals(0).duration_seconds().empty()) {
    return InvalidArgumentError(absl::StrCat(
        "First TAL should only contain start time to indicate start of data "
        "record. Instead, got: ",
        annotation_signal.tals(0).duration_seconds()));
  }
  if (annotation_signal.tals(0).annotations_size() == 0) {
    return InvalidArgumentError(
        "First TAL should have at least one annotation, which should be an "
        "empty string.  Instead, no annotation was included.");
  }
  if (!annotation_signal.tals(0).annotations(0).empty()) {
    return InvalidArgumentError(absl::StrCat(
        "First TAL annotation should be an empty string, denoting start of "
        "data record. Instead, got: ",
        annotation_signal.tals(0).annotations(0)));
  }

  std::tuple<absl::Time, absl::Time> start_end;
  ASSIGN_OR_RETURN(
      start_end,
      GetStartEndTimes(recording_start_time,
                       annotation_signal.tals(0).start_offset_seconds(),
                       annotation_signal.tals(0).duration_seconds()));

  return std::get<0>(start_end);
}

}  // namespace eeg_modelling
