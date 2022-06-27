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

#include "edf/temple_util.h"

#include <fstream>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "edf/base/canonical_errors.h"
#include "edf/base/status_macros.h"
#include "edf/base/time_proto_util.h"
#include "edf/proto/event.pb.h"
#include "edf/proto/segment.pb.h"

namespace eeg_modelling {

bool ParseTemplePatientInfo(const string& segment_filename,
                            const string& patient_info_str,
                            PatientInfo* patient_info) {
  string stripped_patient_info_str = patient_info_str;
  absl::StripTrailingAsciiWhitespace(&stripped_patient_info_str);

  // Patient_info_str is in the following format(s) depending on the data type:
  // <Patient_id> <Gender> <Date-of-Birth> <Patient_id> <Age>, e.g
  // 00000066 M 01-JAN-1948 00000078 Age:52"
  // <Patient_id>
  // 00000066
  const std::vector<absl::string_view> strs =
      absl::StrSplit(stripped_patient_info_str, ' ');
  if (strs.empty()) {
    ABSL_RAW_LOG(FATAL, "Unexpected patient string size in file %s",
                 segment_filename.c_str());
  }

  const absl::string_view patient_id = strs[0];
  patient_info->set_patient_id(string(patient_id));

  if (strs.size() > 1) {
    // Set Gender if it exists.
    const auto gender_str = strs[1];
    if (gender_str == "F") {
      patient_info->set_gender(PatientInfo::FEMALE);
    } else if (gender_str == "M") {
      patient_info->set_gender(PatientInfo::MALE);
    } else {
      patient_info->set_gender(PatientInfo::UNSPECIFIED);
    }
  }

  if (strs.size() > 4) {
    // Set Age if it exists.
    const std::vector<absl::string_view> age_str = absl::StrSplit(strs[4], ':');
    if (age_str.size() != 2) {
      ABSL_RAW_LOG(WARNING, "Age not present in patient string.");
      return false;
    }
    int age;
    if (age_str[0] != "Age" || !absl::SimpleAtoi(age_str[1], &age)) {
      ABSL_RAW_LOG(WARNING, "Invalid Age : %s and %s",
                   std::string(age_str[0]).c_str(),
                   std::string(age_str[1]).c_str());
      return false;
    }
    patient_info->set_age(age);
  }
  return true;
}

StatusOr<Annotation> GetRawTextAnnotationForTemple(
    const Segment& segment, const string& annotation_file_path) {
  std::ifstream annotation_file(annotation_file_path);
  auto segment_start_time = segment.start_time();

  int count = 0;
  Annotation annotation;
  annotation.set_segment_id(segment.segment_id());
  annotation.set_type(RAW_TEXT);
  auto events = annotation.mutable_events();

  std::string line;
  while (std::getline(annotation_file, line)) {
    ++count;
    // The first line contains versioning info and second line is empty.
    if (count <= 2) {
      continue;
    }
    // Parse the event.
    // A sample line is "10.0001 11.0002 seiz 1.0000", where
    //  10.0001 is the event start time in seconds,
    //  11.0002 is the event end time in seconds,
    //  seiz is the event type,
    //  1.0000 is the probablity that the event is correct.
    std::vector<string> strs = absl::StrSplit(line, ' ');
    if (strs.size() != 4) {
      ABSL_RAW_LOG(FATAL, "Invalid size for parsed event");
    }
    float start_time_sec, end_time_sec, probability = 0.0f;
    if (!absl::SimpleAtof(strs[0], &start_time_sec) ||
        !absl::SimpleAtof(strs[1], &end_time_sec) ||
        !absl::SimpleAtof(strs[3], &probability)) {
      return InvalidArgumentError(
          absl::StrCat("Failed to convert event times to float : ", line));
    }

    string label = strs[2];
    // Add all labels except for bckg. TODO(b/115564053): Configure which labels
    // to include.
    if (label != "bckg") {
      if (probability < 0.0f || probability > 1.0f) {
        return InvalidArgumentError(
            absl::StrCat("Probability out of range : ", probability));
      }
      auto event = events->add_event();
      event->set_start_time_sec(start_time_sec);
      event->set_end_time_sec(end_time_sec);
      event->set_label(label);
      event->set_probability(probability);

      absl::Time segment_start =
          DecodeGoogleApiProto(segment.start_time()).ValueOrDie();
      absl::Duration event_start_shift = absl::Seconds(start_time_sec);
      absl::Duration event_end_shift = absl::Seconds(end_time_sec);
      ASSIGN_OR_RETURN(*event->mutable_start_time(),
                       EncodeGoogleApiProto(segment_start + event_start_shift));
      ASSIGN_OR_RETURN(*event->mutable_end_time(),
                       EncodeGoogleApiProto(segment_start + event_end_shift));
    }
  }
  return annotation;
}
}  // namespace eeg_modelling
