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

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "edf/base/statusor.h"
#include "edf/proto/annotation.pb.h"
#include "edf/proto/segment.pb.h"

using std::string;

namespace eeg_modelling {

bool ParseTemplePatientInfo(const string& segment_filename,
                            const string& patient_info_str,
                            PatientInfo* patient_info);

StatusOr<Annotation> GetRawTextAnnotationForTemple(
    const Segment& segment, const string& annotation_file_path);

}  // namespace eeg_modelling
