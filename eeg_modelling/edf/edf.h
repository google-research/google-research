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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_H_

#include <string>
#include <vector>

#include "edf/base/status.h"
#include "edf/base/statusor.h"
#include "edf/edf_file.h"
#include "edf/proto/edf.pb.h"

using std::string;

namespace eeg_modelling {

// Parses .edf file and stores output into proto representation of edf format.
StatusOr<Edf> ParseEdfToEdfProto(const string& filename);

// Converts edf proto into edf file format and writes to file.
Status WriteEdf(const eeg_modelling::Edf& edf, const string& filename);

// Utils.
// Consider moving these to an internal namespace after refactoring the
// dependencies.
Status ParseEdfHeader(EdfFile* fp, EdfHeader* header);
StatusOr<std::vector<EdfHeader::SignalHeader>> ParseEdfSignalHeaders(
    EdfFile* fp, int num_signals);
Status ParseEdfHeader(EdfFile* fp, EdfHeader* header);
StatusOr<IntegerSignal> ParseEdfIntegerSignal(
    EdfFile* fp, int num_samples, const string& signal_label,
    bool return_error_on_num_samples_mismatch);

StatusOr<AnnotationSignal> ParseEdfAnnotationSignal(EdfFile* fp,
                                                    int channel_bytes,
                                                    bool only_parse_first);

}  // namespace eeg_modelling

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_H_
