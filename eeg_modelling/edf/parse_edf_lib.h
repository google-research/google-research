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

#ifndef EEG_MODELLING_EDF_PARSE_EDF_LIB_H_
#define EEG_MODELLING_EDF_PARSE_EDF_LIB_H_

#include <string>
#include <vector>

#include "absl/time/time.h"
#include "edf/base/statusor.h"
#include "edf/edf_file.h"
#include "edf/proto/annotation.pb.h"
#include "edf/proto/segment.pb.h"

using std::string;

namespace eeg_modelling {

// Parses the .edf file and stores the output in a segment proto.
StatusOr<Segment> ParseEdfToSegmentProto(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix);

// Parses the .edf file and stores the output in a list of segment protos.
StatusOr<std::vector<Segment>> ParseEdfToSegmentProtos(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix);

// Parse all EDF annotations from the given file, in the specified time range of
// [start_time, end_time].
StatusOr<Annotation> ParseEdfToAnnotationProto(
    const string& filename, const absl::Time start_time,
    const absl::Time end_time,
    const string& split_segment_by_annotation_with_prefix);

// Parse all EDF annotations from the given file, keyed by segment identifier.
StatusOr<std::vector<Annotation>> ParseEdfToAnnotationProtoPerSegment(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix);

}  // namespace eeg_modelling

#endif  // EEG_MODELLING_EDF_PARSE_EDF_LIB_H_
