#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_PARSE_EDF_LIB_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_PARSE_EDF_LIB_H_

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

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_PARSE_EDF_LIB_H_
