#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_TF_EXAMPLE_LIB_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_TF_EXAMPLE_LIB_H_

#include <string>

#include "edf/base/status.h"
#include "edf/base/statusor.h"
#include "edf/proto/annotation.pb.h"
#include "edf/proto/segment.pb.h"
#include "tensorflow/core/example/example.pb.h"

namespace eeg_modelling {

StatusOr<tensorflow::Example> GenerateExampleForSegment(
    const Segment& segment, const Annotations& annotations);
}

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_TF_EXAMPLE_LIB_H_
