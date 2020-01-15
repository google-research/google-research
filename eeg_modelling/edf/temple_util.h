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
