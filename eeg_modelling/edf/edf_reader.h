#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_READER_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_READER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include "edf/base/statusor.h"
#include "edf/edf_file.h"
#include "edf/proto/edf.pb.h"
#include "edf/proto/event.pb.h"

using std::string;

namespace eeg_modelling {

// EDF reader that prefetches the header and provides an interface to directly
// access any chunk of data. Currently, the reader does not segment EDF+D.
class EdfReader {
 public:
  static StatusOr<std::unique_ptr<EdfReader>> Create(const string& edf_path,
                                                     EdfFile* edf_file);

  // Use Create instead. This constructor is made public only for MakeUnique.
  EdfReader(const string& edf_path, EdfFile* edf_file,
            std::unique_ptr<EdfHeader> edf_header,
            double num_seconds_per_data_record,
            const google::protobuf::Timestamp& start_timestamp);

  // Reads annotations from the data records that covers the given interval
  // defined by [offset start seconds, offset end seconds).
  StatusOr<AnnotationSignal> ReadAnnotations(double start_offset_secs,
                                             double end_offset_secs);

  // Reads all non-annotation channel data from the samples that cover the given
  // interval defined by [offset start seconds, offset end seconds). The
  // returned value is a map from channel labels to a vector of their
  // corresponding physical values. The labels are as specified in the header,
  // e.g. "EEG Fp2-Ref", "POL X1".
  StatusOr<std::unordered_map<string, std::vector<double>>> ReadSignals(
      double start_offset_secs, double end_offset_secs);

  const string& get_edf_path() const { return edf_path_; }
  const EdfHeader& get_edf_header() const { return *edf_header_; }
  google::protobuf::Timestamp get_start_timestamp() const {
    return start_timestamp_;
  }
  double get_num_seconds_per_data_record() const {
    return num_seconds_per_data_record_;
  }
  void SeekToDataRecord(int index, const EdfHeader& edf_header);

 private:
  const string edf_path_;
  EdfFile* edf_file_;
  std::unique_ptr<EdfHeader> edf_header_;
  double num_seconds_per_data_record_;
  google::protobuf::Timestamp start_timestamp_;
  int annotation_index_;
};

}  // namespace eeg_modelling

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_READER_H_
