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

#include "edf/edf_reader.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <tuple>

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "edf/base/canonical_errors.h"
#include "edf/base/status.h"
#include "edf/base/status_macros.h"
#include "edf/base/time_proto_util.h"
#include "edf/edf.h"
#include "edf/edf_util.h"
#include "edf/parse_edf_lib.h"

namespace eeg_modelling {

namespace {

constexpr char kEdfAnnotationsLabel[] = "EDF Annotations";
constexpr int kBytesPerSample = 2;

// Contains index to a specific sample in EDF.
struct SampleIndex {
  int data_record_index;
  int sample_index;

  bool operator<(const SampleIndex& other) const {
    if (data_record_index < other.data_record_index) {
      return true;
    } else if (data_record_index == other.data_record_index) {
      return sample_index < other.sample_index;
    } else {
      return false;
    }
  }

  bool operator==(const SampleIndex& other) const {
    return data_record_index == other.data_record_index &&
           sample_index == other.sample_index;
  }

  bool operator>(const SampleIndex& other) const { return other < *this; }
};

// A range of samples specified by [start sample index, end sample index).
struct SampleRange {
  SampleIndex start;
  SampleIndex end;
};

// Returns the start index for the given start offset seconds.
SampleIndex GetStartSampleIndex(double num_seconds_per_data_record,
                                double num_samples_per_data_record,
                                double start_offset_secs) {
  SampleIndex sample_index;
  sample_index.data_record_index =
      std::floor(start_offset_secs / num_seconds_per_data_record);
  sample_index.sample_index =
      std::floor(std::fmod(start_offset_secs, num_seconds_per_data_record) *
                 num_samples_per_data_record / num_seconds_per_data_record);
  return sample_index;
}

// Returns the end index for the given end offset seconds.
SampleIndex GetEndSampleIndex(double num_seconds_per_data_record,
                              double num_samples_per_data_record,
                              double end_offset_secs) {
  // This is similar to start index except that if the end_offset_secs does not
  // correspond exactly to the beginning of a sample, we have to advance it to
  // the next sample.
  SampleIndex sample_index;
  sample_index.data_record_index =
      std::floor(end_offset_secs / num_seconds_per_data_record);
  double index_in_data_record =
      std::fmod(end_offset_secs, num_seconds_per_data_record) *
      num_samples_per_data_record / num_seconds_per_data_record;
  sample_index.sample_index = std::floor(index_in_data_record);
  if (std::fmod(index_in_data_record, 1.0) != 0.0) {
    // It is okay if sample_index == num_samples_per_data_record.
    ++sample_index.sample_index;
  }
  return sample_index;
}

// Returns sample range corresponding to the given interval.
SampleRange GetSampleRange(double num_seconds_per_data_record,
                           double num_samples_per_data_record,
                           double start_offset_secs, double end_offset_secs) {
  SampleRange sample_range;
  sample_range.start =
      GetStartSampleIndex(num_seconds_per_data_record,
                          num_samples_per_data_record, start_offset_secs);
  sample_range.end =
      GetEndSampleIndex(num_seconds_per_data_record,
                        num_samples_per_data_record, end_offset_secs);
  return sample_range;
}

// Returns sample ranges for all channels.
template <typename SignalHeaderContainer>
std::vector<SampleRange> GetAllSampleRanges(
    double num_seconds_per_data_record,
    const SignalHeaderContainer& signal_headers, double start_offset_secs,
    double end_offset_secs) {
  std::vector<SampleRange> sample_ranges;
  for (const auto& signal_header : signal_headers) {
    sample_ranges.push_back(
        GetSampleRange(num_seconds_per_data_record,
                       signal_header.num_samples_per_data_record(),
                       start_offset_secs, end_offset_secs));
  }
  return sample_ranges;
}

// Helper to decode samples for signal.
class SignalHandler {
 public:
  SignalHandler() = default;

  explicit SignalHandler(const EdfHeader::SignalHeader& signal_header) {
    auto physical_min = std::stod(signal_header.physical_min().c_str());
    auto physical_max = std::stod(signal_header.physical_max().c_str());
    auto digital_min = std::stod(signal_header.digital_min().c_str());
    auto digital_max = std::stod(signal_header.digital_max().c_str());

    scale_ = (physical_max - physical_min) / (digital_max - digital_min);
    offset_ = physical_max / scale_ - digital_max;
  }

  // Decodes samples from the given index range, and insert the values to the
  // result.
  void DecodeRange(const IntegerSignal& integer_signal, const int start,
                   const int end, std::vector<double>* result) {
    for (int i = start; i < end; ++i) {
      result->push_back(Decode(integer_signal.samples(i)));
    }
  }

 private:
  double Decode(int64_t sample) const { return (sample + offset_) * scale_; }

  double offset_;
  double scale_;
};

// Move all timestamped annotations from the input to the output.
void MoveAnnotations(AnnotationSignal* input, AnnotationSignal* output) {
  for (TimeStampedAnnotationList& tal : *input->mutable_tals()) {
    output->add_tals()->Swap(&tal);
  }
}
}  // namespace

void EdfReader::SeekToDataRecord(int index, const EdfHeader& edf_header) {
  int num_bytes_per_data_record = 0;
  for (const auto& signal_header : edf_header.signal_headers()) {
    num_bytes_per_data_record +=
        signal_header.num_samples_per_data_record() * kBytesPerSample;
  }

  if (edf_file_->SeekFromBegin(edf_header.num_header_bytes() +
                               num_bytes_per_data_record * index) !=
      OkStatus()) {
    ABSL_RAW_LOG(FATAL, "File seek failed");
  }
}

StatusOr<std::unique_ptr<EdfReader>> EdfReader::Create(const string& edf_path,
                                                       EdfFile* edf_file) {
  auto edf_header = absl::make_unique<EdfHeader>();
  RETURN_IF_ERROR(ParseEdfHeader(edf_file, edf_header.get()));

  std::vector<EdfHeader::SignalHeader> signal_headers;
  ASSIGN_OR_RETURN(signal_headers,
                   ParseEdfSignalHeaders(edf_file, edf_header->num_signals()));
  for (const auto& it : signal_headers) {
    *(edf_header->add_signal_headers()) = it;
  }

  double num_seconds_per_data_record;
  ASSIGN_OR_RETURN(num_seconds_per_data_record,
                   GetNumSecondsPerDataRecord(*edf_header));
  absl::Time absolute_start_time;
  ASSIGN_OR_RETURN(absolute_start_time, ParseEdfStartTime(*edf_header));
  google::protobuf::Timestamp start_timestamp;
  ASSIGN_OR_RETURN(start_timestamp, EncodeGoogleApiProto(absolute_start_time));

  return absl::make_unique<EdfReader>(edf_path, edf_file, std::move(edf_header),
                                      num_seconds_per_data_record,
                                      start_timestamp);
}

EdfReader::EdfReader(const string& edf_path, EdfFile* edf_file,
                     std::unique_ptr<EdfHeader> edf_header,
                     double num_seconds_per_data_record,
                     const google::protobuf::Timestamp& start_timestamp)
    : edf_path_(edf_path),
      edf_file_(edf_file),
      edf_header_(std::move(edf_header)),
      num_seconds_per_data_record_(num_seconds_per_data_record),
      start_timestamp_(start_timestamp),
      annotation_index_(-1) {
  for (int annotation_index = 0;
       annotation_index < edf_header_->signal_headers().size();
       ++annotation_index) {
    if (edf_header_->signal_headers(annotation_index).label() ==
        kEdfAnnotationsLabel) {
      annotation_index_ = annotation_index;
      break;
    }
  }
}

StatusOr<AnnotationSignal> EdfReader::ReadAnnotations(double start_offset_secs,
                                                      double end_offset_secs) {
  AnnotationSignal annotations;
  if (annotation_index_ < 0) {
    return annotations;
  }

  // Find the sample ranges.
  std::vector<SampleRange> sample_ranges = GetAllSampleRanges(
      num_seconds_per_data_record_, edf_header_->signal_headers(),
      start_offset_secs, end_offset_secs);
  const SampleRange& annotation_range = sample_ranges[annotation_index_];

  // Seek to the first data record.
  SeekToDataRecord(annotation_range.start.data_record_index, *edf_header_);

  for (int data_record_index = annotation_range.start.data_record_index;
       data_record_index <=
       std::min(annotation_range.end.data_record_index,
                static_cast<int>(edf_header_->num_data_records() - 1));
       ++data_record_index) {
    for (int signal_index = 0;
         signal_index < edf_header_->signal_headers().size(); ++signal_index) {
      const auto& signal_header = edf_header_->signal_headers(signal_index);
      const SampleRange& sample_range = sample_ranges[signal_index];
      int num_samples = signal_header.num_samples_per_data_record();
      int channel_bytes = num_samples * kBytesPerSample;
      int start_sample_index =
          data_record_index == sample_range.start.data_record_index
              ? sample_range.start.sample_index
              : 0;
      int end_sample_index =
          data_record_index == sample_range.end.data_record_index
              ? sample_range.end.sample_index
              : num_samples;

      if (signal_index == annotation_index_ &&
          start_sample_index < end_sample_index) {
        AnnotationSignal annotation_signal;
        ASSIGN_OR_RETURN(
            annotation_signal,
            ParseEdfAnnotationSignal(edf_file_, channel_bytes, false));
        MoveAnnotations(&annotation_signal, &annotations);
      } else {
        if (edf_file_->SeekFromBegin(edf_file_->Tell() + channel_bytes) !=
            OkStatus()) {
          ABSL_RAW_LOG(FATAL, "File seek failed");
        }
      }
    }
  }
  return annotations;
}

StatusOr<absl::node_hash_map<string, std::vector<double>>>
EdfReader::ReadSignals(double start_offset_secs, double end_offset_secs) {
  // Prepare output.
  absl::node_hash_map<string, std::vector<double>> signals;
  for (const auto& signal_header : edf_header_->signal_headers()) {
    if (signal_header.label() != kEdfAnnotationsLabel) {
      signals[signal_header.label()];
    }
  }
  if (signals.empty()) {
    return signals;
  }

  // Find the sample ranges.
  std::vector<SampleRange> sample_ranges = GetAllSampleRanges(
      num_seconds_per_data_record_, edf_header_->signal_headers(),
      start_offset_secs, end_offset_secs);

  // Verify that all occupy the same data record range.
  int start_data_record = sample_ranges[0].start.data_record_index;
  int end_data_record = sample_ranges[0].end.data_record_index;
  for (size_t i = 1; i < sample_ranges.size(); ++i) {
    if (start_data_record != sample_ranges[i].start.data_record_index) {
      return InternalError(absl::StrCat(
          "Inconsistent start data record indices: ", start_data_record,
          " != ", sample_ranges[i].start.data_record_index));
    }
    if (end_data_record != sample_ranges[i].end.data_record_index) {
      return InternalError(absl::StrCat(
          "Inconsistent end data record indices: ", end_data_record,
          " != ", sample_ranges[i].end.data_record_index));
    }
  }

  // Prepare signal handlers.
  std::vector<SignalHandler> signal_handlers;
  for (const auto& signal_header : edf_header_->signal_headers()) {
    signal_handlers.emplace_back(signal_header);
  }

  // Seek to the first data record.
  SeekToDataRecord(start_data_record, *edf_header_);

  for (int data_record_index = start_data_record;
       data_record_index <=
       std::min(end_data_record,
                static_cast<int>(edf_header_->num_data_records() - 1));
       ++data_record_index) {
    for (int signal_index = 0;
         signal_index < edf_header_->signal_headers().size(); ++signal_index) {
      const SampleRange& sample_range = sample_ranges[signal_index];
      const auto& signal_header = edf_header_->signal_headers(signal_index);
      int num_samples = signal_header.num_samples_per_data_record();
      int channel_bytes = num_samples * kBytesPerSample;
      int start_sample_index = data_record_index == start_data_record
                                   ? sample_range.start.sample_index
                                   : 0;
      int end_sample_index = data_record_index == end_data_record
                                 ? sample_range.end.sample_index
                                 : num_samples;

      if (signal_header.label() != kEdfAnnotationsLabel &&
          start_sample_index < end_sample_index) {
        IntegerSignal integer_signal;
        ASSIGN_OR_RETURN(integer_signal,
                         ParseEdfIntegerSignal(edf_file_, num_samples,
                                               signal_header.label(), true));
        signal_handlers[signal_index].DecodeRange(
            integer_signal, start_sample_index, end_sample_index,
            &signals[signal_header.label()]);
      } else {
        if (edf_file_->SeekFromBegin(edf_file_->Tell() + channel_bytes) !=
            OkStatus()) {
          ABSL_RAW_LOG(FATAL, "File seek failed");
        }
      }
    }
  }
  return signals;
}

}  // namespace eeg_modelling
