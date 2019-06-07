#include "edf/edf.h"

#include <string>
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "edf/base/canonical_errors.h"
#include "edf/base/status.h"
#include "edf/base/status_macros.h"
#include "edf/base/statusor.h"
#include "edf/edf_oss_file.h"

namespace eeg_modelling {

constexpr char kEdfPlusC[] = "EDF+C";
constexpr char kEdfPlusD[] = "EDF+D";
constexpr char kEdfAnnotationsLabel[] = "EDF Annotations";

constexpr char kChar0 = static_cast<char>(0);
constexpr char kChar20 = static_cast<char>(20);
constexpr char kChar21 = static_cast<char>(21);

const char kDelims[] = {kChar0, kChar20};
const int kDelimsSize = sizeof(kDelims) / sizeof(kDelims[0]);

void TrimRunsInString(string* s, absl::string_view remove) {
  string::iterator dest = s->begin();
  string::iterator src_end = s->end();
  for (string::iterator src = s->begin(); src != src_end;) {
    if (remove.find(*src) == absl::string_view::npos) {
      *(dest++) = *(src++);
    } else {
      // Skip to the end of this run of chars that are in 'remove'.
      for (++src; src != src_end; ++src) {
        if (remove.find(*src) == absl::string_view::npos) {
          if (dest != s->begin()) {
            // This is an internal run; collapse it.
            *(dest++) = remove[0];
          }
          *(dest++) = *(src++);
          break;
        }
      }
    }
  }
  s->erase(dest, src_end);
}

void ReadToString(EdfFile* fp, string* dest, size_t num_bytes) {
  if (fp->ReadToString(dest, num_bytes) != num_bytes) {
    ABSL_RAW_LOG(FATAL, "File reading failed");
  }
}

void Format(string* in) { TrimRunsInString(in, " "); }

int ReadToFirst(EdfFile* fp, const char* delims, int num_delims, char* out,
                int out_size, bool* found_delim_ptr) {
  *found_delim_ptr = false;
  auto buffer_start = fp->Tell();
  auto bytes_read = fp->Read(out, out_size);

  if (bytes_read <= 0) {
    return bytes_read;
  }

  char* delim_pos =
      std::find_first_of(out, out + bytes_read, delims, delims + num_delims);
  if (delim_pos == out + bytes_read) {
    // Did not find the delimiter.
    return bytes_read;
  }

  *found_delim_ptr = true;
  int bytes_including_delim = delim_pos - out + 1;
  // Move buffer to character after the delimiter.
  if (fp->SeekFromBegin(buffer_start + bytes_including_delim) != OkStatus()) {
    ABSL_RAW_LOG(FATAL, "Error: Seek failed.");
  }
  return bytes_including_delim;
}

StatusOr<string> GetNextToken(EdfFile* fp, int* bytes_left_ptr,
                              bool* found_end_ptr) {
  *found_end_ptr = false;
  const int prev_bytes_left = *bytes_left_ptr;
  constexpr int kTokenBufferSize = 32;
  // NOTE(jjtswan): We could also get and re-use cords.
  char token_buffer[kTokenBufferSize];
  bool found_delim = false;
  int bytes_to_read = std::min(*bytes_left_ptr, kTokenBufferSize);

  int bytes_read = ReadToFirst(fp, kDelims, kDelimsSize, token_buffer,
                               bytes_to_read, &found_delim);
  if (bytes_read < 0) {
    return InternalError("Unexpected input error parsing annotations.");
  }
  if (bytes_read > bytes_to_read) {
    ABSL_RAW_LOG(FATAL,
                 "Error: bytes_read should never be greater than specified "
                 "bytes_to_read.");
  }

  *bytes_left_ptr -= bytes_read;
  if (!found_delim) {
    // Something went wrong because we did not find the delimiter.
    if (bytes_read < bytes_to_read) {
      return InvalidArgumentError("Unexpected EOF parsing annotations.");
    }
    // From here on, we know bytes_read == bytes_to_read.
    if (bytes_read == prev_bytes_left) {
      return InvalidArgumentError(
          "Did not find delimiter within specified bytes left to read.");
    }
    if (bytes_read == kTokenBufferSize) {
      return ResourceExhaustedError(
          "Did not find delimiter within max token_buffer buffer size. "
          "Consider increasing.");
    }
    ABSL_RAW_LOG(FATAL,
                 "Error: If bytes_read == bytes_to_read, then they should "
                 "equal either token buffer size, or specified byte limit.");
  }
  // Found a delimiter.  Two valid cases:
  // Case 1: we found \0 at first byte, indicating end of a TAL or no more TALs.
  if (token_buffer[0] == kChar0) {
    *found_end_ptr = true;
    // Return empty string.
    return string("");
  }

  string out_string = string(absl::string_view(token_buffer, bytes_read - 1));

  // Case 2: Find \x14 at last byte read, which is a valid TAL token.
  if (token_buffer[bytes_read - 1] == kChar20) {
    // Clip out the terminator and return token.
    return string(absl::string_view(token_buffer, bytes_read - 1));
  }

  // No more valid cases left, something unexpected happened.
  return InvalidArgumentError("Unexpected error parsing annotations.");
}

Status ParseEdfHeader(EdfFile* fp, EdfHeader* header) {
  // Read version number.
  ReadToString(fp, header->mutable_version(), 8);
  Format(header->mutable_version());

  // Read patient id.
  ReadToString(
      fp, header->mutable_local_patient_identification()->mutable_full_text(),
      80 /* num_bytes */);
  Format(header->mutable_local_patient_identification()->mutable_full_text());

  // Read recording id.
  ReadToString(
      fp, header->mutable_local_recording_information()->mutable_full_text(),
      80 /* num_bytes */);
  Format(header->mutable_local_recording_information()->mutable_full_text());

  // Read start date and time.
  ReadToString(fp, header->mutable_recording_start_date(), 8 /* num_bytes */);
  Format(header->mutable_recording_start_date());

  ReadToString(fp, header->mutable_recording_start_time(), 8 /* num_bytes */);
  Format(header->mutable_recording_start_time());

  // Read num bytes in header record.
  string data;
  ReadToString(fp, &data, 8 /* num_bytes */);
  uint32_t num_header_bytes = 0;
  if (!absl::SimpleAtoi(data, &num_header_bytes)) {
    return InvalidArgumentError(
        absl::StrCat("Failed to parse num_header_bytes field: ", data));
  }
  header->set_num_header_bytes(num_header_bytes);

  // Read total of 44 reserved bytes.
  // First 5 bytes might either be "EDF+C" or "EDF+D".
  string reserved_version;
  ReadToString(fp, &reserved_version, 5 /* num_bytes */);

  if (reserved_version == kEdfPlusD) {
    header->set_type_from_reserved(EdfHeader::EDF_PLUS_D);
  } else if (reserved_version == kEdfPlusC) {
    header->set_type_from_reserved(EdfHeader::EDF_PLUS_C);
  } else {
    // Typically means EDF.
    header->set_type_from_reserved(EdfHeader::UNSPECIFIED);
  }

  // Read and discard the rest of the 44 - 5 = 39 reserved bytes.
  if (fp->SeekFromBegin(fp->Tell() + 39) != OkStatus()) {
    ABSL_RAW_LOG(FATAL, "Error: Seek failed.");
  }

  // Read num data records and record duration.
  ReadToString(fp, &data, 8 /* num_bytes */);
  int32_t num_data_records = 0;
  if (!absl::SimpleAtoi(data, &num_data_records)) {
    return InvalidArgumentError(
        absl::StrCat("Failed to parse num_data_records field: ", data));
  }
  if (num_data_records < 0 && num_data_records != -1) {
    return InvalidArgumentError(absl::StrCat(
        "Num data records is an invalid negative value (only -1 allowed): ",
        num_data_records));
  }
  header->set_num_data_records(num_data_records);

  ReadToString(fp, header->mutable_num_seconds_per_data_record(),
               8 /* num_bytes */);
  Format(header->mutable_num_seconds_per_data_record());

  // Read number of signals in data record.
  ReadToString(fp, &data, 4 /* num_bytes */);
  uint32_t num_signals = 0;
  if (!absl::SimpleAtoi(data, &num_signals)) {
    return InvalidArgumentError(
        absl::StrCat("Failed to parse num_signals field: ", data));
  }
  header->set_num_signals(num_signals);

  return OkStatus();
}

StatusOr<std::vector<EdfHeader::SignalHeader>> ParseEdfSignalHeaders(
    EdfFile* fp, int num_signals) {
  // Read channel parameters
  string channel_name;
  string transducer_type;
  string physical_dimension;
  string physical_min;
  string physical_max;
  string digital_min;
  string digital_max;
  string prefiltering;
  string num_samples_in_data_record_str;
  string reserved;

  ReadToString(fp, &channel_name, 16 * num_signals);
  ReadToString(fp, &transducer_type, 80 * num_signals);
  ReadToString(fp, &physical_dimension, 8 * num_signals);
  ReadToString(fp, &physical_min, 8 * num_signals);
  ReadToString(fp, &physical_max, 8 * num_signals);
  ReadToString(fp, &digital_min, 8 * num_signals);
  ReadToString(fp, &digital_max, 8 * num_signals);
  ReadToString(fp, &prefiltering, 80 * num_signals);
  ReadToString(fp, &num_samples_in_data_record_str, 8 * num_signals);
  ReadToString(fp, &reserved, 32 * num_signals);

  std::vector<EdfHeader::SignalHeader> signal_headers;
  // Fill in channel parameters and calculate offsets and bit values.
  for (int i = 0; i < num_signals; ++i) {
    EdfHeader::SignalHeader signal_header;
    signal_header.set_label(channel_name.substr(i * 16, 16));
    Format(signal_header.mutable_label());
    // TODO(jjtswan): Need to recalculate this when handling EDF+D.
    signal_header.set_num_samples_per_data_record(
        std::stoi(num_samples_in_data_record_str.substr(i * 8, 8)));
    signal_header.set_transducer_type(transducer_type.substr(i * 80, 80));
    Format(signal_header.mutable_transducer_type());
    signal_header.set_physical_dimension(physical_dimension.substr(i * 8, 8));
    Format(signal_header.mutable_physical_dimension());
    signal_header.set_physical_max(physical_max.substr(i * 8, 8).c_str());
    Format(signal_header.mutable_physical_max());
    signal_header.set_physical_min(physical_min.substr(i * 8, 8).c_str());
    Format(signal_header.mutable_physical_min());
    signal_header.set_digital_max(digital_max.substr(i * 8, 8));
    Format(signal_header.mutable_digital_max());
    signal_header.set_digital_min(digital_min.substr(i * 8, 8));
    Format(signal_header.mutable_digital_min());
    signal_header.set_prefiltering(prefiltering.substr(i * 80, 80));
    Format(signal_header.mutable_prefiltering());
    // NOTE(jjtswan): Is this needed to trigger swap mechanics?
    signal_headers.emplace_back(std::move(signal_header));
  }
  // NOTE(jjtswan): Do I need to use std::move to trigger swap mechanics?
  return signal_headers;
}

StatusOr<IntegerSignal> ParseEdfIntegerSignal(
    EdfFile* fp, int num_samples, const string& signal_label,
    bool return_error_on_num_samples_mismatch) {
  IntegerSignal integer_signal;
  unsigned char sample[2];
  int num_samples_read = 0;
  for (int j = 0; j < num_samples; ++j) {
    const int size = fp->Read(sample, 2);
    if (size < 2) {
      string error_msg = absl::StrCat(
          "Mismatch between actual num samples : ", num_samples_read,
          " and expected num samples : ", num_samples,
          " for signal : ", signal_label);
      if (return_error_on_num_samples_mismatch) {
        return InvalidArgumentError(error_msg);
      } else {
        // NOTE(jjtswan): Not sure why we need this?
        ABSL_RAW_LOG(FATAL, "File opening failed");
        return integer_signal;
      }
    }
    const int16_t value =
        static_cast<int16_t>(sample[0]) | static_cast<int16_t>(sample[1] << 8);
    integer_signal.add_samples(value);
  }
  return integer_signal;
}

StatusOr<TimeStampedAnnotationList> ParseTimeStampAnnotationListStart(
    const string& time_span) {
  TimeStampedAnnotationList tal;

  // Expect: +1.5<21>4<20>
  std::vector<string> v = absl::StrSplit(time_span, absl::ByChar(kChar21));
  if (v.size() != 1 && v.size() != 2) {
    return InvalidArgumentError("Could not parse time span: " + time_span);
  }
  string start_str = v.at(0);
  if (start_str.empty()) {
    return InvalidArgumentError("Empty time start");
  }
  tal.set_start_offset_seconds(std::move(start_str));

  const string duration_str = (v.size() == 2) ? v.at(1) : "";
  tal.set_duration_seconds(duration_str);

  return tal;
}

StatusOr<std::vector<TimeStampedAnnotationList>>
ParseTimeStampedAnnotationLists(const string& time_span,
                                const std::vector<string>& annotations) {
  std::vector<TimeStampedAnnotationList> tals;

  TimeStampedAnnotationList first_tal;
  ASSIGN_OR_RETURN(first_tal, ParseTimeStampAnnotationListStart(time_span));

  tals.emplace_back(std::move(first_tal));
  // Parse annotations in this list.
  for (const auto& annotation : annotations) {
    // Handle incorrectly formatted annotation list, where there is no "0" byte
    // separating TALs.  As an approximation, we look for strings that start
    // with a "+".  Parse it as the new start / end, and continue parsing more
    // annotations.
    if (annotation.size() > 1 &&
        (annotation[0] == '+' || annotation[0] == '-')) {
      TimeStampedAnnotationList other_tal;
      ASSIGN_OR_RETURN(other_tal, ParseTimeStampAnnotationListStart(time_span));
      tals.emplace_back(std::move(other_tal));
      continue;
    }

    // NOTE(jjtswan): This could be UTF8. Should we coerce into ANSI?
    tals.back().add_annotations(annotation);
  }
  return tals;
}

// TODO(jjtswan): Refactor and move to edf_lib.
StatusOr<AnnotationSignal> ParseEdfAnnotationSignal(EdfFile* fp,
                                                    int channel_bytes,
                                                    bool only_parse_first) {
  auto channel_end = fp->Tell() + channel_bytes;
  auto bytes_left = channel_bytes;
  std::vector<TimeStampedAnnotationList> tals;
  bool found_end_of_all_lists = false;
  while (bytes_left > 0) {
    // Read the timespan first.
    string time_span;
    ASSIGN_OR_RETURN(time_span,
                     GetNextToken(fp, &bytes_left, &found_end_of_all_lists));
    if (found_end_of_all_lists) {
      // No more annotations left for this channel in this data record.
      break;
    }

    // Read annotations.
    std::vector<string> annotations;
    bool found_end_of_list = false;
    while (bytes_left > 0) {
      // Read the timespan first.
      string annotation;
      ASSIGN_OR_RETURN(annotation,
                       GetNextToken(fp, &bytes_left, &found_end_of_list));
      if (found_end_of_list) {
        // End of this list, but there might be more lists.
        break;
      }
      annotations.emplace_back(annotation);
    }
    std::vector<TimeStampedAnnotationList> new_tals;
    ASSIGN_OR_RETURN(new_tals,
                     ParseTimeStampedAnnotationLists(time_span, annotations));
    std::move(new_tals.begin(), new_tals.end(), std::back_inserter(tals));
    if (only_parse_first) {
      // If we're only parsing the first list.
      break;
    }
  }
  if (bytes_left < 0) {
    ABSL_RAW_LOG(
        FATAL,
        "Logic error: Negative bytes left in parsing EDF Annotation channel.");
  }

  if (fp->SeekFromBegin(channel_end) != OkStatus()) {
    ABSL_RAW_LOG(FATAL, "Error: Seek failed.");
  }

  AnnotationSignal annotation_signal;
  for (const auto& it : tals) {
    *(annotation_signal.add_tals()) = it;
  }
  return annotation_signal;
}

// Parses .edf file and stores output into proto representation of edf format.
StatusOr<Edf> ParseEdfToEdfProto(const string& filename) {
  EdfOssFile fp = EdfOssFile(filename.c_str(), "r");
  Edf edf;

  RETURN_IF_ERROR(ParseEdfHeader(&fp, edf.mutable_header()));

  std::vector<EdfHeader::SignalHeader> signal_headers;
  ASSIGN_OR_RETURN(signal_headers,
                   ParseEdfSignalHeaders(&fp, edf.header().num_signals()));
  for (const auto& it : signal_headers) {
    *(edf.mutable_header()->add_signal_headers()) = it;
  }

  for (int i = 0; i < edf.header().num_data_records(); ++i) {
    DataRecord data_record;
    for (const auto& signal_header : edf.header().signal_headers()) {
      Signal signal;
      int num_samples = signal_header.num_samples_per_data_record();
      if (signal_header.label() == kEdfAnnotationsLabel) {
        int channel_bytes = num_samples * 2 /* bytes_per_sample */;
        AnnotationSignal annotation_signal;
        ASSIGN_OR_RETURN(annotation_signal,
                         ParseEdfAnnotationSignal(&fp, channel_bytes, false));
        *signal.mutable_annotations() = std::move(annotation_signal);
      } else {
        IntegerSignal integer_signal;
        ASSIGN_OR_RETURN(integer_signal,
                         ParseEdfIntegerSignal(&fp, num_samples,
                                               signal_header.label(), true));
        *signal.mutable_integers() = std::move(integer_signal);
      }
      *data_record.add_signals() = std::move(signal);
    }
    *edf.add_data_records() = std::move(data_record);
  }
  return edf;
}

// Write EDF Protos to EDF Files -----------------------------------------------

int WriteString(const string& str, EdfFile* fp) {
  auto bytes = fp->Write(str.c_str(), str.length());
  if (bytes != str.length()) {
    ABSL_RAW_LOG(FATAL, "File writing failed");
  }
  return bytes;
}

// Move to anonymous namespace.
StatusOr<string> MakeField(const string& field, size_t len) {
  if (field.size() > len) {
    return InvalidArgumentError(absl::StrCat(
        "Field is too large for specified size of ", len, ": ", field));
  }
  // Pad with ascii spaces as per EDF spec.
  string padded_field(len, ' ');
  if (padded_field.length() != len) {
    ABSL_RAW_LOG(FATAL, "Padded field != length.");
  }
  return padded_field.replace(0, field.length(), field);
}

Status WriteEdfHeader(const EdfHeader& header, EdfFile* fp) {
  // Edf Header is expected to be padded with spaces.
  int bytes = 0;
  string data;
  ASSIGN_OR_RETURN(data, MakeField(header.version(), 8));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(
      data, MakeField(header.local_patient_identification().full_text(), 80));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(
      data, MakeField(header.local_recording_information().full_text(), 80));
  bytes += WriteString(data, fp);

  ASSIGN_OR_RETURN(data, MakeField(header.recording_start_date(), 8));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(data, MakeField(header.recording_start_time(), 8));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(data, MakeField(absl::StrCat(header.num_header_bytes()), 8));
  bytes += WriteString(data, fp);

  if (header.type_from_reserved() == EdfHeader::EDF_PLUS_C) {
    ASSIGN_OR_RETURN(data, MakeField(kEdfPlusC, 44));
  } else if (header.type_from_reserved() == EdfHeader::EDF_PLUS_D) {
    ASSIGN_OR_RETURN(data, MakeField(kEdfPlusD, 44));
  } else {
    data = string(44, ' ');
  }
  bytes += WriteString(data, fp);

  ASSIGN_OR_RETURN(data, MakeField(absl::StrCat(header.num_data_records()), 8));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(data, MakeField(header.num_seconds_per_data_record(), 8));
  bytes += WriteString(data, fp);
  ASSIGN_OR_RETURN(data, MakeField(absl::StrCat(header.num_signals()), 4));
  bytes += WriteString(data, fp);

  if (bytes != 256) {
    ABSL_RAW_LOG(FATAL,
                 "Expected to have written 256 bytes for initial "
                 "header block, actually wrote %d",
                 bytes);
  }

  if (static_cast<unsigned int>(header.signal_headers_size()) !=
      static_cast<unsigned int>(header.num_signals())) {
    ABSL_RAW_LOG(FATAL, "signal headers size != num_signals");
  }

  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.label(), 16));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.transducer_type(), 80));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.physical_dimension(), 8));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.physical_min(), 8));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.physical_max(), 8));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.digital_min(), 8));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.digital_max(), 8));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(data, MakeField(signal_header.prefiltering(), 80));
    WriteString(data, fp);
  }
  for (const auto& signal_header : header.signal_headers()) {
    ASSIGN_OR_RETURN(
        data,
        MakeField(absl::StrCat(signal_header.num_samples_per_data_record()),
                  8));
    WriteString(data, fp);
  }
  // Reserved per signal header.
  WriteString(string(header.signal_headers_size() * 32, ' '), fp);

  return OkStatus();
}

Status WriteIntegerSignal(const IntegerSignal& integer_signal, EdfFile* fp) {
  unsigned char data[2];
  // Maybe check that we have appropriate number of samples.
  for (const auto& sample : integer_signal.samples()) {
    // Maybe check if in bounds of digital min / max?
    // Maybe check if in bounds of [-2^15, 2^15 - 1]
    const int16_t value = static_cast<int16_t>(sample);
    data[0] = value & 0xFF;  // Drop top 8 bits.
    data[1] = value >> 8;
    fp->Write(&data, 2);
  }
  return OkStatus();
}

Status WriteAnnotationSignal(const AnnotationSignal& annotation_signal,
                             int signal_bytes, EdfFile* fp) {
  // Maybe check that we don't exceed total bytes allotted.
  int bytes_written = 0;
  const char null_data[] = {kChar0};
  for (const auto& tal : annotation_signal.tals()) {
    bytes_written += WriteString(tal.start_offset_seconds(), fp);
    if (tal.has_duration_seconds() && !tal.duration_seconds().empty()) {
      bytes_written += WriteString("\x15", fp);
      bytes_written += WriteString(tal.duration_seconds(), fp);
    }
    bytes_written += WriteString("\x14", fp);
    for (const string& annotation : tal.annotations()) {
      bytes_written += WriteString(annotation, fp);
      bytes_written += WriteString("\x14", fp);
    }
    bytes_written += fp->Write(&null_data, 1);
  }

  if (bytes_written > signal_bytes) {
    return InvalidArgumentError("Annnotations exceeded allocated bytes.");
  }

  if (bytes_written < signal_bytes) {
    int num_nulls = signal_bytes - bytes_written;
    string padding(num_nulls, kChar0);
    fp->Write(padding.c_str(), num_nulls);
  }

  return OkStatus();
}

Status WriteEdfDataRecord(const EdfHeader& header,
                          const DataRecord& data_record, EdfFile* fp) {
  for (int i = 0; i < data_record.signals_size(); ++i) {
    const auto& signal = data_record.signals(i);
    const auto& signal_header = header.signal_headers(i);
    if (signal.has_integers()) {
      RETURN_IF_ERROR(WriteIntegerSignal(signal.integers(), fp));
    } else if (signal.has_annotations()) {
      int signal_bytes = signal_header.num_samples_per_data_record() *
                         2 /* bytes per sample */;
      RETURN_IF_ERROR(
          WriteAnnotationSignal(signal.annotations(), signal_bytes, fp));
    } else {
      return InvalidArgumentError("Signal has unspecified value.");
    }
  }
  return OkStatus();
}

Status WriteEdf(const eeg_modelling::Edf& edf, const string& filename) {
  EdfOssFile fp = EdfOssFile(filename.c_str(), "w");
  RETURN_IF_ERROR(WriteEdfHeader(edf.header(), &fp));

  for (const auto& data_record : edf.data_records()) {
    RETURN_IF_ERROR(WriteEdfDataRecord(edf.header(), data_record, &fp));
  }
  return OkStatus();
}
}  // namespace eeg_modelling
