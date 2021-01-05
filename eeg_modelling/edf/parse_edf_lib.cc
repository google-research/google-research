// Copyright 2021 The Google Research Authors.
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

#include "edf/parse_edf_lib.h"

#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "edf/base/canonical_errors.h"
#include "edf/base/status.h"
#include "edf/base/status_macros.h"
#include "edf/base/statusor.h"
#include "edf/base/time_proto_util.h"
#include "edf/edf.h"
#include "edf/edf_util.h"
#include "edf/proto/annotation.pb.h"
#include "edf/proto/edf.pb.h"
#include "edf/proto/event.pb.h"
#include "edf/proto/segment.pb.h"

namespace eeg_modelling {

namespace {

StatusOr<absl::Time> GetEventStartTime(const Event& event) {
  if (!event.has_start_time()) {
    return InvalidArgumentError(absl::StrCat(
        "Event does not have start time. Label is ", event.label()));
  }
  return DecodeGoogleApiProto(event.start_time());
}

StatusOr<absl::Time> GetEventEndTime(const Event& event) {
  if (!event.has_end_time()) {
    return GetEventStartTime(event);
  }
  return DecodeGoogleApiProto(event.end_time());
}

StatusOr<bool> EventContainedByTimeSpan(const Event& event,
                                        const absl::Time start,
                                        const absl::Time end,
                                        bool include_end_for_instant) {
  if (start == absl::InfiniteFuture() || end == absl::InfinitePast()) {
    return InvalidArgumentError(
        absl::StrCat("Invalid start or end time for bounds checking. Label is ",
                     event.label()));
  }

  // Get the start and end time of the event.
  absl::Time event_start, event_end;
  ASSIGN_OR_RETURN(event_start, GetEventStartTime(event));
  ASSIGN_OR_RETURN(event_end, GetEventEndTime(event));

  if (event_start == event_end) {
    if (include_end_for_instant) {
      // Check that [event_start] contained in [start, end].
      return start <= event_start && event_start <= end;
    } else {
      // Check that [event_start] contained in [start, end).
      return start <= event_start && event_start < end;
    }
  }
  // Check that event [event_start, event_end) is inside given [start, end).
  return start <= event_start && event_end <= end;
}

StatusOr<bool> EventIntersectsTimeSpan(const Event& event,
                                       const absl::Time start,
                                       const absl::Time end,
                                       bool include_end_for_instant) {
  if (start == absl::InfiniteFuture() || end == absl::InfinitePast()) {
    return InvalidArgumentError(
        absl::StrCat("Invalid start or end time for bounds checking. Label is ",
                     event.label()));
  }

  // Get the start and end time of the event.
  absl::Time event_start, event_end;
  ASSIGN_OR_RETURN(event_start, GetEventStartTime(event));
  ASSIGN_OR_RETURN(event_end, GetEventEndTime(event));

  if (event_start == event_end) {
    if (include_end_for_instant) {
      // Check that [event_start] contained in [start, end].
      return start <= event_start && event_start <= end;
    } else {
      // Check that [event_start] contained in [start, end).
      return start <= event_start && event_start < end;
    }
  }

  // Check that event [event_start, event_end) intersects example [start, end).
  return !(event_end <= start || end <= event_start);
}

StatusOr<bool> EventIntersectsSegmentTimeSpan(const Event& event,
                                              const Segment& segment,
                                              bool include_end_for_instant) {
  // Get start and end time for the Segment.
  if (!segment.has_start_time()) {
    return InvalidArgumentError("Segment does not have start time.");
  }
  absl::Time segment_start;
  ASSIGN_OR_RETURN(segment_start, DecodeGoogleApiProto(segment.start_time()));

  const absl::Time segment_end =
      segment_start + absl::Seconds(segment.num_data_records() *
                                    segment.data_record_duration_sec());

  return EventIntersectsTimeSpan(event, segment_start, segment_end,
                                 include_end_for_instant);
}

constexpr char kEdfAnnotationsLabel[] = "EDF Annotations";

Status ParseEdfHeaderToSegment(const EdfHeader& header, Segment* segment) {
  segment->set_version(std::stoi(header.version()));

  *segment->mutable_patient_id() =
      header.local_patient_identification().full_text();
  *segment->mutable_recording_id() =
      header.local_recording_information().full_text();

  absl::Time start_time;
  google::protobuf::Timestamp start_timestamp;
  ASSIGN_OR_RETURN(start_time, ParseEdfStartTime(header));
  ASSIGN_OR_RETURN(start_timestamp, EncodeGoogleApiProto(start_time));
  *segment->mutable_start_time() = start_timestamp;

  if (header.num_data_records() < 0) {
    return InvalidArgumentError(
        absl::StrCat("Number of records in EDF file is invalid: ",
                     header.num_data_records()));
  }
  segment->set_num_data_records(header.num_data_records());

  segment->set_data_record_duration_sec(
      std::strtod(header.num_seconds_per_data_record().c_str(), nullptr));

  segment->set_num_signals(header.num_signals());
  return OkStatus();
}

StatusOr<std::vector<Segment::Channel>> ParseChannelParams(
    const EdfHeader& header, int num_signals, int num_data_records,
    double data_record_duration_sec) {
  std::vector<Segment::Channel> channels;
  // Fill in channel parameters and calculate offsets and bit values.
  for (const auto& signal_header : header.signal_headers()) {
    Segment::Channel channel;
    channel.set_name(signal_header.label());
    int num_samples_in_data_record =
        signal_header.num_samples_per_data_record();
    channel.set_num_samples_in_data_record(num_samples_in_data_record);
    // NOTE(jjtswan): Technically, only EDF+ files can have annotations.  But,
    // seems unlikely that an EDF file would have channel named "EDF
    // Annotations" without intending for it to be interpreted as such.
    if (channel.name() != kEdfAnnotationsLabel) {
      // These fields are only meaningful for non-annotation channels.
      channel.set_transducer(signal_header.transducer_type());
      channel.set_physical_dimension(signal_header.physical_dimension());
      channel.set_physical_max(std::stod(signal_header.physical_max().c_str()));
      channel.set_physical_min(std::stod(signal_header.physical_min().c_str()));
      if (std::isnan(channel.physical_max())) {
        return InvalidArgumentError(absl::StrCat("NaN physical max from : ",
                                                 signal_header.physical_max()));
      }
      if (std::isnan(channel.physical_min())) {
        return InvalidArgumentError(absl::StrCat("NaN physical min from : ",
                                                 signal_header.physical_min()));
      }
      // EDF spec specifies that physical_min and physical_max must contain
      // different values. Support the case of physical_min == physical_max == 0
      // since some clients require this.
      if (channel.physical_min() == channel.physical_max()) {
        ABSL_RAW_LOG(WARNING,
                     "Channel : %s has equal physical min/max: values."
                     "The min is: %s, max is %s",
                     channel.name().c_str(),
                     signal_header.physical_min().c_str(),
                     signal_header.physical_max().c_str());
        if (channel.physical_min() != 0.0f) {
          return InvalidArgumentError(
              absl::StrCat("Physical min and max are equal and non-zero : ",
                           channel.physical_max()));
        }
      }
      channel.set_digital_max(std::stoi(signal_header.digital_max()));
      channel.set_digital_min(std::stoi(signal_header.digital_min()));
      if (std::isnan(channel.digital_max())) {
        return InvalidArgumentError(absl::StrCat("NaN digital max from : ",
                                                 signal_header.digital_max()));
      }
      if (std::isnan(channel.digital_min())) {
        return InvalidArgumentError(absl::StrCat("NaN digital min from : ",
                                                 signal_header.digital_min()));
      }
      if (channel.digital_min() >= channel.digital_max()) {
        return InvalidArgumentError(absl::StrCat(
            "Invalid digital min/max: min is ", signal_header.digital_min(),
            " max is ", signal_header.digital_max()));
      }
      channel.set_prefilter(signal_header.prefiltering());
      // TODO(jjtswan): Need to recalculate this when handling EDF+D.
      channel.set_num_samples(num_samples_in_data_record * num_data_records);
      double bitvalue = 0.0;
      double offset = 0.0;
      if (channel.physical_max() != channel.physical_min()) {
        bitvalue =
            (channel.physical_max() - channel.physical_min()) /
            static_cast<double>(channel.digital_max() - channel.digital_min());
        offset = channel.physical_max() / bitvalue -
                 static_cast<double>(channel.digital_max());
      }
      if (std::isnan(bitvalue)) {
        return InvalidArgumentError(
            "NaN bitvalue from digital/physical min/max.");
      }
      channel.set_bitvalue(bitvalue);
      channel.set_offset(offset);
      const double sampling_frequency_hz =
          channel.num_samples_in_data_record() / data_record_duration_sec;
      channel.set_sampling_frequency_hz(sampling_frequency_hz);
    }
    // NOTE(jjtswan): Is this needed to trigger swap mechanics?
    channels.emplace_back(std::move(channel));
  }
  // NOTE(jjtswan): Do I need to use std::move to trigger swap mechanics?
  return channels;
}

double DigitalToPhysicalValue(int16_t digital_value, double bitvalue,
                              double offset) {
  return bitvalue * (offset + static_cast<double>(digital_value));
}

int ParseSamplesForChannel(const IntegerSignal& integer_signal, int num_samples,
                           Segment::Channel* channel) {
  // TODO(jjtswan): Consider pushing this to edf.cc as extra validation.
  if (integer_signal.samples_size() != num_samples) {
    channel->set_num_samples_consistent(false);
  }
  for (int i = 0; i < integer_signal.samples_size(); ++i) {
    const double physical_value = DigitalToPhysicalValue(
        integer_signal.samples(i), channel->bitvalue(), channel->offset());
    channel->add_samples(physical_value);
  }
  return integer_signal.samples_size();
}

bool AreSplitsInMiddleOfRecord(const std::vector<absl::Time>& splits,
                               const absl::Time start,
                               const absl::Duration delta) {
  const absl::Time end = start + delta;
  for (const auto split : splits) {
    if (start < split && split < end) {
      return true;
    }
  }
  return false;
}

string GenerateSegmentId(const Segment& template_segment,
                         const Segment& segment) {
  absl::Time template_start_time =
      DecodeGoogleApiProto(template_segment.start_time()).ValueOrDie();
  absl::Time start_time =
      DecodeGoogleApiProto(segment.start_time()).ValueOrDie();
  // Append time marker to filename, since we use that as a unique key value.
  return absl::StrCat(template_segment.filename(), "#",
                      absl::ToInt64Seconds(start_time - template_start_time));
}

StatusOr<Segment> ParseDataRecordsForOneSegment(
    const Edf& edf, const Segment& template_segment,
    const std::vector<Segment::Channel>& template_channels,
    const std::vector<absl::Time>& splits, bool parse_samples, int start_record,
    const string& split_segment_by_annotation_with_prefix) {
  // Copy the templates.
  Segment segment(template_segment);
  std::vector<Segment::Channel> channels(template_channels);

  bool is_edf_d = edf.header().type_from_reserved() == EdfHeader::EDF_PLUS_D;
  absl::Time expected_time;
  // TODO(jjtswan): May need to change the seconds to nano-seconds or something
  // that has finer time resolution?
  absl::Duration delta_time =
      absl::Seconds(template_segment.data_record_duration_sec());

  // Each sample value is stored in 1 byte 2s complement format.
  // Loop through each record, and for each record through each channel to
  // read its samples.
  int record;
  for (record = start_record; record < edf.data_records_size(); ++record) {
    // TODO(jjtswan): Check that this lines up with EDF+C.
    if (is_edf_d) {
      // Check if we hit a discontinuity in the timeline.
      absl::Time record_start_time;
      ASSIGN_OR_RETURN(record_start_time,
                       GetEdfDataRecordStartTime(edf, record));
      if (record == start_record) {
        *segment.mutable_start_time() =
            EncodeGoogleApiProto(record_start_time).ValueOrDie();
        expected_time = record_start_time;
      } else if (record_start_time != expected_time) {
        // We have a break in the continuity.  Stop Segment here.
        break;
      } else if (!split_segment_by_annotation_with_prefix.empty() &&
                 std::find(std::begin(splits), std::end(splits),
                           record_start_time) != std::end(splits)) {
        // The start time matches a split defined by the annotations, so stop.
        break;
      }
      if (!split_segment_by_annotation_with_prefix.empty()) {
        // We currently don't handle this case, so better to drop it.
        if (AreSplitsInMiddleOfRecord(splits, record_start_time, delta_time)) {
          return InvalidArgumentError(absl::StrCat(
              "Unhandled situation - found split time in middle of record. "
              "Segment : ",
              segment.filename()));
        }
      }
      expected_time += delta_time;
    }
    for (size_t i = 0; i < channels.size(); ++i) {
      Segment::Channel* channel = &channels.at(i);
      const string name = channel->name();
      const int num_samples_per_record = channel->num_samples_in_data_record();
      if (name == kEdfAnnotationsLabel || !parse_samples) {
        // Skip annotations channels, which contain no numeric data.
        // Or skip everything if we're not parsing samples.
        continue;
      }
      ParseSamplesForChannel(edf.data_records(record).signals(i).integers(),
                             num_samples_per_record, channel);
    }
  }
  if (!(record == edf.data_records_size() || is_edf_d)) {
    ABSL_RAW_LOG(FATAL, "Non-EDF+D should parse through all records.");
  }

  // Fix up a few extra fields, then add channels with data to final segment.
  int num_inconsistent_channels = 0;
  for (auto channel : channels) {
    if (channel.name() != kEdfAnnotationsLabel) {
      channel.set_num_samples(channel.samples_size());
      if (!channel.num_samples_consistent()) {
        ++num_inconsistent_channels;
      }
      *segment.add_channel() = std::move(channel);
    }
  }

  if (num_inconsistent_channels != 0) {
    return InvalidArgumentError(
        absl::StrCat("Num inconsistent channels : ", num_inconsistent_channels,
                     " out of total num channels : ", channels.size()));
  }

  segment.set_num_data_records(record - start_record);
  segment.set_num_signals(segment.channel_size());
  // Override with what is hopefully a unique key for the segment.
  segment.set_segment_id(GenerateSegmentId(template_segment, segment));

  return segment;
}

StatusOr<std::vector<absl::Time>> FindAnnotationDefinedSplits(
    const Edf& edf, const string& starts_with) {
  absl::Time recording_start_time;
  ASSIGN_OR_RETURN(recording_start_time, ParseEdfStartTime(edf.header()));

  std::vector<absl::Time> splits;
  // Iterate over all of the annotation labels, and return a list of the times
  // of those labels that match the given starts_with.
  for (int i = 0; i < edf.data_records_size(); ++i) {
    for (int j = 0; j < edf.data_records(i).signals_size(); ++j) {
      if (!edf.data_records(i).signals(j).has_annotations()) {
        // Skip non-annotation signal.
        continue;
      }
      const auto& annotations = edf.data_records(i).signals(j).annotations();
      for (int k = 0; k < annotations.tals_size(); ++k) {
        for (int m = 0; m < annotations.tals(k).annotations_size(); ++m) {
          if (absl::StartsWith(annotations.tals(k).annotations(m),
                               starts_with)) {
            absl::Time split_time;
            ASSIGN_OR_RETURN(
                split_time,
                GetStartTime(recording_start_time,
                             annotations.tals(k).start_offset_seconds()));
            splits.push_back(split_time);
            // Skip the rest of the entries since they share the same
            // start, being in the same tal.
            break;
          }
        }
      }
    }
  }
  return std::move(splits);
}

StatusOr<std::vector<Segment>> ParseDataRecordsToSegments(
    const Edf& edf, const Segment& template_segment,
    const std::vector<Segment::Channel>& template_channels, bool parse_samples,
    const string& split_segment_by_annotation_with_prefix) {
  std::vector<Segment> segments;

  std::vector<absl::Time> splits;
  // TODO(jjtswan): It turns out that the EDF spec specifies that any annotation
  // that is contained in the timespan of a data record should exist in that
  // data record.  That means that we don't have to pre-scan to find the splits.
  if (!split_segment_by_annotation_with_prefix.empty()) {
    ASSIGN_OR_RETURN(splits, FindAnnotationDefinedSplits(
                                 edf, split_segment_by_annotation_with_prefix));
  }

  // Parse all records.
  int current_record = 0;
  while (current_record < edf.header().num_data_records()) {
    Segment segment;
    ASSIGN_OR_RETURN(segment, ParseDataRecordsForOneSegment(
                                  edf, template_segment, template_channels,
                                  splits, parse_samples, current_record,
                                  split_segment_by_annotation_with_prefix));
    current_record += segment.num_data_records();
    segments.push_back(std::move(segment));
  }
  if (current_record != edf.header().num_data_records()) {
    ABSL_RAW_LOG(FATAL, "current_record != num_data_records");
  }
  return segments;
}

StatusOr<std::vector<Event>> ParseEventsFromTimeStampedAnnotationList(
    const TimeStampedAnnotationList& tal,
    const absl::Time recording_start_time) {
  std::vector<Event> events;
  for (const auto& annotation : tal.annotations()) {
    Event event;
    event.set_label(annotation);

    std::tuple<absl::Time, absl::Time> start_end;
    ASSIGN_OR_RETURN(start_end, GetStartEndTimes(recording_start_time,
                                                 tal.start_offset_seconds(),
                                                 tal.duration_seconds()));
    google::protobuf::Timestamp event_start_time;
    ASSIGN_OR_RETURN(event_start_time,
                     EncodeGoogleApiProto(std::get<0>(start_end)));
    *event.mutable_start_time() = event_start_time;

    if (std::get<0>(start_end) != std::get<1>(start_end)) {
      // Only set this if we have a non-zero duration.
      google::protobuf::Timestamp event_end_time;
      ASSIGN_OR_RETURN(event_end_time,
                       EncodeGoogleApiProto(std::get<1>(start_end)));
      *event.mutable_end_time() = event_end_time;
    }

    events.emplace_back(event);
  }
  return events;
}

StatusOr<std::vector<Event>> ParseEdfAnnotationsChannel(
    const AnnotationSignal& annotation_signal,
    const absl::Time recording_start_time) {
  std::vector<Event> events;
  // Parse annotations in this list.
  for (const auto& tal : annotation_signal.tals()) {
    std::vector<Event> new_events;
    ASSIGN_OR_RETURN(new_events, ParseEventsFromTimeStampedAnnotationList(
                                     tal, recording_start_time));
    std::move(new_events.begin(), new_events.end(), std::back_inserter(events));
  }
  return events;
}

StatusOr<std::vector<Event>> GetRelevantEventsFromDataRecord(
    const AnnotationSignal& annotations, const absl::Time recording_start_time,
    const absl::Time start_time, const absl::Time end_time,
    bool is_first_edf_annotation_channel) {
  std::vector<Event> all_events;
  ASSIGN_OR_RETURN(all_events, ParseEdfAnnotationsChannel(
                                   annotations, recording_start_time));
  std::vector<Event> filtered_events;
  for (size_t i = 0; i < all_events.size(); ++i) {
    if (is_first_edf_annotation_channel && i == 0) {
      // Skip first event in the edf annotation channel as it is used for
      // marking the start time of the data record.
      continue;
    }

    bool intersects;
    ASSIGN_OR_RETURN(
        intersects,
        EventIntersectsTimeSpan(
            all_events.at(i), start_time, end_time,
            // Include events that fall at the very end of the given time span.
            true /*include_end_for_instant*/));
    if (!intersects) {
      // The event is outside of the given time bound.
      continue;
    }
    bool contained;
    ASSIGN_OR_RETURN(
        contained,
        EventContainedByTimeSpan(
            all_events.at(i), start_time, end_time,
            // Include events that fall at the very end of the given time span.
            true /*include_end_for_instant*/));
    if (!contained) {
      return InvalidArgumentError(
          "EDF annotation is not fully contained in the segment timespan");
    }

    filtered_events.emplace_back(std::move(all_events.at(i)));
  }
  return filtered_events;
}

StatusOr<Annotation> ParseDataRecordsToAnnotation(
    const Edf& edf, const Segment& template_segment,
    const std::vector<Segment::Channel>& template_channels,
    const absl::Time start_time, const absl::Time end_time) {
  absl::Time recording_start_time =
      DecodeGoogleApiProto(template_segment.start_time()).ValueOrDie();

  std::vector<Event> all_events;
  for (int i = 0; i < edf.data_records_size(); ++i) {
    int first_edf_annotation_channel = true;
    for (size_t j = 0; j < template_channels.size(); ++j) {
      const auto& channel = template_channels.at(j);
      if (channel.name() != kEdfAnnotationsLabel) {
        continue;
      }
      std::vector<Event> events;
      ASSIGN_OR_RETURN(
          events, GetRelevantEventsFromDataRecord(
                      edf.data_records(i).signals(j).annotations(),
                      recording_start_time, start_time, end_time,
                      // Throw away the 1st annotation in the 1st edf annotation
                      // channel in the data record. That annotation is not a
                      // real annotation.  It is blank and is used to indicate
                      // the start time of the data channel.
                      first_edf_annotation_channel));
      first_edf_annotation_channel = false;

      std::move(std::begin(events), std::end(events),
                std::back_inserter(all_events));
    }
  }

  Annotation annotation;
  annotation.set_type(RAW_TEXT);
  *annotation.mutable_events()->mutable_event() = {all_events.begin(),
                                                   all_events.end()};
  return annotation;
}

}  // namespace

StatusOr<std::vector<Segment>> ParseEdfToSegmentProtos(
    const Edf& edf, const string& session_id, const string& filename,
    bool parse_samples, const string& split_segment_by_annotation_with_prefix) {
  Segment template_segment;
  template_segment.set_filename(filename);
  template_segment.set_session_id(session_id);

  RETURN_IF_ERROR(ParseEdfHeaderToSegment(edf.header(), &template_segment));

  // Parse channel parameters.
  std::vector<Segment::Channel> template_channels;
  ASSIGN_OR_RETURN(
      template_channels,
      ParseChannelParams(edf.header(), template_segment.num_signals(),
                         template_segment.num_data_records(),
                         template_segment.data_record_duration_sec()));

  // Parse channel samples.
  return ParseDataRecordsToSegments(edf, template_segment, template_channels,
                                    parse_samples,
                                    split_segment_by_annotation_with_prefix);
}

StatusOr<std::vector<Segment>> ParseEdfToSegmentProtos(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix) {
  Edf edf;
  ASSIGN_OR_RETURN(edf, ParseEdfToEdfProto(filename));
  return ParseEdfToSegmentProtos(edf, session_id, filename,
                                 true /* parse_samples */,
                                 split_segment_by_annotation_with_prefix);
}

StatusOr<Segment> ParseEdfToSegmentProto(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix) {
  std::vector<Segment> segments;
  ASSIGN_OR_RETURN(segments, ParseEdfToSegmentProtos(
                                 session_id, filename,
                                 split_segment_by_annotation_with_prefix));

  if (segments.size() != 1) {
    return InvalidArgumentError(absl::StrCat(
        "Expected to parse 1 segment, but parsed ", segments.size()));
  }
  return segments.at(0);
}

StatusOr<Annotation> ParseEdfToAnnotationProto(const Edf& edf,
                                               const absl::Time start_time,
                                               const absl::Time end_time) {
  if (start_time > end_time) {
    return InvalidArgumentError(
        "Negative timespan selected for parsing annotations from EDF.");
  } else if (start_time == end_time) {
    return InvalidArgumentError(
        "Zero length timespan selected for parsing annotations from EDF.");
  }

  Segment template_segment;
  RETURN_IF_ERROR(ParseEdfHeaderToSegment(edf.header(), &template_segment));

  // Parse channel parameters.
  std::vector<Segment::Channel> template_channels;
  ASSIGN_OR_RETURN(
      template_channels,
      ParseChannelParams(edf.header(), template_segment.num_signals(),
                         template_segment.num_data_records(),
                         template_segment.data_record_duration_sec()));

  // Parse channel samples.
  return ParseDataRecordsToAnnotation(edf, template_segment, template_channels,
                                      start_time, end_time);
}

StatusOr<Annotation> ParseEdfToAnnotationProto(
    const string& filename, const absl::Time start_time,
    const absl::Time end_time,
    const string& split_segment_by_annotation_with_prefix) {
  Edf edf;
  ASSIGN_OR_RETURN(edf, ParseEdfToEdfProto(filename));
  return ParseEdfToAnnotationProto(edf, start_time, end_time);
}

StatusOr<std::vector<Annotation>> ParseEdfToAnnotationProtoPerSegment(
    const string& session_id, const string& filename,
    const string& split_segment_by_annotation_with_prefix) {
  Edf edf;
  ASSIGN_OR_RETURN(edf, ParseEdfToEdfProto(filename));

  // Pull out all annotations.
  Annotation all;
  ASSIGN_OR_RETURN(all, ParseEdfToAnnotationProto(edf, absl::InfinitePast(),
                                                  absl::InfiniteFuture()));
  // TODO(jjtswan): Modify so we find all segment time boundaries from the edf
  // proto, instead of parsing into actual segments.
  std::vector<Segment> segments;
  ASSIGN_OR_RETURN(segments, ParseEdfToSegmentProtos(
                                 edf, session_id, filename,
                                 // Don't parse actual sample data.
                                 false /*parse_samples*/,
                                 split_segment_by_annotation_with_prefix));

  // Split up the annotations by segments.
  std::vector<Annotation> annotations_per_segment;
  annotations_per_segment.reserve(segments.size());
  for (const auto& segment : segments) {
    Annotation annotation;
    annotation.set_segment_id(segment.segment_id());
    annotation.set_patient_id(segment.patient_id());
    annotation.set_type(RAW_TEXT);
    for (const auto& event : all.events().event()) {
      bool intersects;
      ASSIGN_OR_RETURN(intersects,
                       EventIntersectsSegmentTimeSpan(
                           event, segment, false /*include_end_for_instant*/));
      if (intersects) {
        *annotation.mutable_events()->add_event() = event;
      }
    }
    annotations_per_segment.push_back(std::move(annotation));
  }
  return annotations_per_segment;
}

}  // namespace eeg_modelling
