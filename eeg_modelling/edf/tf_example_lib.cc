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

#include "edf/tf_example_lib.h"

#include <string>

#include "google/protobuf/timestamp.pb.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "edf/base/canonical_errors.h"
#include "edf/base/status_macros.h"
#include "edf/base/time_proto_util.h"
#include "edf/proto/event.pb.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"

namespace eeg_modelling {

using tensorflow::Example;
using tensorflow::GetFeatureValues;

using std::string;

// Segment feature names.
constexpr char kDurationFeatureName[] = "segment/duration_sec";
constexpr char kNumChannelsFeatureName[] = "segment/num_channels";
constexpr char kFilenameFeatureName[] = "segment/filename";
constexpr char kSessionIdFeatureName[] = "segment/session_id";
constexpr char kPatientIdStrFeatureName[] = "segment/patient_id_str";
constexpr char kPatientIdFeatureName[] = "segment/patient_id";
constexpr char kPatientGenderFeatureName[] = "segment/patient_gender";
constexpr char kPatientAgeFeatureName[] = "segment/patient_age";
constexpr char kGroupNameFeatureName[] = "group_name";
constexpr char kStartTimeFeatureName[] = "start_time";

// Channel feature group names.
constexpr char kEegGroupName[] = "eeg_channel";
constexpr char kCtgGroupName[] = "ctg_channel";
constexpr char kEcgGroupName[] = "ecg_channel";

// Channel feature field names.
constexpr char kSamplingRateFieldName[] = "sampling_frequency_hz";
constexpr char kSamplesFieldName[] = "samples";
constexpr char kNumSamplesFieldName[] = "num_samples";
constexpr char kUnitsFieldName[] = "units";
constexpr char kMaxFieldName[] = "max";
constexpr char kMinFieldName[] = "min";

constexpr char kRawLabelEventsFeatureName[] = "raw_label_events";

namespace {

string GetFilename(const tensorflow::Example& example) {
  if (!tensorflow::HasFeature<string>(kFilenameFeatureName, example)) {
    return "";
  }
  return tensorflow::GetFeatureValues<string>(kFilenameFeatureName, example)
      .Get(0);
}

string CreateFeatureName(const string& group_name, const string& channel_name,
                         const string& field) {
  const string name_prefix = channel_name.empty()
                                 ? group_name
                                 : absl::StrCat(group_name, "/", channel_name);
  return absl::StrCat(name_prefix, "/", field);
}

string CreateFeatureName(const string& group_name, const string& field) {
  return CreateFeatureName(group_name, "" /* channel_name */, field);
}

std::vector<std::pair<string, tensorflow::Feature::KindCase>> GetFeatureNames(
    const tensorflow::Example& example) {
  const auto& features_map = tensorflow::GetFeatures(example).feature();
  std::vector<std::pair<string, tensorflow::Feature::KindCase>> keys;
  for (const auto& feature : features_map) {
    keys.push_back(std::make_pair(feature.first, feature.second.kind_case()));
  }
  return keys;
}

StatusOr<std::vector<string>> GetChannelFeatureNames(
    const tensorflow::Example& example) {
  std::vector<std::pair<string, tensorflow::Feature::KindCase>>
      feature_names_kind = GetFeatureNames(example);
  // Find the channel names.
  absl::node_hash_set<string> channel_names;
  for (const auto& feature_name_kind : feature_names_kind) {
    const string feature_name = feature_name_kind.first;
    if (feature_name.find(kUnitsFieldName) != string::npos) {
      std::vector<string> strs =
          absl::StrSplit(feature_name, '/', absl::SkipEmpty());
      if (strs.size() != 3) {
        return InvalidArgumentError(
            absl::StrCat("Unexpected feature name does not split into three: ",
                         feature_name));
      }
      const string channel_name = CreateFeatureName(strs[0], strs[1]);
      // Check that this occurs only once.
      if (channel_names.find(channel_name) != channel_names.end()) {
        return InvalidArgumentError(
            absl::StrCat("Duplicate channel names found: ", channel_name));
      }
      channel_names.insert(channel_name);
    }
  }
  const int num_expected_channels =
      tensorflow::GetFeatureValues<int64_t>(kNumChannelsFeatureName, example)
          .Get(0);
  if (num_expected_channels != channel_names.size()) {
    ABSL_RAW_LOG(FATAL, "Invalid TfExample for file: %s",
                 GetFilename(example).c_str());
  }
  return std::vector<string>(channel_names.begin(), channel_names.end());
}

Status HasValidChannelFeatures(const tensorflow::Example& example) {
  // TODO(jjtswan): Add more sanity checks here as needed.
  return GetChannelFeatureNames(example).status();
}

StatusOr<absl::Time> GetStartTime(const tensorflow::Example& example) {
  if (!tensorflow::HasFeature<string>(kStartTimeFeatureName, example)) {
    return absl::InfiniteFuture();
  }
  google::protobuf::Timestamp pb;
  pb.ParseFromString(
      tensorflow::GetFeatureValues<string>(kStartTimeFeatureName, example)
          .Get(0));
  return DecodeGoogleApiProto(pb);
}

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

Status SetRelativeEventTimeSecFromExample(const tensorflow::Example& example,
                                          Event* event) {
  absl::Time ex_start;
  ASSIGN_OR_RETURN(ex_start, GetStartTime(example));
  if (event->has_start_time()) {
    absl::Time event_start;
    ASSIGN_OR_RETURN(event_start, GetEventStartTime(*event));
    event->set_start_time_sec(absl::ToDoubleSeconds(event_start - ex_start));
  } else {
    event->clear_start_time_sec();
  }

  if (event->has_end_time()) {
    absl::Time event_end;
    ASSIGN_OR_RETURN(event_end, GetEventEndTime(*event));
    event->set_end_time_sec(absl::ToDoubleSeconds(event_end - ex_start));
  } else {
    event->clear_end_time_sec();
  }

  return OkStatus();
}

Status AddRawTextAnnotation(const Annotation& annotation, Example* example) {
  if (!annotation.has_events()) {
    ABSL_RAW_LOG(WARNING, "Annotation does not have any events.");
    return OkStatus();
  }

  std::vector<string> serialized_events;
  serialized_events.reserve(annotation.events().event_size());
  for (const auto& event : annotation.events().event()) {
    Event time_adjusted_event = event;
    RETURN_IF_ERROR(
        SetRelativeEventTimeSecFromExample(*example, &time_adjusted_event));
    serialized_events.push_back(time_adjusted_event.SerializeAsString());
  }
  auto annotations_ptr =
      GetFeatureValues<string>(kRawLabelEventsFeatureName, example);
  annotations_ptr->CopyFrom(
      {std::begin(serialized_events), std::end(serialized_events)});
  return OkStatus();
}

Status AddAnnotation(const Annotation& annotation, Example* example) {
  switch (annotation.type()) {
    case RAW_TEXT:
      return AddRawTextAnnotation(annotation, example);
    default:
      return Status(eeg_modelling::StatusCode::kInvalidArgument,
                    "Invalid type");
  }
}

void AddChannel(const string& group_name, const Segment::Channel& channel,
                Example* example) {
  // Add fields common to all channels.
  // Add sampling frequency and num_samples only if it's not been added.
  // We expect this to be the same for the same type of waveform in the segment,
  // and is checked in the call to IsSegmentValid().
  auto sampling_frequency_ptr = GetFeatureValues<float>(
      CreateFeatureName(group_name, kSamplingRateFieldName), example);
  if (sampling_frequency_ptr->empty()) {
    *sampling_frequency_ptr->Add() = channel.sampling_frequency_hz();
  }

  auto num_samples_ptr = GetFeatureValues<int64_t>(
      CreateFeatureName(group_name, kNumSamplesFieldName), example);
  if (num_samples_ptr->empty()) {
    *num_samples_ptr->Add() = channel.num_samples();
  }
  // Add fields specific to this channel.
  auto units_ptr = GetFeatureValues<string>(
      CreateFeatureName(group_name, channel.name(), kUnitsFieldName), example);
  *units_ptr->Add() = channel.physical_dimension();
  auto max_ptr = GetFeatureValues<float>(
      CreateFeatureName(group_name, channel.name(), kMaxFieldName), example);
  *max_ptr->Add() = channel.physical_max();
  auto min_ptr = GetFeatureValues<float>(
      CreateFeatureName(group_name, channel.name(), kMinFieldName), example);
  *min_ptr->Add() = channel.physical_min();
  auto samples_ptr = GetFeatureValues<float>(
      CreateFeatureName(group_name, channel.name(), kSamplesFieldName),
      example);
  samples_ptr->CopyFrom(channel.samples());
}

string GetGroupName(const Segment& segment) {
  switch (segment.data_type()) {
    case DATATYPE_EEG:
      return kEegGroupName;
    case DATATYPE_CTG:
      return kCtgGroupName;
    case DATATYPE_ECG:
      return kEcgGroupName;
    default:
      ABSL_RAW_LOG(WARNING, "Invalid data type : %d", segment.data_type());
  }
  return "";
}

Status IsChannelValid(const DataType& data_type,
                      const Segment::Channel& channel) {
  if (data_type == DATATYPE_EEG && !(channel.physical_dimension() == "uV" ||
                                     channel.physical_dimension() == "mV")) {
    return InvalidArgumentError(
        absl::StrCat("Unexpected channel with physical dimension : ",
                     channel.physical_dimension(), " name : ", channel.name()));
  }
  if (data_type == DATATYPE_CTG && channel.physical_dimension() != "nd" &&
      channel.physical_dimension() != "bpm") {
    return InvalidArgumentError(
        absl::StrCat("Unexpected channel with physical dimension : ",
                     channel.physical_dimension(), " name : ", channel.name()));
  }
  if (!channel.num_samples_consistent()) {
    return InvalidArgumentError(absl::StrCat(
        "Channel with inconsistent num samples : ", channel.name()));
  }
  return OkStatus();
}

Status IsSegmentValid(const Segment& segment) {
  float sampling_rate = 0.0f;
  bool found_sampling_rate = false;
  std::set<string> channel_names;
  for (const auto& channel : segment.channel()) {
    if (!IsChannelValid(segment.data_type(), channel).ok()) {
      continue;
    }
    if (!found_sampling_rate) {
      sampling_rate = channel.sampling_frequency_hz();
      found_sampling_rate = true;
    } else if (channel.sampling_frequency_hz() != sampling_rate) {
      ABSL_RAW_LOG(
          INFO, "Skipping channel with different frequency: %s, Filename: %s",
          channel.name().c_str(), segment.filename().c_str());
      continue;
    }
    if (channel_names.find(channel.name()) != channel_names.end()) {
      return InvalidArgumentError(absl::StrCat(
          "Duplicate channel name detected. Channel name: ", channel.name(),
          " Filename: ", segment.filename()));
    }
    channel_names.insert(channel.name());
  }
  return OkStatus();
}

}  // namespace
StatusOr<tensorflow::Example> GenerateExampleForSegment(
    const Segment& segment, const Annotations& annotations) {
  RETURN_IF_ERROR(IsSegmentValid(segment));

  // Create an example for the whole segment.
  Example example;
  // Add segment information.
  auto filename_ptr = GetFeatureValues<string>(kFilenameFeatureName, &example);
  *filename_ptr->Add() = segment.filename();
  auto session_id_ptr =
      GetFeatureValues<string>(kSessionIdFeatureName, &example);
  *session_id_ptr->Add() = segment.session_id();
  auto patient_id_str_ptr =
      GetFeatureValues<string>(kPatientIdStrFeatureName, &example);
  *patient_id_str_ptr->Add() = segment.patient_id();
  const int segment_duration_sec =
      segment.num_data_records() * segment.data_record_duration_sec();
  auto duration_ptr = GetFeatureValues<int64_t>(kDurationFeatureName, &example);
  *duration_ptr->Add() = segment_duration_sec;
  auto start_time_ptr =
      GetFeatureValues<string>(kStartTimeFeatureName, &example);
  *start_time_ptr->Add() = segment.start_time().SerializeAsString();
  const string group_name = GetGroupName(segment);
  // Add channel information and signals
  int num_channels = 0;
  // Currently, we put all channel under EegGroupName. In the future, we will
  // need to handle other types of channels.
  for (const auto& channel : segment.channel()) {
    if (IsChannelValid(segment.data_type(), channel).ok()) {
      AddChannel(group_name, channel, &example);
      ++num_channels;
    }
  }
  ABSL_RAW_LOG(INFO, "Added: %d channels", num_channels);
  auto num_channels_ptr =
      GetFeatureValues<int64_t>(kNumChannelsFeatureName, &example);
  *num_channels_ptr->Add() = num_channels;
  // Add group name
  auto group_name_ptr =
      GetFeatureValues<string>(kGroupNameFeatureName, &example);
  *group_name_ptr->Add() = group_name;
  // Add any annotations.
  for (int i = 0; i < annotations.annotation_size(); ++i) {
    const auto& annotation = annotations.annotation(i);
    if (AddAnnotation(annotation, &example) != OkStatus()) {
      ABSL_RAW_LOG(FATAL, "AddAnnotation failed.");
    }
  }
  // Add any patient info.
  const auto& patient_info = segment.patient_info();
  auto patient_id_ptr =
      GetFeatureValues<string>(kPatientIdFeatureName, &example);
  *patient_id_ptr->Add() = patient_info.patient_id();
  if (patient_info.has_gender()) {
    auto patient_gender_ptr =
        GetFeatureValues<int64_t>(kPatientGenderFeatureName, &example);
    *patient_gender_ptr->Add() = patient_info.gender();
  }
  if (patient_info.has_age()) {
    auto patient_age_ptr =
        GetFeatureValues<int64_t>(kPatientAgeFeatureName, &example);
    *patient_age_ptr->Add() = patient_info.age();
  }
  // Call to validate we have valid channel features.  Better to do now right
  // after generating the example, than later on in the pipeline.
  RETURN_IF_ERROR(HasValidChannelFeatures(example));
  return example;
}

}  // namespace eeg_modelling
