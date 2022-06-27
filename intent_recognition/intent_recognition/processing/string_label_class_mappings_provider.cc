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

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "intent_recognition/annotated_recording_collection_utils.h"
#include "intent_recognition/processing/class_mappings_provider.h"
#include "intent_recognition/processing/class_mappings_provider_helpers.h"
#include "intent_recognition/processing/string_label_class_map.pb.h"
#include "intent_recognition/processing/string_label_class_mappings_provider_helpers.h"

namespace ambient_sensing {
namespace {

void NormalizeMapping(StringLabelClassMap* mapping) {
  for (LabelClassMapEntry& entry : *mapping->mutable_entry()) {
    for (std::string& substring :
         *entry.mutable_identifier()->mutable_include_substrings()) {
      substring = absl::AsciiStrToLower(substring);
    }
    for (std::string& substring :
         *entry.mutable_identifier()->mutable_exclude_substrings()) {
      substring = absl::AsciiStrToLower(substring);
    }
  }
}

// Version of ClassMappingsProvider that provides mapppings for string labels.
class StringLabelClassMappingsProvider : public ClassMappingsProvider {
 public:
  struct MappingInfo {
    StringLabelClassMap label_class_map;
    absl::flat_hash_map<std::string, int> prediction_to_id_map;
  };

  explicit StringLabelClassMappingsProvider(
      const std::map<std::string, MappingInfo>& mapping_name_to_info)
      : mapping_name_to_info_(mapping_name_to_info) {}

  absl::Status AddClassMappings(
      AnnotatedRecordingCollection* arc) const override {
    AnnotationGroup* annotation_group = arc->add_annotation_group();
    annotation_group->mutable_metadata()->set_group_type(
        AnnotationGroupMetadata::GROUND_TRUTH);
    annotation_group->mutable_metadata()->set_group_descriptor(
        std::string(kStringLabelMappingGroupDescriptor));
    for (const auto& [mapping_name, mapping_info] : mapping_name_to_info_) {
      absl::StatusOr<std::string> target_class_or =
          SubstringLabelMapper(arc->recording_collection()
                                   .metadata()
                                   .mobile_collection_metadata()
                                   .session_activity(),
                               mapping_info.label_class_map);
      if (!target_class_or.ok()) return target_class_or.status();
      std::string target_class = target_class_or.value();
      if (mapping_info.prediction_to_id_map.contains(target_class)) {
        AnnotationSequence* annotation_sequence =
            annotation_group->add_annotation_sequence();
        annotation_sequence->mutable_metadata()->set_annotation_type(
            AnnotationMetadata::CLASS_LABEL);
        annotation_sequence->mutable_metadata()
            ->mutable_source_details()
            ->mutable_identifier()
            ->set_name(mapping_name);
        Annotation* annotation = annotation_sequence->add_annotation();
        AnnotationLabel* label = annotation->add_label();

        label->set_name(target_class);
        label->set_id(mapping_info.prediction_to_id_map.at(target_class));
      }
    }
    return absl::OkStatus();
  }

 private:
  std::map<std::string, MappingInfo> mapping_name_to_info_;
};

}  // namespace

absl::StatusOr<std::string> SubstringLabelMapper(
    const std::string& label, const StringLabelClassMap& label_class_map) {
  std::string lowercase_label = absl::AsciiStrToLower(label);
  for (const auto& mapping : label_class_map.entry()) {
    for (const auto& substring : mapping.identifier().include_substrings()) {
      if (!absl::StrContains(lowercase_label, substring)) {
        goto NextIter;
      }
    }
    for (const auto& substring : mapping.identifier().exclude_substrings()) {
      if (absl::StrContains(lowercase_label, substring)) {
        goto NextIter;
      }
    }
    // If we reach here, then we found the prediction of the label.
    return mapping.target_class_string();
  NextIter : {}
  }
  return absl::NotFoundError(
      absl::Substitute("$0: Couldn't find a label mapping for $1",
                       label_class_map.name(), label));
}

absl::StatusOr<std::unique_ptr<ClassMappingsProvider>>
ClassMappingsProvider::NewStringLabelProvider(
    const std::vector<std::string>& label_mapping_files) {
  std::map<std::string, StringLabelClassMappingsProvider::MappingInfo>
      mapping_name_to_info;
  // Extract the information for each label mapping file.
  for (const auto& label_class_map_file_path : label_mapping_files) {
    StringLabelClassMappingsProvider::MappingInfo mapping_info;

    bool initialized_mapping = false;
    std::ifstream input_stream(label_class_map_file_path);
    std::string proto_string;
    if (input_stream.is_open()) {
      proto_string = std::string((std::istreambuf_iterator<char>(input_stream)),
                                 (std::istreambuf_iterator<char>()));

      if (!google::protobuf::TextFormat::ParseFromString(proto_string,
                                               &mapping_info.label_class_map)) {
        return absl::InvalidArgumentError(
            "Failed to parse text proto to string");
      }
      initialized_mapping = true;
    }


    if (!initialized_mapping) {
      return absl::InvalidArgumentError("Unable to open file");
    }

    NormalizeMapping(&mapping_info.label_class_map);
    mapping_info.prediction_to_id_map =
        CreatePredictionToIdMap(mapping_info.label_class_map);
    CHECK(mapping_name_to_info
              .try_emplace(mapping_info.label_class_map.name(), mapping_info)
              .second)
        << "Cannot use the same label class map twice";
  }

  return absl::make_unique<StringLabelClassMappingsProvider>(
      mapping_name_to_info);
}

}  // namespace ambient_sensing
