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

#ifndef INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_H_
#define INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "intent_recognition/annotated_recording_collection.pb.h"

namespace ambient_sensing {

// Class to provide label mappings to AnnotatedRecordingCollection samples.
class ClassMappingsProvider {
 public:
  virtual ~ClassMappingsProvider() = default;

  // Add label to class mappings to AnnotatedRecordingCollection. If label fails
  // to be mapped, returns a non-OK status.
  virtual absl::Status AddClassMappings(
      AnnotatedRecordingCollection* arc) const = 0;

  // Creates a ClassMappingsProvider that provides mapppings for string labels.
  static absl::StatusOr<std::unique_ptr<ClassMappingsProvider>>
  NewStringLabelProvider(const std::vector<std::string>& label_mapping_files);

};

}  // namespace ambient_sensing

#endif  // INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_H_
