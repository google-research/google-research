// Copyright 2025 The Google Research Authors.
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

#ifndef INTENT_RECOGNITION_CONVERSION_CONVERT_ANNOTATED_RECORDING_COLLECTION_TO_SEQUENCE_EXAMPLE_H_
#define INTENT_RECOGNITION_CONVERSION_CONVERT_ANNOTATED_RECORDING_COLLECTION_TO_SEQUENCE_EXAMPLE_H_

#include "absl/status/statusor.h"
#include "intent_recognition/annotated_recording_collection.pb.h"
#include "tensorflow/core/example/example.pb.h"


namespace ambient_sensing {

absl::StatusOr<tensorflow::SequenceExample>
ConvertAnnotatedRecordingCollectionToSequenceExample(
    bool flatten, const std::string& resource_name,
    const AnnotatedRecordingCollection& input
);

}  // namespace ambient_sensing

#endif  // INTENT_RECOGNITION_CONVERSION_CONVERT_ANNOTATED_RECORDING_COLLECTION_TO_SEQUENCE_EXAMPLE_H_
