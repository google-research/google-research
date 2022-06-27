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

#ifndef PHOTOS_VISION_AMBIENT_LOLIGO_PROCESSING_STRING_LABEL_CLASS_MAPPINGS_PROVIDER_HELPERS_H_
#define PHOTOS_VISION_AMBIENT_LOLIGO_PROCESSING_STRING_LABEL_CLASS_MAPPINGS_PROVIDER_HELPERS_H_

#include "absl/status/statusor.h"
#include "intent_recognition/processing/string_label_class_map.pb.h"

namespace ambient_sensing {

// A function to return the prediction string for a label of a given label type
// based on the provided label mapping. The mapping should be normalized to
// lower case, since the label will be lower-cased. If a prediction can be found
// for a label, this function should return the predicition target class string
// for that prediction. Otherwise, it will return a not found error.
absl::StatusOr<std::string> SubstringLabelMapper(
    const std::string& label, const StringLabelClassMap& label_class_map);

}  // namespace ambient_sensing

#endif  // PHOTOS_VISION_AMBIENT_LOLIGO_PROCESSING_STRING_LABEL_CLASS_MAPPINGS_PROVIDER_HELPERS_H_
