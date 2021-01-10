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



#include "scann/metadata/metadata_getter.h"

#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

Status UntypedMetadataGetter::AppendMetadata(const GenericFeatureVector& gfv) {
  return OkStatus();
}

bool UntypedMetadataGetter::needs_dataset() const { return true; }

Status UntypedMetadataGetter::UpdateMetadata(DatapointIndex idx,
                                             const GenericFeatureVector& gfv) {
  return UnimplementedError("UpdateMetadata not implemented by default.");
}

Status UntypedMetadataGetter::RemoveMetadata(DatapointIndex removed_idx) {
  return UnimplementedError("UpdateMetadata not implemented by default.");
}

UntypedMetadataGetter::~UntypedMetadataGetter() {}

}  // namespace research_scann
