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

#ifndef xxx_PREPROCESS_SCAM_FEATURIZE_H_
#define xxx_PREPROCESS_SCAM_FEATURIZE_H_

#include "xxx/flume.h"
#include "xxx/features.proto.h"

namespace research_biology {
namespace aptamers {

typedef flume::PCollection<string> SequenceCollection;
typedef flume::KV<string, research_scam::GenericFeatureVector>
    FeatureVectorEntry;
typedef flume::PTable<string, research_scam::GenericFeatureVector>
    FeatureVectorTable;

FeatureVectorTable FeaturizeSequences(const SequenceCollection &in);

}  // namespace aptamers
}  // namespace research_biology

#endif  // xxx_PREPROCESS_SCAM_FEATURIZE_H_
