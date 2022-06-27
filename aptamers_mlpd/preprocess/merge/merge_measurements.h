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

#ifndef xxx_PREPROCESS_MERGE_MERGE_MEASUREMENTS_H_
#define xxx_PREPROCESS_MERGE_MERGE_MEASUREMENTS_H_

#include <vector>

#include "pipeline/flume/public/flume.h"
#include "xxx/util/measurement.proto.h"

namespace research_biology {
namespace aptamers {

// MeasurementEntry is the type of values in a MeasurementTable.
typedef flume::KV<string, Measurement> MeasurementEntry;
typedef flume::PTable<string, Measurement> MeasurementTable;

// JoinEntry is the type of values in a joined table.
typedef flume::KV<string, flume::JoinResult> JoinEntry;
typedef flume::JoinTag<Measurement> JoinTag;
typedef flume::PTable<string, flume::JoinResult> JoinTable;

// Merge produces a single table containing all the entries from the joined
// table.
MeasurementTable MergeMeasurements(
    const std::vector<JoinTag> &tags, const JoinTable &in);

}  // namespace aptamers
}  // namespace research_biology

#endif  // xxx_PREPROCESS_MERGE_MERGE_MEASUREMENTS_H_
