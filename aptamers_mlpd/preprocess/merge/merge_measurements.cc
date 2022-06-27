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

#include "xxx/merge/merge_measurements.h"

#include "glog/logging.h"
#include "pipeline/flume/public/flume.h"
#include "absl/strings/string_view.h"

namespace research_biology {
namespace aptamers {

namespace {

// MergeFn is the function that merges multiple MeasurementEntry values
// into a single one by taking the union of the counts.
//
// This is a stateful operation because we need to the join tags
// for accessing the fields in a JoinResult.
class MergeFn : public flume::DoFn<JoinEntry, MeasurementEntry> {
 public:
  explicit MergeFn(const std::vector<JoinTag> &tags)
      : tags_(tags) {}

  // Do performs the merge operation.
  void Do(const JoinEntry &in,
          const flume::EmitFn<MeasurementEntry> &emit_fn) override {
    const flume::JoinResult &value = in.value();

    // Combine the counts values.  We assume that the columns are disjoint,
    // meaning that the union never encounters the same map key in multiple
    // input tables.
    Measurement result;
    proto2::Map<int32, int32> *counts = result.mutable_counts();
    for (const JoinTag &tag : tags_) {
      flume::Stream<Measurement> vs = value.Get(tag);
      for (; !vs.at_end(); ++vs) {
        const Measurement m = *vs;
        for (const auto &value : m.counts()) {
          const std::pair<proto2::Map<int32, int32>::iterator, bool> p =
              counts->insert(value);
          if (!p.second) {
            LOG(ERROR) << "Duplicate key: " << value.first;
          }
        }
      }
    }
    emit_fn.Emit(MeasurementEntry(in.key(), result));
  }

  // Encode the state, for saving an instance of this class.
  void EncodeState(string *encoded_state) const override {
    std::vector<string> keys;
    for (const auto &tag : tags_) {
      keys.push_back(tag.id());
    }
    flume::VectorsOf(flume::Strings()).Encode(keys, encoded_state);
  }

  // Recreate an instance of this class from a copy of the state.
  static MergeFn *DecodeState(absl::string_view encoded_state) {
    std::vector<string> keys;
    CHECK(flume::VectorsOf(flume::Strings()).Decode(encoded_state, &keys));
    std::vector<JoinTag> tags;
    for (const string &key : keys) {
      tags.push_back(JoinTag(key));
    }
    return new MergeFn(tags);
  }

 private:
  const std::vector<JoinTag> tags_;

  REGISTER_AS_STATEFUL_FN(MergeFn);
};

}  // namespace

// Perform the join.
MeasurementTable MergeMeasurements(
    const std::vector<JoinTag> &keys, const JoinTable &in) {
  return in.ParDo("merge", new MergeFn(keys));
}

}  // namespace aptamers
}  // namespace research_biology
