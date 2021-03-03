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

// A FlumeC++ pipeline that transforms measurements into ScaM inputs.

#include <gflags/gflags.h>

#include "xxx/flume/public/flume.h"
#include "xxx/flume/public/recordio.h"
#include "xxx/flume/public/sstableio.h"
#include "xxx/preprocess/scam_featurize.h"

DEFINE_FLAG(string, input_file, "", "A RecordIO of DNA sequences.");
DEFINE_FLAG(string, output_file, "",
            "An SSTable of N-gram ScaM GFVs keyed by DNA sequence.");

namespace research_biology {
namespace aptamers {

// Run the ScaM featurization on the input file and write to the output file.
void Run(const string& input_file, const string& output_file) {
  // Initialize Flume.
  flume::Flume flume;

  SequenceCollection input = SequenceCollection::Read(
      "ReadSequences",
      flume::RecordIOFile::Source(input_file, flume::Strings()));

  // Transform into ScaM GFV features.
  FeatureVectorTable scam_features = FeaturizeSequences(input);

  // Write to the specified output file.
  flume::PSink<FeatureVectorEntry> output =
      flume::SSTableSink<string, research_scam::GenericFeatureVector>(
          output_file, flume::Strings(),
          flume::Protos<research_scam::GenericFeatureVector>());

  scam_features.Write("WriteFeatures", output);
}

}  // namespace aptamers
}  // namespace research_biology

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (base::GetFlag(FLAGS_input_file).empty()) {
    LOG(ERROR) << "--input_file is required";
    return 1;
  }
  if (base::GetFlag(FLAGS_output_file).empty()) {
    LOG(ERROR) << "--output_file is required";
    return 1;
  }

  research_biology::aptamers::Run(base::GetFlag(FLAGS_input_file),
                                  base::GetFlag(FLAGS_output_file));

  return 0;
}
