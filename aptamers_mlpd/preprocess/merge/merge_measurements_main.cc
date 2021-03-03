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

// A FlumeC++ pipeline that merges measurements tables.

#include <gflags/gflags.h>

#include "pipeline/flume/public/flume.h"
#include "pipeline/flume/public/sstableio.h"
#include "xxx/preprocess/merge/merge_measurements.h"

DEFINE_string(output_filename,
              "",
              "output filename for sstable");

namespace research_biology {
namespace aptamers {

// Run the merge on the input files, specified by <argc, argv>.
void Run(int argc, char **argv) {
  // Initialize Flume now, and automatically clean up at the end of
  // this scope.
  flume::Flume flume;

  // Open the sstable inputs and construct the join table.
  std::vector<JoinTag> tags;
  flume::JoinOp<string> join("Merge");
  for (int i = 1; i < argc; i++) {
    const string key(argv[i]);
    MeasurementTable input =
        MeasurementTable::Read(
            key,
            flume::SSTableSource<string, Measurement>(
                key,
                flume::Strings(),
                flume::Protos<Measurement>()));

    JoinTag tag(key);
    tags.push_back(tag);
    join = join.With(flume::JoinArg::Of(tag, input));
  }
  JoinTable table = join.Join();

  // Transform the collection.
  MeasurementTable merged = MergeMeasurements(tags, table);

  // Write to the specified output file.
  flume::PSink<MeasurementEntry> output =
      flume::SSTableSink<string, Measurement>(
              FLAGS_output_filename,
              flume::Strings(),
              flume::Protos<Measurement>());

  // Write it out.
  merged.Write("Write", output);
}

}  // namespace aptamers
}  // namespace research_biology

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (base::GetFlag(FLAGS_output_filename).empty()) {
    LOG(ERROR) << "--output_filename is required";
    return 1;
  }

  // Copy the file.
  research_biology::aptamers::Run(argc, argv);

  return 0;
}
