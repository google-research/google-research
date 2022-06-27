# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
r"""Run the Rust ELO calculator on all original images.

Filters out evaluations for each of the original images
and runs the Rust ELO calulator on each original image
to create ELO results for all distortions per original.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input_json",
                    dest="input_json",
                    help="Read evaluations from this file")
parser.add_argument("-o",
                    "--output_directory",
                    dest="output_directory",
                    help="Write the ELO results in this directory")

args = parser.parse_args()

if args.input_json is None or args.output_directory is None:
  parser.print_help()
  sys.exit(1)

with open(args.input_json) as f:
  evaluations = json.loads(f.read())

originals = {}
for evaluation in evaluations:
  if evaluation["image"] not in originals:
    originals[evaluation["image"]] = []
  originals[evaluation["image"]].append(evaluation)

for original in originals:
  with tempfile.NamedTemporaryFile() as filtered_file:
    # Repeat the matches to get more stable results.
    # Arbitrarily all matches get ~ original number of matches for ELO
    # calculation purposes.
    repeated_matches = (originals[original] *
                        int(len(evaluations) / len(originals[original])))
    filtered_file.write(json.dumps(repeated_matches).encode("utf-8"))
    filtered_file.flush()
    p = subprocess.Popen(
        f"cargo run --release {filtered_file.name}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True)
    out, _ = p.communicate()
    outdir = os.path.join(args.output_directory, original)
    os.makedirs(outdir, exist_ok=True)
    outfilename = os.path.join(outdir, "elos")
    with open(outfilename, "w") as outfile:
      outfile.write(out.decode("utf-8"))
    print(f"Saved ELOs for all distortions on {original} in {outfilename}")
