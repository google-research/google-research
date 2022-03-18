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
r"""Ensure existence of crops in evaluation file.

Generates crops for all evaluations in the input file
that doesn't already exist and outputs a new evaluation
file with the crop paths added

Example:
  python3 scripts/crop.py -i evaluations.json -o evaluations.json \
    -id ~/Downloads/kolasnittar -od ~/tmp/kolasnittar_crops
"""

import argparse
import hashlib
import json
import os
import sys

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input_json",
                    dest="input_json",
                    help="Read evaluations from this file")
parser.add_argument("-id",
                    "--input_directory",
                    dest="input_directory",
                    help="Read compared images from this directory")
parser.add_argument("-o",
                    "--output_json",
                    dest="output_json",
                    help="Write crop paths and evaluations to this file")
parser.add_argument("-od",
                    "--output_directory",
                    dest="output_directory",
                    help="Write cropped version of the images to this " +
                    "directory")

args = parser.parse_args()

if args.input_json is None or args.input_directory is None or args.output_json is None or args.output_directory is None:
  parser.print_help()
  sys.exit(1)

with open(args.input_json) as f:
  evaluations = json.loads(f.read())

out_data = []

for idx, evaluation in enumerate(evaluations):
  sys.stdout.write(f"\r{idx}/{len(evaluations)}    ")
  im = Image.open(f"{args.input_directory}/originals/{evaluation['image']}")
  evaluation["image_dims"] = list(im.size)

  def _crop(directory):
    """Crops the evaluated file in the given directory.

    Args:
      directory: Directory where the file is found.
    Returns:
      The file name of the cropped file.
    Raises:
      Exception: if the crop command returns != 0.
    """
    crop_command = (f"convert '{args.input_directory}/{directory}/" +
                    "{evaluation['image']}' -crop '{evaluation['crop'][2]}x" +
                    "{evaluation['crop'][3]}+{evaluation['crop'][0]}+" +
                    "{evaluation['crop'][1]}'")
    # pylint: disable=cell-var-from-loop
    name_hash = hashlib.sha256(
        f"{directory}{evaluation['image']}{evaluation['crop']}".encode(
            "utf-8")).hexdigest()
    output_file = f"{args.output_directory}/{name_hash}.png"
    if not os.path.exists(output_file):
      crop_command = f"{crop_command} '{output_file}'"
      if os.system(crop_command) != 0:
        raise Exception(f"Unable to execute {crop_command}!")
    return output_file
  evaluation["greater_file"] = _crop(evaluation["greater"])
  evaluation["lesser_file"] = _crop(evaluation["lesser"])
  evaluation["original_file"] = _crop("originals")
  out_data.append(evaluation)
print("")

with open(args.output_json, "w") as f:
  f.write(json.dumps(out_data))

