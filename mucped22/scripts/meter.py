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
r"""Run a set of objective visual perceptual metrics.

Runs a defined set of objective visual perceptual metrics
on all evaluations in the evaluation file. Will generate
.PFM files for all crops if missing.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument("-ioj",
                    "--input_output_json",
                    dest="input_output_json",
                    help="Read evaluations from this file")
parser.add_argument("-id",
                    "--input_directory",
                    dest="input_directory",
                    help="Read compared images from this directory")
parser.add_argument("-od",
                    "--output_directory",
                    dest="output_directory",
                    help="Write intermediate version of the " +
                    "images to this directory")
parser.add_argument("-md",
                    "--metrics_directory",
                    dest="metrics_directory",
                    help="Directory where metrics binaries can be found " +
                    "(typically libjxl/tools/benchmark/metrics " +
                    "inside a checkout of https://github.com/libjxl/libjxl)")
parser.add_argument("-ed",
                    "--elo_directory",
                    dest="elo_directory",
                    help="Directory where ELO data can be found")

args = parser.parse_args()

if args.input_output_json is None or args.input_directory is None or args.output_directory is None or args.metrics_directory is None or args.elo_directory is None:
  parser.print_help()
  sys.exit(1)

elos_by_image = {}


def elo(image, distortion):
  if image not in elos_by_image:
    elos_by_image[image] = {}
  if distortion not in elos_by_image[image]:
    with open(f"{args.elo_directory}/{image}/elos") as elo_file:
      r = csv.reader(elo_file, delimiter=" ")
      for row in r:
        elos_by_image[image][row[0]] = float(row[1])
  return elos_by_image[image][distortion]

with open(args.input_output_json) as input_file:
  evaluations = json.loads(input_file.read())


def spopen(cmd):
  p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, close_fds=True)
  out, _ = p.communicate()
  return out.decode("utf-8")


def butteraugli():
  def run_metric(original, distortion):
    output = spopen(
        f"butteraugli_main '{original}' '{distortion}' --pnorm 6").split("\n")
    return {"butteraugli_max": float(output[0]),
            "butteraugli_6": float(output[1].split(":")[1])}
  return {"f": run_metric,
          "metrics": ["butteraugli_max", "butteraugli_6"]}


def jxl_metric_func(fname):
  def run_metric(original, distortion):
    orig_pfm = pfm_of(original)
    dist_pfm = pfm_of(distortion)
    with tempfile.NamedTemporaryFile() as result_file:
      cmd = f"{args.metrics_directory}/{fname}.sh {orig_pfm} {dist_pfm} {result_file.name} 2> /dev/null"
      if os.system(cmd) != 0:
        raise Exception(f"Unable to run {cmd}")
      return {fname: float(result_file.read().decode("utf-8"))}
  return {"f": run_metric,
          "metrics": [fname]}


def pfm_of(png):
  parts = png.split(".")
  pfm_file = f"{parts[0]}.pfm"
  if not os.path.exists(pfm_file):
    cmd = f"convert {png} -colorspace RGB {pfm_file}"
    if os.system(cmd) != 0:
      raise Exception(f"Unable to run {cmd}")
  return pfm_file


metrics = [butteraugli(),
           # jxl_metric_func("lpips-rgb"),  # tries to use CUDA even when
           # it doesn't exist
           jxl_metric_func("fsim-y"),
           jxl_metric_func("fsim-rgb"),
           jxl_metric_func("msssim-y"),
           jxl_metric_func("nlpd-y"),
           jxl_metric_func("ssimulacra"),
           # jxl_metric_func("vmaf"),  # silently dies due to missing from
           # libjxl/third_party
           # jxl_metric_func("hdrvdp"),  # convert-im6.q16: invalid argument for
           # option `-evaluate': --path @
           # error/convert.c/ConvertImageCommand/1461.
          ]
for eval_idx, evaluation in enumerate(evaluations):
  changed = False
  for metric_idx, metric_data in enumerate(metrics):
    sys.stdout.write(f"\r{eval_idx * len(metrics) + metric_idx}/" +
                     "{len(evaluations) * len(metrics)}    ")
    # pylint: disable=cell-var-from-loop
    if not all(found
               for found in map(lambda m: f"greater_{m}" in evaluation,
                                metric_data["metrics"])):
      changed = True
      try:
        for name, result in metric_data["f"](
            evaluation["original_file"], evaluation["greater_file"]).items():
          evaluation[f"greater_{name}"] = result
      except:  # pylint: disable=bare-except
        print(f"\nFailed to run {metric_data['metrics']}!\n")
    if not all(found
               for found in map(lambda m: f"lesser_{m}" in evaluation,
                                metric_data["metrics"])):
      changed = True
      try:
        for name, result in metric_data["f"](
            evaluation["original_file"], evaluation["lesser_file"]).items():
          evaluation[f"lesser_{name}"] = result
      except:  # pylint: disable=bare-except
        print(f"\nFailed to run {metric_data['metrics']}!\n")
  evaluation["greater_elo"] = elo(evaluation["image"], evaluation["greater"])
  evaluation["lesser_elo"] = elo(evaluation["image"], evaluation["lesser"])
  evaluations[eval_idx] = evaluation
  if changed and eval_idx % 1 == 0:
    with open(args.input_output_json, "w") as output_file:
      output_file.write(json.dumps(evaluations))

print("")

with open(args.input_output_json, "w") as f:
  f.write(json.dumps(evaluations))


