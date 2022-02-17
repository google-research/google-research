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

"""Computes error rate by comparing rows for exact matchs."""

import argparse


def main(args):
  error = 0
  total = 0
  with open(args.gold, "r") as gold_file, open(args.hypo, "r") as hypo_file:
    for gold_line, hypo_line in zip(gold_file, hypo_file):
      if gold_line != hypo_line:
        error += 1
      total += 1
  error_rate = 100 * error / total
  print(f"Error rate:\t{error_rate:.2f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gold", required=True, help="path to the gold data")
  parser.add_argument(
      "--hypo", required=True, help="path to the hypothesis data")
  main(parser.parse_args())
