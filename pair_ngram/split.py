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

"""Performs a format-agnostic 80/10/10 split assuming one example per line."""

import argparse
import contextlib
import logging
import time

import numpy


def main(args: argparse.Namespace) -> None:
  # Creates indices for splits.
  with open(args.input, "r") as source:
    n_samples = sum(1 for _ in source)
  # pylint: disable=logging-fstring-interpolation
  logging.info(f"Total set:\t{n_samples:,} lines")
  numpy.random.seed(args.seed)
  indices = numpy.random.permutation(n_samples)
  train_right = int(n_samples * 0.8)
  dev_right = int(n_samples * 0.9)
  # We don't explicitly create the `train_indices` set.
  logging.info(f"Train set:\t{train_right:,} lines")
  dev_indices = frozenset(indices[train_right:dev_right])
  logging.info(f"Dev set:\t{len(dev_indices):,} lines")
  test_indices = frozenset(indices[dev_right:])
  logging.info(f"Test set:\t{len(test_indices):,} lines")
  # Writes out the splits.
  with contextlib.ExitStack() as stack:
    source = stack.enter_context(open(args.input, "r"))
    train = stack.enter_context(open(args.train, "w"))
    dev = stack.enter_context(open(args.dev, "w"))
    test = stack.enter_context(open(args.test, "w"))
    for i, line in enumerate(source):
      if i in dev_indices:
        sink = dev
      elif i in test_indices:
        sink = test
      else:
        sink = train
      print(line.rstrip(), file=sink)


if __name__ == "__main__":
  logging.basicConfig(level="INFO", format="%(levelname)s:\t%(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--seed",
      type=int,
      default=time.time_ns(),
      help="random seed (default: current time)",
  )
  parser.add_argument("--input", required=True, help="path to input data")
  parser.add_argument(
      "--train", required=True, help="path for output training data")
  parser.add_argument(
      "--dev", required=True, help="path for output development data")
  parser.add_argument("--test", required=True, help="path for output test data")
  main(parser.parse_args())
