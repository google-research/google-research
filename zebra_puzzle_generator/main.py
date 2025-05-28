# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Main file for Zebra Puzzle Generation."""

from absl import app
from absl import logging
import ml_collections

from zebra_puzzle_generator import zebra_puzzle_generator


logging.set_verbosity(logging.INFO)


def get_config():
  """Get the default puzzle generation hyperparameter configuration.

  Returns:
    A ConfigDict object.
  """

  config = ml_collections.ConfigDict()
  config.n = 5
  config.m1 = 5
  return config


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  cfgs = get_config()
  logging.info(cfgs)

  puzzle_generator = zebra_puzzle_generator.RandomZebraPuzzleGenerator(
      n=cfgs.n,
      m1=cfgs.m1,
      m2=1,
  )
  puzzle, ground_truth, _, detailed_solution, ordered_fills = (
      puzzle_generator.generate_symbolic_zebra_puzzle()
  )
  return puzzle, ground_truth, detailed_solution, ordered_fills


if __name__ == '__main__':
  app.run(main)
