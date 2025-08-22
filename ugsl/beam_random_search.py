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

"""Invokes a random-search of hyperparameters using (distributed) beam."""

from collections.abc import Sequence

from absl import app
import apache_beam as beam


from ugsl import random_search_lib
from ugsl import trainer


def pipeline(root):
  configs = random_search_lib.sample_random_configs()
  return (
      root
      | beam.Create(configs)
      | beam.Reshuffle()
      | beam.Map(trainer.train))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  p = beam.Pipeline()
  pipeline(p)
  p.run()


if __name__ == "__main__":
  app.run(main)
