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

"""Converts a .json file to a series of flags and prints them.

Used to generate flags to call a script from a json config file.
"""
import json
import pathlib
from typing import List

from absl import app


def main(argv):
  path = pathlib.Path(argv[1])

  try:
    with open(path) as fin:
      args = json.load(fin)
  except FileNotFoundError:
    # Tensorflow takes a while to load, we don't unless we have to
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    with tf.io.gfile.GFile(path) as fin:
      args = json.loads(fin.read())

  print(" ".join(f"--{k}={v}" for k, v in args.items()))


if __name__ == "__main__":
  app.run(main)
