# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Simple utility for computing the entropy differences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

flags.DEFINE_string(
    "corpus", "",
    "Corpus path.")

flags.DEFINE_string(
    "pperp", "/tmp/ptest.perp",
    "Path to spoken perplexity.")

flags.DEFINE_string(
    "wperp", "/tmp/wtest.perp",
    "Path to written perplexity.")

FLAGS = flags.FLAGS


def main(unused_argv):
  with open(FLAGS.wperp) as s:
    for line in s:
      if "perplexity =" in line:
        wperp = float(line.split()[-1])
        break
  with open(FLAGS.pperp) as s:
    for line in s:
      if "perplexity =" in line:
        pperp = float(line.split()[-1])
        break
  went = math.log(wperp, 2)
  pent = math.log(pperp, 2)
  print("{} written entropy: {}".format(FLAGS.corpus, went))
  print("{} spoken entropy: {}".format(FLAGS.corpus, pent))
  print("{} w - p: {}".format(FLAGS.corpus, went - pent))


if __name__ == "__main__":
  app.run(main)
