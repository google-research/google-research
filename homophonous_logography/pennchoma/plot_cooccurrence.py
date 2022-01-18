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

"""Utility for plotting character cooccurrence matrices.

List of available seaborn palettes:
-----------------------------------
Also see: https://seaborn.pydata.org/tutorial/color_palettes.html
"""

import logging

from typing import Sequence

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

flags.DEFINE_string(
    "input_file", None,
    ("Path to the text file containing the correlations. The file is produced "
     "by the `penn_choma` tool."))

flags.DEFINE_string(
    "output_file", None,
    ("Output file for the resulting figure (based on the extension: PDF, PNG, "
     "etc.)."))

flags.DEFINE_string(
    "color_map", "vlag",
    ("Color map to use. Some examples: \"PuBu\", \"vlag\", "
     "\"viridis\", \"mako\"."))

flags.DEFINE_integer(
    "dpi", 700, "Dots per pixel (DPI) resolution for the final images.")

FLAGS = flags.FLAGS

_PAIR_PREFIX = "1:"


def _plot(df):
  """Plots the heatmap defined in the dataframe."""
  plt.rcParams["font.family"] = "sans-serif"
  plt.rcParams["font.sans-serif"] = [
      "Noto Sans CJK SC", "Noto Sans CJK KR", "sans-serif"]
  logging.info("Saving PDF to %s ...", FLAGS.output_file)
  fig = plt.figure()
  sns.set(font_scale=0.7)
  ax = sns.heatmap(df, fmt=".2g", cmap=FLAGS.color_map,
                   cbar_kws=dict(use_gridspec=False,
                                 orientation="horizontal",
                                 pad=0.02,
                                 shrink=0.60),
                   square=True, vmin=-0.3, vmax=1.0, center=0.4,
                   yticklabels=False, xticklabels=False)
  ax.set_aspect("equal")
  fig.savefig(FLAGS.output_file, bbox_inches="tight",
              pad_inches=0, dpi=FLAGS.dpi)


def _process_input_file():
  """Processes input file and plots the results."""
  logging.info("Reading %s ...", FLAGS.input_file)

  # Parse the file into intermediate data structures.
  corr_pairs = []
  char_to_id = {}
  num_chars = 0
  with open(FLAGS.input_file, mode="r", encoding="utf8") as f:
    lines = [line.rstrip() for line in f.readlines()]
    for line in lines:
      if not line.startswith(_PAIR_PREFIX):
        continue
      toks = line.replace(_PAIR_PREFIX, "").split("\t")
      if len(toks) != 2:
        raise ValueError("Expected two tokens in {}".format(toks))
      char_toks = toks[0].split(",")
      if len(char_toks) != 2:
        raise ValueError("Expected two char tokens in {}".format(char_toks))

      if char_toks[0] not in char_to_id:
        from_char_id = num_chars
        char_to_id[char_toks[0]] = from_char_id
        num_chars += 1
      else:
        from_char_id = char_to_id[char_toks[0]]
      if char_toks[1] not in char_to_id:
        to_char_id = num_chars
        char_to_id[char_toks[1]] = to_char_id
        num_chars += 1
      else:
        to_char_id = char_to_id[char_toks[1]]
      corr_pairs.append(((from_char_id, to_char_id), float(toks[1])))
  sqrt_num_chars = int(np.sqrt(len(corr_pairs)))
  assert num_chars == sqrt_num_chars
  logging.info("Read %d correlation pairs, %d characters.", len(corr_pairs),
               num_chars)

  # Convert raw containers to Pandas data frame and plot.
  sorted_by_id = [char for char, _ in sorted(
      char_to_id.items(), key=lambda item: item[1])]
  corr = np.zeros((num_chars, num_chars))
  for chars, r in corr_pairs:
    corr[chars[0], chars[1]] = r
  corr_df = pd.DataFrame(corr, columns=sorted_by_id, index=sorted_by_id)
  corr_df.info()
  _plot(corr_df)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.input_file:
    raise app.UsageError("Specify --input_file")
  if not FLAGS.output_file:
    raise app.UsageError("Specify --output_file")
  _process_input_file()


if __name__ == "__main__":
  app.run(main)
